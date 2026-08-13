#include "FieldSolverADI.h"
#include "Field.h"
#include "Beam.h"
#include "SimdBatch.h"

namespace {

// Contiguous complex spans: one Eigen expression per grid region replaces the
// hand-written batch loop + scalar remainder pairs (Eigen vectorizes the body
// and handles the tail itself).
using cspan = Eigen::Map<Eigen::ArrayXcd>;
using ccspan = Eigen::Map<const Eigen::ArrayXcd>;

// Full forward/backward tridiagonal recurrence for B row-blocks
// (B * cbatch_width rows) starting at `row`. The recurrence runs along the
// contiguous (x) dimension, so lanes hold cbatch_width independent rows
// instead; the recurrence values stay in registers across k. B > 1 advances
// several row-blocks together so their serial mul-add chains overlap
// (single-chain throughput is latency-bound).
//
// The gather/scatter bounce through the aligned lane buffer is deliberate: a
// strided Eigen Map has no packet access, which would drop every complex
// multiply onto the (much slower) scalar std::complex path.
template <int B>
void tridagxRows(complex<double> *up, const complex<double> *rp,
                 const vector<complex<double>> &c, const vector<complex<double>> &cbet,
                 const vector<complex<double>> &cwet, int n, int row)
{
    alignas(64) complex<double> lane[cbatch_width];
    auto gather = [&](const complex<double> *p, int base, int k) -> cbatch {
        for (int l = 0; l < cbatch_width; l++) {
            lane[l] = p[base + l * n + k];
        }
        return cbatch::MapAligned(lane);
    };
    auto scatter = [&](const cbatch &b, int base, int k) {
        cbatch::MapAligned(lane) = b;
        for (int l = 0; l < cbatch_width; l++) {
            up[base + l * n + k] = lane[l];
        }
    };

    int base[B];
    cbatch uk[B];
    for (int b = 0; b < B; b++) {
        base[b] = (row + b * cbatch_width) * n;
        uk[b] = gather(rp, base[b], 0) * cbet[0];
        scatter(uk[b], base[b], 0);
    }
    for (int k = 1; k < n; k++) {
        for (int b = 0; b < B; b++) {
            uk[b] = (gather(rp, base[b], k) - c[k] * uk[b]) * cbet[k];
            scatter(uk[b], base[b], k);
        }
    }
    for (int k = n - 2; k >= 0; k--) {
        for (int b = 0; b < B; b++) {
            uk[b] = gather(up, base[b], k) - cwet[k + 1] * uk[b];
            scatter(uk[b], base[b], k);
        }
    }
}

} // namespace


FieldSolverADI::~FieldSolverADI() = default;

void FieldSolverADI::advance(double delz, Field *field, Beam *beam, Undulator *und) {

    for (unsigned long ii = 0; ii < field->field.size(); ii++) {  // ii is index for the beam

        field->constructSource(crsource, beam, und, delz, ii);

        unsigned long i = (ii + field->first) % field->field.size();           // index for the field
        this->ADI(field->field[i]);
    }
}


void FieldSolverADI::ADI(vector<complex<double> > &crfield)
{
  const int n = static_cast<int>(ngrid);
  const complex<double> *fld = crfield.data();
  const complex<double> *src = crsource.data();
  complex<double> *rp = r.data();

  // r = source + field + cstep*(neighbors - field - field); boundary
  // rows/columns see only their single existing neighbor. The doubled field
  // term is spelled -F-F rather than -2.0*F: a double*complex node has no
  // Eigen packet support ("TODO vectorize mixed product") and would drop the
  // whole expression onto the scalar std::complex path.
  auto F = [&](int off, int len) { return ccspan(fld + off, len); };
  auto S = [&](int off, int len) { return ccspan(src + off, len); };

  // implicit direction in x: neighbors live at +-ngrid, so the first row, the
  // contiguous interior and the last row are one span each
  const int mid = n * (n - 2);
  const int lastrow = n * (n - 1);
  cspan(rp, n) = S(0, n) + F(0, n) + cstep * (F(n, n) - F(0, n) - F(0, n));
  cspan(rp + n, mid) = S(n, mid) + F(n, mid) +
                       cstep * (F(2 * n, mid) + F(0, mid) - F(n, mid) - F(n, mid));
  cspan(rp + lastrow, n) = S(lastrow, n) + F(lastrow, n) +
                           cstep * (F(lastrow - n, n) - F(lastrow, n) - F(lastrow, n));

  // solve tridiagonal system in x
  this->tridagx(crfield);

  // implicit direction in y: neighbors live at +-1 within each contiguous row
  for (int ix = 0; ix < n * n; ix += n) {
    rp[ix] = src[ix] + fld[ix] + cstep * (fld[ix + 1] - 2.0 * fld[ix]);
    cspan(rp + ix + 1, n - 2) =
        S(ix + 1, n - 2) + F(ix + 1, n - 2) +
        cstep * (F(ix + 2, n - 2) + F(ix, n - 2) - F(ix + 1, n - 2) - F(ix + 1, n - 2));
    const int last = ix + n - 1;
    rp[last] = src[last] + fld[last] + cstep * (fld[last - 1] - 2.0 * fld[last]);
  }

  // solve tridiagonal system in y
  this->tridagy(crfield);

}


void FieldSolverADI::tridagx(vector<complex<double > > &u) {
    const int n = static_cast<int>(ngrid);
    complex<double> *up = u.data();
    const complex<double> *rp = r.data();

    int row = 0;
    for (; row + 2 * cbatch_width <= n; row += 2 * cbatch_width) {
        tridagxRows<2>(up, rp, c, cbet, cwet, n, row);
    }
    for (; row + cbatch_width <= n; row += cbatch_width) {
        tridagxRows<1>(up, rp, c, cbet, cwet, n, row);
    }
    // scalar tail: fewer rows than one batch. The backward sweep updates u in
    // place, so re-running an overlapping block would be wrong - this stays.
    for (; row < n; row++) {
        const int i = row * n;
        up[i] = rp[i] * cbet[0];
        for (int k = 1; k < n; k++) {
            up[k + i] = (rp[k + i] - c[k] * up[k + i - 1]) * cbet[k];
        }
        for (int k = n - 2; k >= 0; k--) {
            up[k + i] -= cwet[k + 1] * up[k + i + 1];
        }
    }
}

// The recurrences run across rows (k), so each whole-row sweep is an
// independent cwise expression.
void FieldSolverADI::tridagy(vector<complex<double > > &u) {
    const int n = static_cast<int>(ngrid);
    complex<double> *up = u.data();
    const complex<double> *rp = r.data();

    cspan(up, n) = ccspan(rp, n) * cbet[0];
    for (int k = 1; k < n; k++) {
        const int off = k * n;
        cspan(up + off, n) = (ccspan(rp + off, n) - c[k] * ccspan(up + off - n, n)) * cbet[k];
    }
    for (int k = n - 2; k >= 0; k--) {
        const int off = k * n;
        cspan(up + off, n) -= cwet[k + 1] * ccspan(up + off + n, n);
    }
}


void FieldSolverADI::init(double delz,double dgrid, double xks, unsigned int ngrid_in) {

    if (delz == delz_save) {
        return;
    }
    delz_save = delz;
    ngrid = ngrid_in;


    double rtmp = 0.25 * delz / (xks * dgrid * dgrid); //factor dz/(4 ks dx^2)
    cstep = complex<double>(0, rtmp);

    auto *mupp = new double[ngrid];
    auto *mmid = new double[ngrid];
    auto *mlow = new double[ngrid];
    auto *cwrk1 = new complex<double>[ngrid];
    auto *cwrk2 = new complex<double>[ngrid];
    if (c.size() != ngrid) {
        c.resize(ngrid);
        r.resize(ngrid * ngrid);
        cbet.resize(ngrid);
        cwet.resize(ngrid);
        crsource.resize(ngrid * ngrid);
    }

    mupp[0] = rtmp;
    mmid[0] = -2 * rtmp;
    mlow[0] = 0;
    for (int i = 1; i < (ngrid - 1); i++) {
        mupp[i] = rtmp;
        mmid[i] = -2 * rtmp;
        mlow[i] = rtmp;
    }
    mupp[ngrid - 1] = 0;
    mmid[ngrid - 1] = -2 * rtmp;
    mlow[ngrid - 1] = rtmp;

    for (int i = 0; i < ngrid; i++) {
        cwrk1[i] = complex<double>(0, -mupp[i]);
        cwrk2[i] = complex<double>(1, -mmid[i]);
        c[i] = complex<double>(0, -mlow[i]);
    }


    cbet[0] = 1. / cwrk2[0];
    cwet[0] = 0.;
    for (int i = 1; i < ngrid; i++) {
        cwet[i] = cwrk1[i - 1] * cbet[i - 1];
        cbet[i] = 1. / (cwrk2[i] - c[i] * cwet[i]);

    }

    delete[] mupp;
    delete[] mmid;
    delete[] mlow;
    delete[] cwrk1;
    delete[] cwrk2;
}
