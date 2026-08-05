#include "FieldSolverADI.h"
#include "Field.h"
#include "Beam.h"
#include "SimdBatch.h"

#include <algorithm>

namespace {

// r = source + field + cstep*(sum_of_neighbors - 2*field)
inline cbatch stencil(const cbatch &s, const cbatch &f, const cbatch &neigh, const cbatch &vstep)
{
    return s + f + vstep * (neigh - f - f);
}

// Full forward/backward tridiagonal recurrence for B row-blocks
// (B * cbatch_width rows) starting at `row`. The recurrence runs along the
// contiguous (x) dimension, so lanes are filled with cbatch_width independent
// rows instead; the recurrence values stay in registers across k. B > 1
// advances several row-blocks together so their serial mul-add chains overlap
// (single-chain throughput is latency-bound).
template <int B>
void tridagxRows(complex<double> *up, const complex<double> *rp,
                 const vector<complex<double>> &c, const vector<complex<double>> &cbet,
                 const vector<complex<double>> &cwet, int n, int row)
{
    alignas(64) complex<double> lane[cbatch_width];
    auto gather = [&](const complex<double> *p, int base, int k) {
        for (int l = 0; l < cbatch_width; l++) {
            lane[l] = p[base + l * n + k];
        }
        return cbatch::load_aligned(lane);
    };
    auto scatter = [&](const cbatch &b, int base, int k) {
        b.store_aligned(lane);
        for (int l = 0; l < cbatch_width; l++) {
            up[base + l * n + k] = lane[l];
        }
    };

    int base[B];
    cbatch uk[B];
    for (int b = 0; b < B; b++) {
        base[b] = (row + b * cbatch_width) * n;
        uk[b] = gather(rp, base[b], 0) * cbatch(cbet[0]);
        scatter(uk[b], base[b], 0);
    }
    for (int k = 1; k < n; k++) {
        const cbatch ck(c[k]);
        const cbatch bk(cbet[k]);
        for (int b = 0; b < B; b++) {
            uk[b] = (gather(rp, base[b], k) - ck * uk[b]) * bk;
            scatter(uk[b], base[b], k);
        }
    }
    for (int k = n - 2; k >= 0; k--) {
        const cbatch wk(cwet[k + 1]);
        for (int b = 0; b < B; b++) {
            uk[b] = gather(up, base[b], k) - wk * uk[b];
            scatter(uk[b], base[b], k);
        }
    }
}

} // namespace


FieldSolverADI::~FieldSolverADI() = default;

void FieldSolverADI::advance(double delz, Field *field, Beam *beam, Undulator *und) {

    for (unsigned long ii = 0; ii < field->field.size(); ii++) {  // ii is index for the beam

        // clear source term
        std::fill(crsource.begin(), crsource.end(), complex<double>(0));

        // constructing source term
        int harm = field->getHarm();
        if (und->inUndulator() && field->isEnabled() && (harm % 2 == 1)) { // do not need to calculate for even harmonics
            double scl = und->fc(harm) * vacimp * beam->current[ii] * field->xks * delz;
            scl /= 4 * eev * static_cast<double>(beam->beam[ii].size()) * field->dgrid * field->dgrid;

            // scatter one particle onto its four surrounding grid points
            auto deposit = [&](const complex<double> &cpart, double wx, double wy, int idx) {
                crsource[idx] += wx * wy * cpart;
                crsource[idx + 1] += (1 - wx) * wy * cpart;
                idx += ngrid;
                crsource[idx] += wx * (1 - wy) * cpart;
                crsource[idx + 1] += (1 - wx) * (1 - wy) * cpart;
            };

            auto &particles = beam->beam.at(ii);
            const double *x_s = particles.x();
            const double *y_s = particles.y();
            const double *th_s = particles.theta();
            const double *g_s = particles.gamma();
            const int np = static_cast<int>(particles.size());
            const UndTransverse utp = und->transverseParams();
            alignas(64) double wxv[dbatch_width], wyv[dbatch_width], idxv[dbatch_width];
            alignas(64) double rev[dbatch_width], imv[dbatch_width];

            // whole batches over the padded arrays: everything batched except
            // the scatter onto the grid, which stays scalar per lane. The
            // scatter is a side effect, so off-grid lanes and tail lanes
            // (replicas of the last particle) must not deposit.
            for (int ip = 0; ip < np; ip += dbatch_width) {
                const dbatch x = dbatch::load_aligned(x_s + ip);
                const dbatch y = dbatch::load_aligned(y_s + ip);
                dbatch wx, wy, idx;
                const auto on = grid_weights(x, y, field->gridmax, field->dgrid,
                                             static_cast<double>(ngrid), wx, wy, idx);
                // off-grid lanes get 0 so the sqrt below stays NaN-free
                const dbatch f2 = xsimd::select(on, utp.faw2(x, y), dbatch(0.));

                const auto [s, c] = xsimd::sincos(static_cast<double>(harm) * dbatch::load_aligned(th_s + ip));
                // tmp  should be also normalized with beta parallel
                const dbatch part = xsimd::sqrt(f2) * scl / dbatch::load_aligned(g_s + ip);
                (s * part).store_aligned(rev);
                (c * part).store_aligned(imv);
                wx.store_aligned(wxv);
                wy.store_aligned(wyv);
                idx.store_aligned(idxv);

                const uint64_t onm = on.mask();
                const int lanes = std::min(dbatch_width, np - ip);
                for (int l = 0; l < lanes; l++) {
                    if (onm & (1ull << l)) {
                        deposit(complex<double>(rev[l], imv[l]), wxv[l], wyv[l],
                                static_cast<int>(idxv[l]));
                    }
                }
            }
        }  // end of source term construction

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
  const cbatch vstep(cstep);

  // implicit direction in x: neighbors live at +-ngrid, rows are contiguous
  int idx = 0;
  for (; idx + cbatch_width <= n; idx += cbatch_width) {
    const auto f = cbatch::load_unaligned(fld + idx);
    const auto s = cbatch::load_unaligned(src + idx);
    const auto up = cbatch::load_unaligned(fld + idx + n);
    stencil(s, f, up, vstep).store_unaligned(rp + idx);
  }
  for (; idx < n; idx++) {
    rp[idx] = src[idx] + fld[idx] + cstep * (fld[idx + n] - 2.0 * fld[idx]);
  }

  for (idx = n; idx + cbatch_width <= n * (n - 1); idx += cbatch_width) {
    const auto f = cbatch::load_unaligned(fld + idx);
    const auto s = cbatch::load_unaligned(src + idx);
    const auto neigh = cbatch::load_unaligned(fld + idx + n) + cbatch::load_unaligned(fld + idx - n);
    stencil(s, f, neigh, vstep).store_unaligned(rp + idx);
  }
  for (; idx < n * (n - 1); idx++) {
    rp[idx] = src[idx] + fld[idx] + cstep * (fld[idx + n] - 2.0 * fld[idx] + fld[idx - n]);
  }

  for (idx = n * (n - 1); idx + cbatch_width <= n * n; idx += cbatch_width) {
    const auto f = cbatch::load_unaligned(fld + idx);
    const auto s = cbatch::load_unaligned(src + idx);
    const auto dn = cbatch::load_unaligned(fld + idx - n);
    stencil(s, f, dn, vstep).store_unaligned(rp + idx);
  }
  for (; idx < n * n; idx++) {
    rp[idx] = src[idx] + fld[idx] + cstep * (fld[idx - n] - 2.0 * fld[idx]);
  }

  // solve tridiagonal system in x
  this->tridagx(crfield);

  // implicit direction in y: neighbors live at +-1 within each contiguous row
  for (int ix = 0; ix < n * n; ix += n) {
    rp[ix] = src[ix] + fld[ix] + cstep * (fld[ix + 1] - 2.0 * fld[ix]);
    int i = ix + 1;
    for (; i + cbatch_width <= ix + n - 1; i += cbatch_width) {
      const auto f = cbatch::load_unaligned(fld + i);
      const auto s = cbatch::load_unaligned(src + i);
      const auto neigh = cbatch::load_unaligned(fld + i + 1) + cbatch::load_unaligned(fld + i - 1);
      stencil(s, f, neigh, vstep).store_unaligned(rp + i);
    }
    for (; i < ix + n - 1; i++) {
      rp[i] = src[i] + fld[i] + cstep * (fld[i + 1] - 2.0 * fld[i] + fld[i - 1]);
    }
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

// The recurrences run across rows (k), so each contiguous inner i-loop is
// independent and vectorizes directly.
void FieldSolverADI::tridagy(vector<complex<double > > &u) {
    const int n = static_cast<int>(ngrid);
    complex<double> *up = u.data();
    const complex<double> *rp = r.data();

    {
        const cbatch b0(cbet[0]);
        int i = 0;
        for (; i + cbatch_width <= n; i += cbatch_width) {
            (cbatch::load_unaligned(rp + i) * b0).store_unaligned(up + i);
        }
        for (; i < n; i++) {
            up[i] = rp[i] * cbet[0];
        }
    }
    for (int k = 1; k < n; k++) {
        const int off = k * n;
        const cbatch ck(c[k]);
        const cbatch bk(cbet[k]);
        int i = 0;
        for (; i + cbatch_width <= n; i += cbatch_width) {
            const auto prev = cbatch::load_unaligned(up + off - n + i);
            const auto rr = cbatch::load_unaligned(rp + off + i);
            ((rr - ck * prev) * bk).store_unaligned(up + off + i);
        }
        for (; i < n; i++) {
            up[off + i] = (rp[off + i] - c[k] * up[off + i - n]) * cbet[k];
        }
    }
    for (int k = n - 2; k >= 0; k--) {
        const int off = k * n;
        const cbatch wk(cwet[k + 1]);
        int i = 0;
        for (; i + cbatch_width <= n; i += cbatch_width) {
            const auto cur = cbatch::load_unaligned(up + off + i);
            const auto nxt = cbatch::load_unaligned(up + off + n + i);
            (cur - wk * nxt).store_unaligned(up + off + i);
        }
        for (; i < n; i++) {
            up[off + i] -= cwet[k + 1] * up[off + i + n];
        }
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
