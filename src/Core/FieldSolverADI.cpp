#include "FieldSolverADI.h"
#include "Field.h"
#include "Beam.h"
#include "SimdMath.h"

#include <algorithm>

// The ADI step is bit-identical to the original scalar solver: every
// expression keeps its operation order, and complex products are spelled out
// in cmul so neither the NaN-recovery path of std::complex operator* nor FMA
// contraction (off for this file, see CMakeLists.txt) changes a rounding.
// G4_ADI_FAST trades that for fused multiply-adds and shorter products.
namespace {

inline complex<double> cmul(const complex<double> &a, const complex<double> &b)
{
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}

// product with a purely imaginary coefficient (cstep, c[k]); the full product
// keeps the zero real part's terms so the rounding matches the original
inline complex<double> cmul_imag(const complex<double> &a, const complex<double> &b)
{
#ifdef G4_ADI_FAST
    return {-a.imag() * b.imag(), a.imag() * b.real()};
#else
    return cmul(a, b);
#endif
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


// x-implicit half-step. The explicit stencil couples neighbouring rows
// (+-n) and is evaluated row by row; the tridiagonal solve runs along x,
// i.e. along each row, so the rows are taken in blocks of `lanes`, transposed
// into an L1-resident buffer where the recurrence advances all rows of the
// block together in contiguous memory, and written back.
//
//   r_k[i] = S_k[i] + U_k[i] + cstep*((U_{k+1}[i] - 2 U_k[i]) + U_{k-1}[i])   (one neighbour at the edges)
//   u[0] = r[0] * cbet_0,  u[i] = (r[i] - c_i * u[i-1]) * cbet_i             forward along the row
//   u[i] -= cwet_{i+1} * u[i+1]                                               backward along the row
void FieldSolverADI::halfStepX(const complex<double> *__restrict src, complex<double> *__restrict u)
{
    const int n = static_cast<int>(ngrid);
    constexpr int L = lanes;
    complex<double> *__restrict rb = rowbuf.data();    // stencil of the block's rows, L x n
    complex<double> *__restrict T = laneT.data();      // the block transposed, n x L
    complex<double> *__restrict prev = prevrow.data(); // field of the row above the block, before it was overwritten

    for (int row0 = 0; row0 < n; row0 += L) {
        const int nl = std::min(L, n - row0);
        for (int b = 0; b < nl; b++) {
            const int k = row0 + b;
            const complex<double> *S = src + k * n;
            const complex<double> *U = u + k * n;
            const complex<double> *Um = (b == 0) ? prev : U - n;
            const complex<double> *Up = U + n;
            complex<double> *r = rb + b * n;
            if (k == 0) {
                for (int i = 0; i < n; i++) {
                    r[i] = S[i] + U[i] + cmul_imag(cstep, Up[i] - 2.0 * U[i]);
                }
            } else if (k == n - 1) {
                for (int i = 0; i < n; i++) {
                    r[i] = S[i] + U[i] + cmul_imag(cstep, Um[i] - 2.0 * U[i]);
                }
            } else {
                for (int i = 0; i < n; i++) {
                    r[i] = S[i] + U[i] + cmul_imag(cstep, Up[i] - 2.0 * U[i] + Um[i]);
                }
            }
        }
        // the next block needs this block's last row as it is now
        std::copy(u + (row0 + nl - 1) * n, u + (row0 + nl) * n, prev);
        // forward and backward sweeps along the rows, the block's rows
        // advancing together: lane b is row row0+b, and the running value of
        // each lane stays in a small contiguous array
        complex<double> *ub = u + row0 * n;
        complex<double> cur[L];
        G4_SIMD_LOOP
        for (int b = 0; b < nl; b++) {
            const complex<double> v = cmul(rb[b * n], cbet[0]);
            cur[b] = v;
            ub[b * n] = v;
        }
        for (int i = 1; i < n; i++) {
            const complex<double> ci = c[i];
            const complex<double> bi = cbet[i];
            G4_SIMD_LOOP
            for (int b = 0; b < nl; b++) {
                const complex<double> v = cmul(rb[b * n + i] - cmul_imag(ci, cur[b]), bi);
                cur[b] = v;
                ub[b * n + i] = v;
            }
        }
        for (int i = n - 2; i >= 0; i--) {
            const complex<double> wi = cwet[i + 1];
            G4_SIMD_LOOP
            for (int b = 0; b < nl; b++) {
                const complex<double> v = ub[b * n + i] - cmul(wi, cur[b]);
                cur[b] = v;
                ub[b * n + i] = v;
            }
        }
    }
}

// y-implicit half-step: the explicit stencil along each row (neighbours at
// +-1), the tridiagonal solve across the rows. Row k is read into the row
// buffer before it is overwritten, so the forward sweep runs in the same pass
// as the stencil; every inner loop is over a contiguous row.
//
//   r_k[i] = S_k[i] + U_k[i] + cstep*((U_k[i+1] - 2 U_k[i]) + U_k[i-1])   (one neighbour at the edges)
//   u_0 = r_0 * cbet_0,   u_k = (r_k - c_k * u_{k-1}) * cbet_k            forward,  k = 1..n-1
//   u_k -= cwet_{k+1} * u_{k+1}                                           backward, k = n-2..0
void FieldSolverADI::halfStepY(const complex<double> *__restrict src, complex<double> *__restrict u)
{
    const int n = static_cast<int>(ngrid);
    complex<double> *__restrict r = rowbuf.data();

    for (int k = 0; k < n; k++) {
        const complex<double> *S = src + k * n;
        const complex<double> *U = u + k * n;
        r[0] = S[0] + U[0] + cmul_imag(cstep, U[1] - 2.0 * U[0]);
        for (int i = 1; i < n - 1; i++) {
            r[i] = S[i] + U[i] + cmul_imag(cstep, U[i + 1] - 2.0 * U[i] + U[i - 1]);
        }
        r[n - 1] = S[n - 1] + U[n - 1] + cmul_imag(cstep, U[n - 2] - 2.0 * U[n - 1]);
        complex<double> *uk = u + k * n;
        if (k == 0) {
            for (int i = 0; i < n; i++) {
                uk[i] = cmul(r[i], cbet[0]);
            }
        } else {
            const complex<double> *um = uk - n;
            const complex<double> ck = c[k];
            const complex<double> bk = cbet[k];
            for (int i = 0; i < n; i++) {
                uk[i] = cmul(r[i] - cmul_imag(ck, um[i]), bk);
            }
        }
    }
    for (int k = n - 2; k >= 0; k--) {
        complex<double> *uk = u + k * n;
        const complex<double> *up = uk + n;
        const complex<double> wk = cwet[k + 1];
        for (int i = 0; i < n; i++) {
            uk[i] -= cmul(wk, up[i]);
        }
    }
}

void FieldSolverADI::ADI(vector<complex<double>> &crfield)
{
    halfStepX(crsource.data(), crfield.data());
    halfStepY(crsource.data(), crfield.data());
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
        cbet.resize(ngrid);
        cwet.resize(ngrid);
        crsource.resize(ngrid * ngrid);
        rowbuf.resize(static_cast<size_t>(lanes) * ngrid);
        laneT.assign(static_cast<size_t>(lanes) * ngrid, 0.);  // unused lanes of a partial block stay finite
        prevrow.resize(ngrid);
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
