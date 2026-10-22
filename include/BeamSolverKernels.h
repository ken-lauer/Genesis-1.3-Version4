#ifndef __GENESIS_BEAMSOLVERKERNELS__
#define __GENESIS_BEAMSOLVERKERNELS__

// Longitudinal push of one slice: a scalar gather of each particle's field
// coupling, then the RK4 integration as a plain SoA loop the compiler
// vectorizes (SimdMath.h).

#include <cmath>
#include <complex>
#include <vector>

#include "Field.h"
#include "SimdMath.h"
#include "UndTransverse.h"

using namespace std;

namespace beamsolver {

// constants of one z-step, shared by all slices
struct StepParams {
    double delz, aw, autophase, xks, xku;
    UndTransverse utp;     // transverse dependence of the undulator field
    const double *rharm;   // harmonic number of each coupled field
    const double *rtmp;    // coupling constant of each coupled field
    int nf;                // number of coupled fields
};

// the particle arrays and field grids of one slice
struct SliceData {
    double eloss;          // long range space charge field, in units of the rest mass
    double *gamma, *theta;
    const double *x, *y, *px, *py;
    const double *ez;      // short range space charge field
    const Field *const *flds;
    const vector<complex<double>> *const *grid;  // this slice's grid of each coupled field
};

// coupling of one particle to one field:
// rtmp * awloc * conj(bilinear field sample at the particle position), zero off grid
inline complex<double> fieldCoupling(const Field &fld, const vector<complex<double>> &grid,
                                     double rtmp, double x, double y, double awloc)
{
    double wx, wy;
    int idx;
    if (!fld.getLLGridpoint(x, y, &wx, &wy, &idx)) { return 0; }  // particle off the grid
    complex<double> cpart = grid[idx] * wx * wy;
    cpart += grid[idx + 1] * (1 - wx) * wy;
    idx += fld.ngrid;
    cpart += grid[idx] * wx * (1 - wy);
    cpart += grid[idx + 1] * (1 - wx) * (1 - wy);
    return rtmp * awloc * conj(cpart);
}

// differential equation for longitudinal motion of particle ip
G4_ALWAYS_INLINE void ode(double tgam, double tthet, double &k2gg, double &k2pp,
                          double btpar, double ez, int ip, int np, int nf,
                          const double *__restrict re, const double *__restrict im,
                          const double *rharm, double ztemp1, double xks, double xku)
{
    double ctmp_re = 0, ctmp_im = 0;
    for (int f = 0; f < nf; f++) {
        // ctmp += rpart * (cos(h theta) - i sin(h theta))
        double sn, cs;
        sincos_poly(rharm[f] * tthet, sn, cs);
        const double r = re[f * np + ip];
        const double i = im[f * np + ip];
        ctmp_re += r * cs + i * sn;
        ctmp_im += i * cs - r * sn;
    }
    const double btper0 = btpar + ztemp1 * ctmp_re;               // perpendicular velocity
    const double btpar0 = std::sqrt(1. - btper0 / (tgam * tgam));  // parallel velocity
    k2pp += xks * (1. - 1. / btpar0) + xku;                        // dtheta/dz
    k2gg += ctmp_im / (btpar0 * tgam) - ez;                        // dgamma/dz
}

// Runge Kutta 4th order over the particles of a slice - taken from pushp
// from the old Fortran source. The number of coupled fields NF is a template
// parameter so the inner loop unrolls and the particle loop vectorizes;
// NF < 0 means the runtime count p.nf (not vectorized).
template <int NF>
void rungeKutta(int np, const StepParams &p, const SliceData &s,
                const double *__restrict re, const double *__restrict im)
{
    double *__restrict g = s.gamma;
    double *__restrict th = s.theta;
    const double *__restrict x = s.x;
    const double *__restrict y = s.y;
    const double *__restrict px = s.px;
    const double *__restrict py = s.py;
    const double *__restrict ezs = s.ez;
    const double delz = p.delz, aw = p.aw, autophase = p.autophase;
    const double xks = p.xks, xku = p.xku, eloss = s.eloss;
    const double ztemp1 = -2. / xks;
    const UndTransverse utp = p.utp;
    const double *rharm = p.rharm;
    const int nf = NF >= 0 ? NF : p.nf;

    G4_SIMD_LOOP
    for (int ip = 0; ip < np; ip++) {
        const double awloc = utp.faw(x[ip], y[ip]);   // transverse dependence of the undulator field
        const double btpar = 1. + px[ip] * px[ip] + py[ip] * py[ip] + aw * aw * awloc * awloc;
        const double ez = ezs[ip] + eloss;            // adding global long range space charge field
        double gamma = g[ip];
        double theta = th[ip] + autophase;            // add autophase here

        // first step
        double k2gg = 0, k2pp = 0;
        ode(gamma, theta, k2gg, k2pp, btpar, ez, ip, np, nf, re, im, rharm, ztemp1, xks, xku);

        // second step
        double stpz = 0.5 * delz;
        gamma += stpz * k2gg;
        theta += stpz * k2pp;
        double k3gg = k2gg, k3pp = k2pp;
        k2gg = 0;
        k2pp = 0;
        ode(gamma, theta, k2gg, k2pp, btpar, ez, ip, np, nf, re, im, rharm, ztemp1, xks, xku);

        // third step
        gamma += stpz * (k2gg - k3gg);
        theta += stpz * (k2pp - k3pp);
        k3gg /= 6.0;
        k3pp /= 6.0;
        k2gg *= -0.5;
        k2pp *= -0.5;
        ode(gamma, theta, k2gg, k2pp, btpar, ez, ip, np, nf, re, im, rharm, ztemp1, xks, xku);

        // fourth step
        stpz = delz;
        gamma += stpz * k2gg;
        theta += stpz * k2pp;
        k3gg -= k2gg;
        k3pp -= k2pp;
        k2gg *= 2;
        k2pp *= 2;
        ode(gamma, theta, k2gg, k2pp, btpar, ez, ip, np, nf, re, im, rharm, ztemp1, xks, xku);
        gamma += stpz * (k3gg + k2gg / 6.0);
        theta += stpz * (k3pp + k2pp / 6.0);

        g[ip] = gamma;
        th[ip] = theta;
    }
}

// advance all particles of one slice by one step; re/im are scratch for the
// field coupling of every particle (SoA, one block per field)
inline void pushSlice(int np, const StepParams &p, const SliceData &s,
                      vector<double> &re, vector<double> &im)
{
    re.resize(static_cast<size_t>(p.nf) * np);
    im.resize(static_cast<size_t>(p.nf) * np);
    // the four-point field gather, scalar by nature
    for (int f = 0; f < p.nf; f++) {
        for (int ip = 0; ip < np; ip++) {
            const double x = s.x[ip];
            const double y = s.y[ip];
            const complex<double> c =
                fieldCoupling(*s.flds[f], *s.grid[f], p.rtmp[f], x, y, p.utp.faw(x, y));
            re[f * np + ip] = c.real();
            im[f * np + ip] = c.imag();
        }
    }
    // the vectorized push; more than three coupled fields (harmonics) falls
    // back to the runtime loop count, which is not vectorized
    switch (p.nf) {
        case 0: rungeKutta<0>(np, p, s, re.data(), im.data()); break;
        case 1: rungeKutta<1>(np, p, s, re.data(), im.data()); break;
        case 2: rungeKutta<2>(np, p, s, re.data(), im.data()); break;
        case 3: rungeKutta<3>(np, p, s, re.data(), im.data()); break;
        default: rungeKutta<-1>(np, p, s, re.data(), im.data());
    }
}

} // namespace beamsolver

#endif
