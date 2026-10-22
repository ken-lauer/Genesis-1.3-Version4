// The vectorized longitudinal push must agree with a straightforward scalar
// implementation of the same RK4 (libm sin/cos, master's formulation) per
// particle, and the inlineable sincos must match libm to a few ulp.
#include <cmath>
#include <cstdio>
#include <random>

#include "BeamSolverKernels.h"
#include "Particle.h"

using namespace beamsolver;

namespace {

int failures = 0;

void check(bool ok, const char *what)
{
    if (!ok) {
        failures++;
        std::printf("FAIL: %s\n", what);
    }
}

double reldiff(double a, double b)
{
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return scale == 0 ? 0 : std::fabs(a - b) / scale;
}

// master's BeamSolver::ODE / RungeKutta, one particle, libm trig
struct ScalarReference {
    double xks, xku, btpar, ez;
    std::vector<complex<double>> rpart;
    std::vector<double> rharm;

    void ODE(double tgam, double tthet, double &k2gg, double &k2pp) const
    {
        double ztemp1 = -2. / xks;
        complex<double> ctmp = 0;
        for (size_t i = 0; i < rpart.size(); i++) {
            ctmp += rpart[i] * complex<double>(cos(rharm[i] * tthet), -sin(rharm[i] * tthet));
        }
        double btper0 = btpar + ztemp1 * ctmp.real();
        double btpar0 = sqrt(1. - btper0 / (tgam * tgam));
        k2pp += xks * (1. - 1. / btpar0) + xku;
        k2gg += ctmp.imag() / btpar0 / tgam - ez;
    }
    void RungeKutta(double delz, double &gamma, double &theta) const
    {
        double k2gg = 0, k2pp = 0;
        ODE(gamma, theta, k2gg, k2pp);
        double stpz = 0.5 * delz;
        gamma += stpz * k2gg;
        theta += stpz * k2pp;
        double k3gg = k2gg, k3pp = k2pp;
        k2gg = 0;
        k2pp = 0;
        ODE(gamma, theta, k2gg, k2pp);
        gamma += stpz * (k2gg - k3gg);
        theta += stpz * (k2pp - k3pp);
        k3gg /= 6.0;
        k3pp /= 6.0;
        k2gg *= -0.5;
        k2pp *= -0.5;
        ODE(gamma, theta, k2gg, k2pp);
        stpz = delz;
        gamma += stpz * k2gg;
        theta += stpz * k2pp;
        k3gg -= k2gg;
        k3pp -= k2pp;
        k2gg *= 2;
        k2pp *= 2;
        ODE(gamma, theta, k2gg, k2pp);
        gamma += stpz * (k3gg + k2gg / 6.0);
        theta += stpz * (k3pp + k2pp / 6.0);
    }
};

} // namespace

int main()
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(-1., 1.);

    // sincos_poly against libm over the range theta*harmonic can take
    {
        double worst = 0;
        for (int i = 0; i < 200000; i++) {
            const double x = 300. * u(rng);
            double s, c;
            sincos_poly(x, s, c);
            worst = std::max(worst, std::fabs(s - std::sin(x)));
            worst = std::max(worst, std::fabs(c - std::cos(x)));
        }
        std::printf("sincos_poly vs libm: worst absolute error %.3g\n", worst);
        check(worst <= 1e-15, "sincos_poly within a few ulp of libm");
    }

    // two field grids with random content (members set directly; no MPI, no solver)
    const int nfields = 2;
    Field *flds[nfields];
    const vector<complex<double>> *grids[nfields];
    for (int f = 0; f < nfields; f++) {
        Field *fld = new Field;  // leaked on purpose: the destructor needs an initialized solver
        fld->ngrid = 31;
        fld->gridmax = 1e-3;
        fld->dgrid = 2 * fld->gridmax / static_cast<double>(fld->ngrid - 1);
        fld->first = 0;
        fld->field.resize(1);
        fld->field[0].resize(fld->ngrid * fld->ngrid);
        for (auto &c : fld->field[0]) { c = complex<double>(1e3 * u(rng), 1e3 * u(rng)); }
        flds[f] = fld;
        grids[f] = &fld->field[0];
    }
    const double rharm[nfields] = {1., 3.};
    const double rtmp[nfields] = {0.7, 0.2};
    const double pi = 2 * asin(1.);
    const UndTransverse utp{1e-5, -2e-5, 3e4, 2e4, 0.1, -0.2};

    for (int nf = 0; nf <= nfields; nf++) {
        const int np = 4 * 16 + 5;   // not a multiple of any vector width
        Particles P;
        P.resize(np);
        for (int i = 0; i < np; i++) {
            P[i].gamma = 2000. + 10. * u(rng);
            P[i].theta = 3. * u(rng);
            P[i].x = 1.2e-3 * u(rng);  // some particles fall off the grid
            P[i].y = 1.2e-3 * u(rng);
            P[i].px = 1e-3 * u(rng);
            P[i].py = 1e-3 * u(rng);
        }
        Particles R = P;
        std::vector<double> ez(np);
        for (auto &e : ez) { e = 1e-6 * u(rng); }

        const StepParams prm{0.01, 1.45, 0.3, 2 * pi / 1.34e-8, 2 * pi / 0.033, utp, rharm, rtmp, nf};
        const SliceData s{-1e-4, P.gamma(), P.theta(), P.x(), P.y(), P.px(), P.py(),
                          ez.data(), reinterpret_cast<const Field *const *>(flds), grids};
        std::vector<double> re, im;
        pushSlice(np, prm, s, re, im);

        // reference: master's scalar push, particle by particle
        for (int ip = 0; ip < np; ip++) {
            auto p = R[ip];
            const double awloc = utp.faw(p.x, p.y);
            ScalarReference ref;
            ref.xks = prm.xks;
            ref.xku = prm.xku;
            ref.btpar = 1 + p.px * p.px + p.py * p.py + prm.aw * prm.aw * awloc * awloc;
            ref.ez = ez[ip] + s.eloss;
            for (int f = 0; f < nf; f++) {
                ref.rpart.push_back(fieldCoupling(*flds[f], *grids[f], rtmp[f], p.x, p.y, awloc));
                ref.rharm.push_back(rharm[f]);
            }
            double gamma = p.gamma;
            double theta = p.theta + prm.autophase;
            ref.RungeKutta(prm.delz, gamma, theta);
            p.gamma = gamma;
            p.theta = theta;
        }

        double worst = 0;
        for (int i = 0; i < np; i++) {
            worst = std::max(worst, reldiff(P[i].gamma, R[i].gamma));
            worst = std::max(worst, reldiff(P[i].theta, R[i].theta));
            check(std::isfinite(P[i].gamma) && std::isfinite(P[i].theta), "finite result");
        }
        std::printf("nf=%d: vectorized vs scalar push, worst relative difference %.3g over %d particles\n",
                    nf, worst, np);
        check(worst <= 1e-12, "vectorized and scalar push agree to 1e-12");
    }

    std::printf(failures ? "FAILED (%d)\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
