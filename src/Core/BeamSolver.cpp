#include "BeamSolver.h"
#include "Field.h"
#include "Beam.h"

#include <xsimd/xsimd.hpp>


using dbatch = xsimd::batch<double>;
constexpr int dbatch_width = static_cast<int>(dbatch::size);

// per-particle field coupling rpart for a batch of particles, SoA with one
// lane per particle
struct BatchCoupling {
    const double *rharm;
    const double *re;
    const double *im;
    int nfld;

    std::pair<dbatch, dbatch> rpart(int i) const
    {
        return {dbatch::load_unaligned(re + i * dbatch_width),
                dbatch::load_unaligned(im + i * dbatch_width)};
    }
};

// same view over the per-particle rpart of the scalar remainder path
struct ScalarCoupling {
    const double *rharm;
    const complex<double> *rp;
    int nfld;

    std::pair<double, double> rpart(int i) const
    {
        return {rp[i].real(), rp[i].imag()};
    }
};

#ifdef G4_DBGDIAG
inline bool anyNegative(double v) { return v < 0.; }
inline bool anyNegative(const dbatch &v) { return xsimd::any(v < 0.); }
#endif

// scalar/batch-generic RK4 and ODE, defined below advance(); T = double
// advances one particle, T = dbatch a batch of particles
template <class T, class Coupling>
void RungeKutta(T &gamma, T &theta, double delz, const T &btpar,
                double xks, double xku, const T &ez, const Coupling &cpl);
template <class T, class Coupling>
void ODE(T &k2gg, T &k2pp, double xks, const T &tgam, const T &tthet,
         const T &btpar, double xku, const T &ez, const Coupling &cpl);

// bilinear field sample at the particle, scaled by the coupling constant;
// zero for particles off the grid
inline complex<double> sampleField(Field *pfld, const vector<complex<double>> &slc,
                                   const Particle &particle, double coupling)
{
    double wx, wy;
    int idx;
    if (!pfld->getLLGridpoint(particle.x, particle.y, &wx, &wy, &idx)) {
        return 0;
    }
    complex<double> cp = slc[idx] * wx * wy;
    cp += slc[idx + 1] * (1 - wx) * wy;
    idx += pfld->ngrid;
    cp += slc[idx] * wx * (1 - wy);
    cp += slc[idx + 1] * (1 - wx) * (1 - wy);
    return coupling * conj(cp);
}

BeamSolver::BeamSolver()
{
  onlyFundamental=false;
}

BeamSolver::~BeamSolver()= default;


void BeamSolver::advance(double delz, Beam *beam, vector< Field *> *field, Undulator *und) {

    // here the harmonics needs to be taken into account

    vector<int> nfld;
    vector<double> rtmp;
    rpart.clear();
    rharm.clear();
    double xks = 1;  // default value in the case that no field is defined

    for (int i = 0; i < field->size(); i++) {
        auto pfld = (*field)[i];
        int harm = pfld->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = pfld->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / pfld->xks);      // here the harmonics have to be taken care
            rpart.emplace_back(0);
            rharm.push_back(static_cast<double>(harm));
        }
    }

    double xku = und->getku();
    if (xku ==
        0) {   // in the case of drifts - the beam stays in phase if it has the reference energy // this requires that the phase slippage is not applied
        xku = xks * 0.5 / und->getGammaRef() / und->getGammaRef();
    }

    double aw = und->getaw();
    double autophase = und->autophase();

    // obtaining long range space charge field
    efield.longRange(beam, und->getGammaRef(), aw);  // defines the array beam->longESC

    // Runge Kutta solver to advance particle
    const auto gammaz2 = und->getGammaRef()*und->getGammaRef()/(1+aw*aw);
    const int nf = static_cast<int>(nfld.size());
    vector<double> rp_re(static_cast<size_t>(nf) * dbatch_width);
    vector<double> rp_im(static_cast<size_t>(nf) * dbatch_width);
    const BatchCoupling fc{rharm.data(), rp_re.data(), rp_im.data(), nf};
    double thv[dbatch_width], gmv[dbatch_width], btv[dbatch_width], ezv[dbatch_width];
    // per-slice field slice lookups, constant over the particles of a slice
    vector<Field *> pfldv(nf);
    vector<const vector<complex<double>> *> slcv(nf);

    for (int is = 0; is < beam->beam.size(); is++) {
        auto &beam_is = beam->beam[is];
        // accumulate space charge field
        const double eloss = -beam->longESC[is] / 511000; // convert eV to units of electron rest mass
        efield.shortRange(&beam_is, beam->current[is], gammaz2, is);

        for (int ifld = 0; ifld < nf; ifld++) {
            auto pfld = field->at(nfld[ifld]);
            const auto islice = (is + pfld->first) % pfld->field.size();
            pfldv[ifld] = pfld;
            slcv[ifld] = &pfld->field[islice];
        }

        const int np = static_cast<int>(beam_is.size());
        int ip = 0;
        // batched path: field interpolation and write-back stay scalar per
        // lane, the RK4 (trig/sqrt/div) runs on full batches
        for (; ip + dbatch_width <= np; ip += dbatch_width) {
            for (int l = 0; l < dbatch_width; l++) {
                auto &particle = beam_is[ip + l];
                const double awloc = und->faw(particle.x, particle.y);                 // get the transverse dependence of the undulator field
                btv[l] = 1 + particle.px * particle.px + particle.py * particle.py + aw * aw * awloc * awloc;
                // adding global long range space charge field to each particle
                ezv[l] = efield.getEField(ip + l) + eloss;
                thv[l] = particle.theta + autophase; // add autophase here
                gmv[l] = particle.gamma;

                for (int ifld = 0; ifld < nf; ifld++) {
                    const complex<double> rp = sampleField(pfldv[ifld], *slcv[ifld], particle, rtmp[ifld] * awloc);
                    rp_re[ifld * dbatch_width + l] = rp.real();
                    rp_im[ifld * dbatch_width + l] = rp.imag();
                }
            }

            dbatch gamma = dbatch::load_unaligned(gmv);
            dbatch theta = dbatch::load_unaligned(thv);
            RungeKutta(gamma, theta, delz, dbatch::load_unaligned(btv), xks, xku,
                       dbatch::load_unaligned(ezv), fc);
            gamma.store_unaligned(gmv);
            theta.store_unaligned(thv);
            for (int l = 0; l < dbatch_width; l++) {
                beam_is[ip + l].gamma = gmv[l];
                beam_is[ip + l].theta = thv[l];
            }
        }
        // scalar remainder
        for (; ip < np; ip++) {
            auto &particle = beam_is[ip];
            const double awloc = und->faw(particle.x, particle.y);                 // get the transverse dependence of the undulator field
            const double btpar = 1 + particle.px * particle.px + particle.py * particle.py + aw * aw * awloc * awloc;
            // adding global long range space charge field to each particle
            // efield.ez[ip]
            const double ez = efield.getEField(ip) + eloss;
            for (int ifld = 0; ifld < nf; ifld++) {
                rpart[ifld] = sampleField(pfldv[ifld], *slcv[ifld], particle, rtmp[ifld] * awloc);
            }

            double gamma = particle.gamma;
            double theta = particle.theta + autophase; // add autophase here
            RungeKutta(gamma, theta, delz, btpar, xks, xku, ez,
                       ScalarCoupling{rharm.data(), rpart.data(), nf});
            particle.gamma = gamma;
            particle.theta = theta;
        }
    }
}

template <class T, class Coupling>
void RungeKutta(T &gamma, T &theta, double delz, const T &btpar, double xks, double xku, const T &ez, const Coupling &cpl) {
    // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source

    // first step
    T k2gg(0.);
    T k2pp(0.);

    ODE(k2gg, k2pp, xks, gamma, theta, btpar, xku, ez, cpl);

    // second step
    double stpz = 0.5 * delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    T k3gg = k2gg;
    T k3pp = k2pp;

    k2gg = T(0.);
    k2pp = T(0.);

    ODE(k2gg, k2pp, xks, gamma, theta, btpar, xku, ez, cpl);

    // third step
    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);

    k3gg /= 6;
    k3pp /= 6;

    k2gg *= -0.5;
    k2pp *= -0.5;

    ODE(k2gg, k2pp, xks, gamma, theta, btpar, xku, ez, cpl);

    // fourth step
    stpz = delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;

    k2gg *= 2;
    k2pp *= 2;

    ODE(k2gg, k2pp, xks, gamma, theta, btpar, xku, ez, cpl);
    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

}


template <class T, class Coupling>
void ODE(T &k2gg, T &k2pp, double xks, const T &tgam, const T &tthet, const T &btpar, double xku, const T &ez, const Coupling &cpl) {

    // differential equation for longitudinal motion
    // T = double advances one particle, T = dbatch a batch of particles;
    // xsimd provides the scalar overloads of sincos/sqrt
    double ztemp1 = -2. / xks;
    T ctmp_re(0.);
    T ctmp_im(0.);
    for (int i = 0; i < cpl.nfld; i++) {
        auto angle = cpl.rharm[i] * tthet;
        // rpart * (cos - i sin)
        const auto [re, im] = cpl.rpart(i);
        const auto [s, c] = xsimd::sincos(angle);
        ctmp_re += re * c + im * s;
        ctmp_im += im * c - re * s;
    }
    T btper0 = btpar + ztemp1 * ctmp_re;   //perpendicular velocity
    T btpar0 = xsimd::sqrt(1. - btper0 / (tgam * tgam));     //parallel velocity
#ifdef G4_DBGDIAG
    // CL: detect negative radicands as NaN theta values can be the result
    T btpar0_sq=1.-btper0/(tgam*tgam);     //(parallel velocity)^2
    if(anyNegative(btpar0_sq)) {
      cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
    }
#endif
    k2pp += xks * (1. - 1. / btpar0) + xku;             //dtheta/dz
    k2gg += ctmp_im / (btpar0 * tgam) - ez;         //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}

