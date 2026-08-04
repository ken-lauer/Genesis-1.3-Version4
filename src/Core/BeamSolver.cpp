#include "BeamSolver.h"
#include "Field.h"
#include "Beam.h"
#include "SimdBatch.h"

// per-particle field coupling rpart for a batch of particles, SoA with one
// lane per particle
struct BatchCoupling {
    const double *rharm;
    const double *re;
    const double *im;
    int nfld;

    // the staging arrays are 64-byte aligned and indexed in whole batches
    std::pair<dbatch, dbatch> rpart(int i) const
    {
        return {dbatch::load_aligned(re + i * dbatch_width),
                dbatch::load_aligned(im + i * dbatch_width)};
    }
};

// batched RK4 and ODE, defined below advance(); each lane advances one particle
void RungeKutta(dbatch &gamma, dbatch &theta, double delz, const dbatch &btpar,
                double xks, double xku, const dbatch &ez, const BatchCoupling &cpl);
void ODE(dbatch &k2gg, dbatch &k2pp, double xks, const dbatch &tgam, const dbatch &tthet,
         const dbatch &btpar, double xku, const dbatch &ez, const BatchCoupling &cpl);

// bilinear field sample at the particle position, scaled by the coupling
// constant; zero for particles off the grid
inline complex<double> sampleField(Field *pfld, const vector<complex<double>> &slc,
                                   double x, double y, double coupling)
{
    double wx, wy;
    int idx;
    if (!pfld->getLLGridpoint(x, y, &wx, &wy, &idx)) {
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
    rharm.clear();
    double xks = 1;  // default value in the case that no field is defined

    for (int i = 0; i < field->size(); i++) {
        auto pfld = (*field)[i];
        int harm = pfld->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = pfld->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / pfld->xks);      // here the harmonics have to be taken care
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
    using avec = vector<double, xsimd::aligned_allocator<double, 64>>;
    avec rp_re(static_cast<size_t>(nf) * dbatch_width);
    avec rp_im(static_cast<size_t>(nf) * dbatch_width);
    const BatchCoupling fc{rharm.data(), rp_re.data(), rp_im.data(), nf};
    alignas(64) double btv[dbatch_width], ezv[dbatch_width];
    // per-slice field slice lookups, constant over the particles of a slice
    vector<Field *> pfldv(nf);
    vector<const vector<complex<double>> *> slcv(nf);

    for (int is = 0; is < beam->beam.size(); is++) {
        auto &beam_is = beam->beam[is];
        // accumulate space charge field
        const double eloss = -beam->longESC[is] / eev; // convert eV to units of electron rest mass
        efield.shortRange(&beam_is, beam->current[is], gammaz2, is);

        for (int ifld = 0; ifld < nf; ifld++) {
            auto pfld = field->at(nfld[ifld]);
            const auto islice = (is + pfld->first) % pfld->field.size();
            pfldv[ifld] = pfld;
            slcv[ifld] = &pfld->field[islice];
        }

        double *g_s = beam_is.gamma();
        double *th_s = beam_is.theta();
        const double *x_s = beam_is.x();
        const double *y_s = beam_is.y();
        const double *px_s = beam_is.px();
        const double *py_s = beam_is.py();

        const int np = static_cast<int>(beam_is.size());
        // whole batches over the padded arrays (tail lanes replicate the last
        // particle): field interpolation and undulator lookups stay scalar per
        // lane, the RK4 (trig/sqrt/div) runs on full batches
        for (int ip = 0; ip < np; ip += dbatch_width) {
            for (int l = 0; l < dbatch_width; l++) {
                const double awloc = und->faw(x_s[ip + l], y_s[ip + l]);                 // get the transverse dependence of the undulator field
                btv[l] = 1 + px_s[ip + l] * px_s[ip + l] + py_s[ip + l] * py_s[ip + l] + aw * aw * awloc * awloc;
                // adding global long range space charge field to each particle
                ezv[l] = efield.getEField(ip + l) + eloss;

                for (int ifld = 0; ifld < nf; ifld++) {
                    const complex<double> rp = sampleField(pfldv[ifld], *slcv[ifld], x_s[ip + l], y_s[ip + l], rtmp[ifld] * awloc);
                    rp_re[ifld * dbatch_width + l] = rp.real();
                    rp_im[ifld * dbatch_width + l] = rp.imag();
                }
            }

            dbatch gamma = dbatch::load_aligned(g_s + ip);
            dbatch theta = dbatch::load_aligned(th_s + ip) + autophase; // add autophase here
            RungeKutta(gamma, theta, delz, dbatch::load_aligned(btv), xks, xku,
                       dbatch::load_aligned(ezv), fc);
            gamma.store_aligned(g_s + ip);
            theta.store_aligned(th_s + ip);
        }
    }
}

void RungeKutta(dbatch &gamma, dbatch &theta, double delz, const dbatch &btpar, double xks, double xku, const dbatch &ez, const BatchCoupling &cpl) {
    // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source

    // first step
    dbatch k2gg(0.);
    dbatch k2pp(0.);

    ODE(k2gg, k2pp, xks, gamma, theta, btpar, xku, ez, cpl);

    // second step
    double stpz = 0.5 * delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    dbatch k3gg = k2gg;
    dbatch k3pp = k2pp;

    k2gg = dbatch(0.);
    k2pp = dbatch(0.);

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


void ODE(dbatch &k2gg, dbatch &k2pp, double xks, const dbatch &tgam, const dbatch &tthet, const dbatch &btpar, double xku, const dbatch &ez, const BatchCoupling &cpl) {

    // differential equation for longitudinal motion; each lane advances one particle
    double ztemp1 = -2. / xks;
    dbatch ctmp_re(0.);
    dbatch ctmp_im(0.);
    for (int i = 0; i < cpl.nfld; i++) {
        auto angle = cpl.rharm[i] * tthet;
        // rpart * (cos - i sin)
        const auto [re, im] = cpl.rpart(i);
        const auto [s, c] = xsimd::sincos(angle);
        ctmp_re += re * c + im * s;
        ctmp_im += im * c - re * s;
    }
    dbatch btper0 = btpar + ztemp1 * ctmp_re;   //perpendicular velocity
    dbatch btpar0 = xsimd::sqrt(1. - btper0 / (tgam * tgam));     //parallel velocity
#ifdef G4_DBGDIAG
    // CL: detect negative radicands as NaN theta values can be the result
    dbatch btpar0_sq=1.-btper0/(tgam*tgam);     //(parallel velocity)^2
    if(xsimd::any(btpar0_sq < 0.)) {
      cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
    }
#endif
    k2pp += xks * (1. - 1. / btpar0) + xku;             //dtheta/dz
    k2gg += ctmp_im / (btpar0 * tgam) - ez;         //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}

