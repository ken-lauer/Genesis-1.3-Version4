#include "BeamSolver.h"
#include "Field.h"
#include "Beam.h"
#include "SimdBatch.h"

// Per-step constants of the batched RK4: the wavenumbers and the
// per-particle field coupling rpart, staged SoA with one lane per particle
// and one batch per field.
struct RKParams {
    double xks;
    double xku;
    const double *rharm;
    const double *re;
    const double *im;
    int nfld;

    // the staging arrays are packet-aligned and indexed in whole batches
    std::pair<dbatch, dbatch> rpart(int i) const
    {
        return {dbatch::MapAligned(re + i * dbatch_width),
                dbatch::MapAligned(im + i * dbatch_width)};
    }
};

// batched RK4 and ODE, defined below advance(); each lane advances one particle
void RungeKutta(dbatch &gamma, dbatch &theta, double delz, const dbatch &btpar,
                const dbatch &ez, const RKParams &prm);
void ODE(dbatch &k2gg, dbatch &k2pp, const dbatch &tgam, const dbatch &tthet,
         const dbatch &btpar, const dbatch &ez, const RKParams &prm);

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
        int harm = field->at(i)->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = field->at(i)->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / field->at(i)->xks);      // here the harmonics have to be taken care
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
    auto gammaz2 = und->getGammaRef()*und->getGammaRef()/(1+aw*aw);
    const int nf = static_cast<int>(nfld.size());
    using avec = vector<double, Eigen::aligned_allocator<double>>;
    avec rp_re(static_cast<size_t>(nf) * dbatch_width);
    avec rp_im(static_cast<size_t>(nf) * dbatch_width);
    const RKParams prm{xks, xku, rharm.data(), rp_re.data(), rp_im.data(), nf};
    // per-slice field slice lookups, constant over the particles of a slice
    vector<Field *> pfldv(nf);
    vector<const vector<complex<double>> *> slcv(nf);
    // per-step undulator transverse dependence, constant over the particles
    const UndTransverse utp = und->transverseParams();

    for (int is = 0; is < beam->beam.size(); is++) {
        auto &beam_is = beam->beam.at(is);
        // accumulate space charge field
        double eloss = -beam->longESC[is] / eev; // convert eV to units of electron rest mass
        efield.shortRange(&beam_is, beam->current.at(is), gammaz2, is);

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

        const double *ez_s = efield.getEFieldData();

        const int np = static_cast<int>(beam_is.size());
        // whole batches over the padded arrays (tail lanes replicate the last
        // particle): everything is batched except the four-point gather from
        // the field grid, which stays scalar per lane
        for (int ip = 0; ip < np; ip += dbatch_width) {
            const dbatch x = dbatch::MapAligned(x_s + ip);
            const dbatch y = dbatch::MapAligned(y_s + ip);
            const dbatch px = dbatch::MapAligned(px_s + ip);
            const dbatch py = dbatch::MapAligned(py_s + ip);
            // transverse dependence of the undulator field
            const dbatch awloc = utp.faw(x, y);
            const dbatch btpar = 1. + px * px + py * py + aw * aw * awloc * awloc;
            // adding global long range space charge field to each particle
            // (ez comes from a plain vector<double>, not batch-aligned)
            const dbatch ez = dbatch::Map(ez_s + ip) + eloss;

            for (int ifld = 0; ifld < nf; ifld++) {
                const Field *pfld = pfldv[ifld];
                const vector<complex<double>> &slc = *slcv[ifld];
                dbatch wx, wy, idx;
                const dmask on = pfld->getLLGridpointBatch(x, y, wx, wy, idx);
                double *re = rp_re.data() + ifld * dbatch_width;
                double *im = rp_im.data() + ifld * dbatch_width;
                dbatch::MapAligned(re).setZero();
                dbatch::MapAligned(im).setZero();
                const double rt = rtmp[ifld];
                const int ng = pfld->ngrid;
                // bilinear sample at the particle position, scaled by the
                // coupling constant; lanes off the grid keep zero. All lanes
                // are visited: tail lanes replicate the last particle and
                // must hold valid couplings like every other lane.
                for_each_lane(on, dbatch_width,
                              [&](int l, double wxl, double wyl, double idxl, double awl) {
                                  int i = static_cast<int>(idxl);
                                  complex<double> cp = slc[i] * wxl * wyl;
                                  cp += slc[i + 1] * (1 - wxl) * wyl;
                                  i += ng;
                                  cp += slc[i] * wxl * (1 - wyl);
                                  cp += slc[i + 1] * (1 - wxl) * (1 - wyl);
                                  const complex<double> rp = rt * awl * conj(cp);
                                  re[l] = rp.real();
                                  im[l] = rp.imag();
                              },
                              wx, wy, idx, awloc);
            }

            dbatch gamma = dbatch::MapAligned(g_s + ip);
            dbatch theta = dbatch::MapAligned(th_s + ip) + autophase; // add autophase here
            RungeKutta(gamma, theta, delz, btpar, ez, prm);
            dbatch::MapAligned(g_s + ip) = gamma;
            dbatch::MapAligned(th_s + ip) = theta;
        }
    }
}

void RungeKutta(dbatch &gamma, dbatch &theta, double delz, const dbatch &btpar, const dbatch &ez, const RKParams &prm) {
    // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source

    // first step
    dbatch k2gg = dbatch::Zero();
    dbatch k2pp = dbatch::Zero();

    ODE(k2gg, k2pp, gamma, theta, btpar, ez, prm);

    // second step
    double stpz = 0.5 * delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    dbatch k3gg = k2gg;
    dbatch k3pp = k2pp;

    k2gg.setZero();
    k2pp.setZero();

    ODE(k2gg, k2pp, gamma, theta, btpar, ez, prm);

    // third step
    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);

    k3gg /= 6;
    k3pp /= 6;

    k2gg *= -0.5;
    k2pp *= -0.5;

    ODE(k2gg, k2pp, gamma, theta, btpar, ez, prm);

    // fourth step
    stpz = delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;

    k2gg *= 2;
    k2pp *= 2;

    ODE(k2gg, k2pp, gamma, theta, btpar, ez, prm);
    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

}


void ODE(dbatch &k2gg, dbatch &k2pp, const dbatch &tgam, const dbatch &tthet, const dbatch &btpar, const dbatch &ez, const RKParams &prm) {

    // differential equation for longitudinal motion; each lane advances one particle
    double ztemp1 = -2. / prm.xks;
    dbatch ctmp_re = dbatch::Zero();
    dbatch ctmp_im = dbatch::Zero();
    for (int i = 0; i < prm.nfld; i++) {
        const dbatch angle = prm.rharm[i] * tthet;
        // rpart * (cos - i sin)
        const auto [re, im] = prm.rpart(i);
        const dbatch s = angle.sin();
        const dbatch c = angle.cos();
        ctmp_re += re * c + im * s;
        ctmp_im += im * c - re * s;
    }
    dbatch btper0 = btpar + ztemp1 * ctmp_re;   //perpendicular velocity
    dbatch btpar0 = (1. - btper0 / (tgam * tgam)).sqrt();     //parallel velocity
#ifdef G4_DBGDIAG
    // CL: detect negative radicands as NaN theta values can be the result
    dbatch btpar0_sq=1.-btper0/(tgam*tgam);     //(parallel velocity)^2
    if((btpar0_sq < 0.).any()) {
      cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
    }
#endif
    k2pp += prm.xks * (1. - 1. / btpar0) + prm.xku;     //dtheta/dz
    k2gg += ctmp_im / (btpar0 * tgam) - ez;         //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}

