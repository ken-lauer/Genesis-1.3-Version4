#include "BeamSolver.h"
#include "Field.h"
#include "Beam.h"

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
    xks = 1;  // default value in the case that no field is defined

    for (int i = 0; i < field->size(); i++) {
        auto pfld = field->at(i);
        int harm = pfld->getHarm();
        if ((harm == 1) || !onlyFundamental) {
            xks = pfld->xks / static_cast<double>(harm);    // fundamental field wavenumber used in ODE below
            nfld.push_back(i);
            rtmp.push_back(und->fc(harm) / pfld->xks);      // here the harmonics have to be taken care
            rpart.emplace_back(0);
            rharm.push_back(static_cast<double>(harm));
        }
    }

    xku = und->getku();
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
    for (int is = 0; is < beam->beam.size(); is++) {
        auto &beam_is = beam->beam.at(is);
        // accumulate space charge field
        double eloss = -beam->longESC[is] / 511000; // convert eV to units of electron rest mass
        efield.shortRange(&beam_is, beam->current.at(is), gammaz2, is);
        for (int ip = 0; ip < beam_is.size(); ip++) {
            auto &particle = beam->beam.at(is).at(ip);
            gamma = particle.gamma;
            theta = particle.theta + autophase; // add autophase here
            double x = particle.x;
            double y = particle.y;
            double px = particle.px;
            double py = particle.py;
            double awloc = und->faw(x, y);                 // get the transverse dependence of the undulator field
            btpar = 1 + px * px + py * py + aw * aw * awloc * awloc;
            ez = efield.getEField(ip) + eloss;  // adding global long range space charge field to each particle
            cpart = 0;
            double wx, wy;
            int idx;
            for (int ifld = 0; ifld < nfld.size(); ifld++) {
                auto pfld = field->at(nfld[ifld]);
                auto islice = (is + pfld->first) % pfld->field.size();

                if (pfld->getLLGridpoint(x, y, &wx, &wy, &idx)) { // check whether particle is on grid
                    auto slc = pfld->field[islice].at(idx);
                    cpart = slc * wx * wy;
                    idx++;
                    cpart += slc * (1 - wx) * wy;
                    idx += pfld->ngrid - 1;
                    cpart += slc * wx * (1 - wy);
                    idx++;
                    cpart += slc * (1 - wx) * (1 - wy);
                    rpart[ifld] = rtmp[ifld] * awloc * conj(cpart);
                } else {
                    rpart[ifld] = 0;
                }
            }
            this->RungeKutta(delz);

            particle.gamma = gamma;
            particle.theta = theta;
        }
    }
}

void BeamSolver::RungeKutta(double delz) {
    // Runge Kutta Solver 4th order - taken from pushp from the old Fortran source


    // first step
    k2gg = 0;
    k2pp = 0;

    this->ODE(gamma, theta);

    // second step
    double stpz = 0.5 * delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg = k2gg;
    k3pp = k2pp;

    k2gg = 0;
    k2pp = 0;

    this->ODE(gamma, theta);

    // third step
    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);

    k3gg /= 6;
    k3pp /= 6;

    k2gg *= -0.5;
    k2pp *= -0.5;

    this->ODE(gamma, theta);

    // fourth step
    stpz = delz;

    gamma += stpz * k2gg;
    theta += stpz * k2pp;

    k3gg -= k2gg;
    k3pp -= k2pp;

    k2gg *= 2;
    k2pp *= 2;

    this->ODE(gamma, theta);
    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

}


void BeamSolver::ODE(double tgam,double tthet) {

    // differential equation for longitudinal motion
    double ztemp1 = -2. / xks;
    complex<double> ctmp = 0;
    for (int i = 0; i < rpart.size(); i++) {
        ctmp += rpart[i] * complex<double>(cos(rharm[i] * tthet), -sin(rharm[i] * tthet));
    }
    double btper0 = btpar + ztemp1 * ctmp.real();   //perpendicular velocity
    double btpar0 = sqrt(1. - btper0 / (tgam * tgam));     //parallel velocity
#ifdef G4_DBGDIAG
    // CL: detect negative radicands as NaN theta values can be the result
    double btpar0_sq=1.-btper0/(tgam*tgam);     //(parallel velocity)^2
    if(btpar0_sq<0) {
      cout << "DBGDIAG(BeamSolver::ODE): error, negative radicand detected" << endl;
    }
#endif
    k2pp += xks * (1. - 1. / btpar0) + xku;             //dtheta/dz
    k2gg += ctmp.imag() / btpar0 / tgam - ez;         //dgamma/dz
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}

