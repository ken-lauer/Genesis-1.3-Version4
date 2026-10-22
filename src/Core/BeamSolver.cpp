#include "BeamSolver.h"
#include "Field.h"
#include "Beam.h"
#include "BeamSolverKernels.h"

using namespace beamsolver;


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
    const StepParams prm{delz, aw, autophase, xks, xku, und->transverseParams(),
                         rharm.data(), rtmp.data(), nf};
    // per-slice field grid lookups, constant over the particles of a slice
    vector<const Field *> flds(nf);
    vector<const vector<complex<double>> *> grids(nf);

    for (int is = 0; is < beam->beam.size(); is++) {
        auto &beam_is = beam->beam.at(is);
        // accumulate space charge field
        efield.shortRange(&beam_is, beam->current.at(is), gammaz2, is);

        for (int ifld = 0; ifld < nf; ifld++) {
            const Field *pfld = field->at(nfld[ifld]);
            flds[ifld] = pfld;
            grids[ifld] = &pfld->field[(is + pfld->first) % pfld->field.size()];
        }

        const SliceData sd{-beam->longESC[is] / eev,  // convert eV to units of electron rest mass
                           beam_is.gamma(), beam_is.theta(),
                           beam_is.x(), beam_is.y(), beam_is.px(), beam_is.py(),
                           efield.getEFieldData(), flds.data(), grids.data()};
        pushSlice(static_cast<int>(beam_is.size()), prm, sd, rp_re, rp_im);
    }
}

void BeamSolver::checkAllocation(unsigned long nslice) {
    efield.allocateForOutput(nslice);
}

