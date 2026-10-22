
#include "Field.h"
#include "Beam.h"
#include "SimdMath.h"
#include "FieldSolverADI.h"
#include "FieldSolverFFT.h"
#include <algorithm>
#include <fstream>



Field::~Field() = default;

Field::Field(){
  harm=1;
  polarization=false;
  disabled=false;
  out_global=true;

  /* controlled by calls to Field::setOutput */
  doFFT=true;
  doSpatial=true;
  doIntensity=true;
  doDumpField=true;

  accuslip = 0;
}


void Field::init(int nsize, int ngrid_in, double dgrid_in, double xlambda0, double slicelen_in, double s0_in, int harm_in)
{  

  slicelength=slicelen_in;
  s0=s0_in;


  harm=harm_in;   // harmonics
  xlambda=xlambda0/static_cast<double>(harm);   // wavelength : xlambda0 is the reference wavelength of the fundamental
  ngrid=ngrid_in;

  gridmax=dgrid_in;
  dgrid=2*gridmax/static_cast<double>(ngrid-1); // grid pointe separation

  if (field.size()!=nsize){                  // allocate the memory in advance
     field.resize(nsize);
  }
  
  if (field[0].size()!=ngrid*ngrid){
    for (int i=0;i<nsize;i++){
        field[i].resize(ngrid*ngrid); 
    } 
  } 

  xks=4.*asin(1)/xlambda;
  first=0;                                // pointer to slice which correspond to first in the time window
  dz_save=0;

  solver  = new FieldSolverADI;
}

void Field::initSolver(bool isFFT,bool filter, double xc, double yc, double sig) {
    if (hasSolver_) {
        delete solver;
    }
    if (isFFT) {
        solver = new FieldSolverFFT;
        solver->initSourceFilter(xc,yc,sig,filter);
    } else {
        solver = new FieldSolverADI;
    }
    hasSolver_=true;
}

// at each run the buffer should be cleared.
void Field::initDiagnostics(int nz)
{
  idx=0;
  accuslip=0;    
  int ns=field.size();


  power.resize(ns*nz);
  if (doSpatial){
    xavg.resize(ns*nz);
    xsig.resize(ns*nz);
    yavg.resize(ns*nz);
    ysig.resize(ns*nz);
  }
#ifdef FFTW
  if (doFFT){  // output of FFT is requested
    txavg.resize(ns*nz);
    txsig.resize(ns*nz);
    tyavg.resize(ns*nz);
    tysig.resize(ns*nz);
  } else {
    txavg.resize(0);
    txsig.resize(0);
    tyavg.resize(0);
    tysig.resize(0);
  }
#endif
  nf_intensity.resize(ns*nz);
  nf_phi.resize(ns*nz);
  ff_intensity.resize(ns*nz);
  ff_phi.resize(ns*nz);
  if (out_global){
    energy.resize(nz);
    if (doSpatial){
      gl_xsig.resize(nz);
      gl_xavg.resize(nz);
      gl_ysig.resize(nz);
      gl_yavg.resize(nz);
    } else {
      gl_xsig.resize(0);
      gl_xavg.resize(0);
      gl_ysig.resize(0);
      gl_yavg.resize(0);
    }
    gl_nf_intensity.resize(nz);
    gl_ff_intensity.resize(nz);
#ifdef FFTW
    if (doFFT){
      gl_txsig.resize(nz);
      gl_txavg.resize(nz);
      gl_tysig.resize(nz);
      gl_tyavg.resize(nz);
    } else {
      gl_txsig.resize(0);
      gl_txavg.resize(0);
      gl_tysig.resize(0);
      gl_tyavg.resize(0);
    }
#endif
  }

}


void Field::setStepsize(double delz)
{

  if (delz!=dz_save){
    dz_save=delz;
    //getdiag_(&delz,&dgrid,&xks,&ngrid);
  }

}

bool Field::getLLGridpoint(double x, double y, double *wx, double *wy, int *idx) const {

      if ((x>-gridmax) && (x < gridmax) && (y>-gridmax) && (y < gridmax)){
           *wx = (x+gridmax)/dgrid;
           *wy = (y+gridmax)/dgrid;
           int ix= static_cast<int> (floor(*wx));
           int iy= static_cast<int> (floor(*wy));
           *wx=1+floor(*wx)-*wx;
           *wy=1+floor(*wy)-*wy;
           *idx=ix+iy*ngrid;
           return true;
      } else {
           return false;
      }
}

// Source term of one beam slice, shared by the ADI and FFT solvers (which own
// the crsource grid): a vectorized pass for each particle's phasor, then the
// scatter onto the four surrounding grid points, scalar by nature.
void Field::constructSource(vector<complex<double>> &crsource, Beam *beam,
                            Undulator *und, double delz, unsigned long islice)
{
    std::fill(crsource.begin(), crsource.end(), complex<double>(0));

    // no source needed for even harmonics
    if (!und->inUndulator() || !this->isEnabled() || (harm % 2 == 0)) { return; }

    double scl = und->fc(harm) * vacimp * beam->current[islice] * xks * delz;
    scl /= 4 * eev * static_cast<double>(beam->beam[islice].size()) * dgrid * dgrid;

    const auto &particles = beam->beam.at(islice);
    const int np = static_cast<int>(particles.size());
    const double *__restrict x = particles.x();
    const double *__restrict y = particles.y();
    const double *__restrict th = particles.theta();
    const double *__restrict g = particles.gamma();
    const UndTransverse utp = und->transverseParams();
    const double dharm = static_cast<double>(harm);
    const double gmax = gridmax;
    source_re.resize(np);
    source_im.resize(np);
    double *__restrict re = source_re.data();
    double *__restrict im = source_im.data();

    G4_SIMD_LOOP
    for (int ip = 0; ip < np; ip++) {
        // off-grid particles get 0 so the sqrt stays NaN-free; the scatter skips them
        const bool on = (x[ip] > -gmax) && (x[ip] < gmax) && (y[ip] > -gmax) && (y[ip] < gmax);
        const double faw2 = on ? utp.faw2(x[ip], y[ip]) : 0.;
        double sh, ch;
        sincos_poly(dharm * th[ip], sh, ch);
        // tmp  should be also normalized with beta parallel
        const double part = sqrt(faw2) * scl / g[ip];
        re[ip] = sh * part;
        im[ip] = ch * part;
    }

    for (int ip = 0; ip < np; ip++) {
        double wx, wy;
        int idx;
        if (!getLLGridpoint(x[ip], y[ip], &wx, &wy, &idx)) { continue; }
        const complex<double> cpart(re[ip], im[ip]);
        crsource[idx] += wx * wy * cpart;
        crsource[idx + 1] += (1 - wx) * wy * cpart;
        idx += ngrid;
        crsource[idx] += wx * (1 - wy) * cpart;
        crsource[idx + 1] += (1 - wx) * (1 - wy) * cpart;
    }
}


void Field::track(double delz, Beam *beam, Undulator *und)
{
  

  solver->init(delz,dgrid,xks,ngrid);  // check whether step size has changed and recalculate aux arrays
  solver->advance(delz,this,beam,und);
  return;

}


bool Field::subharmonicConversion(int harm_in, bool resample)
{
  // note that only an existing harmonic will be preserved
  harm=harm * harm_in;
  if ((!resample)|| (harm_in==1)){
    return true;
  }

  slicelength*=static_cast<double>(harm_in);
  int nsize=field.size();

  // step zero - calculate the radiation power, per slice

  vector<double> pow;
  pow.resize(nsize);
  for (int i=0;i<nsize;i++){
    pow[i]=0;
    for (int k=0; k<ngrid*ngrid;k++){
      complex<double> loc =field[i].at(k);  
      pow[i]+=loc.real()*loc.real()+loc.imag()*loc.imag();
    }
  }

 
  // step one - merge adjacent files into the first. 
  for (int i=0; i<nsize;i=i+harm_in){
    int is0= (i+first) % nsize; // loop over slices in the right order
    for (int j=1; j<harm_in; j++){
       int is1= (i+j+first) % nsize; // loop over slices to be merged
       for (int k=0; k<ngrid*ngrid;k++){
	 field[is0].at(k)+=field[is1].at(k);  // add field to slice
       }
       pow[is0]+=pow[is1];                  // add up power

    }
  }

  // step two - copy the merged one in the right order
  double scl=1./static_cast<double>(harm_in);

  int di=first % harm_in;
  for (int i=0; i<nsize/harm_in;i++){
     for (int k=0; k<ngrid*ngrid;k++){
	field[i].at(k)=field[i*harm_in+di].at(k);
     }
     pow[i]=pow[i*harm_in+di]*scl;    // calculate the normalized power
  }
  first=(first-di)/harm_in;

  // step three - average field by the subharmonic number
  field.resize(nsize/harm_in);
  for (int i=0; i<field.size();i++){
    double powloc=0;
    for (int k=0; k<ngrid*ngrid;k++){
       complex<double> loc =field[i].at(k);  
       powloc+=loc.real()*loc.real()+loc.imag()*loc.imag();
    }
    if (pow[i]>0){
     for (int k=0; k<ngrid*ngrid;k++){
       field[i].at(k)*=sqrt(pow[i]/powloc);  // do mean average because field were added up.
     }
    }

  }


  

  return true;
}



bool Field::harmonicConversion(int harm_in, bool resample)
{
  // note that only an existing harmonic will be preserved
  harm=1;
  if ((!resample)|| (harm_in==1)){
    return true;
  }

  slicelength/=static_cast<double>(harm_in);

  int nloc=field.size();
  field.resize(nloc*harm_in);
  for (int i=nloc;i<nloc*harm_in;i++){
    field.at(i).resize(ngrid*ngrid); // allocate space
  }

  // copy old slices into full field record
  for (int i=nloc;i>0;i--){
    for (int j=0;j<harm_in;j++){
      int idx=i*harm_in-1-j;
      for (int k=0; k<ngrid*ngrid;k++){
	field[idx].at(k)=field[(i-1)].at(k);
      }
    }
  }
  first*=harm_in; // adjust pointer of first slice
  return true;
}



