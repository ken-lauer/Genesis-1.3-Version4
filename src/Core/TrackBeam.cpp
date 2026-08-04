#include "TrackBeam.h"
#include "Beam.h"
#include "SimdBatch.h"

enum class QuadMode { Drift, Focus, Defocus };

inline QuadMode quadMode(double q)
{
    if (q == 0) {
        return QuadMode::Drift;
    }
    return q > 0 ? QuadMode::Focus : QuadMode::Defocus;
}

// transverse transport through one step, defined below track(); each lane
// advances one particle
void applyQuad(QuadMode mode, double delz, double qf, dbatch &x, dbatch &px,
               const dbatch &gammaz, double dx);

TrackBeam::TrackBeam(){}
TrackBeam::~TrackBeam(){}


void TrackBeam::track(double delz, Beam *beam,Undulator *und,bool lastStep=true)
{

  // get undulator parameter for the given step
  double aw,dax,day,ku,kx,ky;
  double qf,dqx,dqy;
  double cx,cy;
  double angle,lb,ld,lt;

  double gamma0=und->getGammaRef();
  und->getUndulatorParameters( &aw,&dax,&day,&ku,&kx,&ky);
  und->getQuadrupoleParameters(&qf,&dqx,&dqy);
  und->getCorrectorParameters(&cx,&cy);
  und->getChicaneParameters(&angle,&lb,&ld,&lt);

  double betpar0=sqrt(1-(1+aw*aw)/gamma0/gamma0);
  
  // effective focusing in x and y with the correct energy dependence
  double qquad=qf*gamma0;
  double qnatx=kx*aw*aw/gamma0/betpar0;  // kx has already the scaling with ku^2
  double qnaty=ky*aw*aw/gamma0/betpar0;  // same with ky

  double qx= qquad+qnatx;
  double qy=-qquad+qnaty;

  double xoff= qquad*dqx+qnatx*dax;
  double yoff=-qquad*dqy+qnaty*day;
  // cout << "qnaty: " << qnaty/gamma0 << " Gamma0: " << gamma0 << endl;

  if (lastStep){
    if ((cx!=0) || (cy!=0)) { this->applyCorrector(beam,cx*gamma0,cy*gamma0); }
  } else {
    if (angle!=0) { this->applyChicane(beam,angle,lb,ld,lt,gamma0); }
  }
  // handle the different cases (drift, focusing and defocusing)
  if (qx!=0){ xoff=xoff/qx; }
  if (qy!=0){ yoff=yoff/qy; }
  const QuadMode modeX = quadMode(qx);
  const QuadMode modeY = quadMode(qy);

  for (int i=0; i<beam->beam.size();i++){
    auto &slice = beam->beam.at(i);
    double *x_s = slice.x();
    double *px_s = slice.px();
    double *y_s = slice.y();
    double *py_s = slice.py();
    const double *g_s = slice.gamma();
    const int np = static_cast<int>(slice.size());
    // whole batches over the padded arrays; tail lanes replicate the last particle
    for (int j = 0; j < np; j += dbatch_width) {
      dbatch x = dbatch::load_aligned(x_s + j);
      dbatch px = dbatch::load_aligned(px_s + j);
      dbatch y = dbatch::load_aligned(y_s + j);
      dbatch py = dbatch::load_aligned(py_s + j);
      const dbatch g = dbatch::load_aligned(g_s + j);
      const dbatch gammaz = xsimd::sqrt(g * g - 1. - aw * aw - px * px - py * py); // = gamma*betaz=gamma*(1-(1+aw*aw)/gamma^2);
#ifdef G4_DBGDIAG
// G4_DBGDIAG: add test against negative radicand? Note that the particles probably already made lots of noise elsewhere.
#endif
      applyQuad(modeX, delz, qx, x, px, gammaz, xoff);
      applyQuad(modeY, delz, qy, y, py, gammaz, yoff);
      x.store_aligned(x_s + j);
      px.store_aligned(px_s + j);
      y.store_aligned(y_s + j);
      py.store_aligned(py_s + j);
    }
  }



  return;
} 


// the mode is uniform for all particles of a step, so the per-particle math
// is branch-free
void applyQuad(QuadMode mode, double delz, double qf, dbatch &x, dbatch &px, const dbatch &gammaz, double dx)
{
  if (mode == QuadMode::Drift){
    x+=px*delz/gammaz;
    return;
  }
  if (mode == QuadMode::Focus){
    dbatch foc=xsimd::sqrt(qf/gammaz);
    dbatch omg=foc*delz;
    const auto [s1, a1]=xsimd::sincos(omg);
    dbatch a2=s1/foc;
    dbatch a3=-a2*foc*foc;
    dbatch xtmp=x-dx;
    x =a1*xtmp+a2*px/gammaz+dx;
    px=a3*xtmp*gammaz+a1*px;
    return;
  }
  dbatch foc=xsimd::sqrt(-qf/gammaz);
  dbatch omg=foc*delz;
  dbatch a1=xsimd::cosh(omg);
  dbatch a2=xsimd::sinh(omg)/foc;
  dbatch a3=a2*foc*foc;
  dbatch xtmp=x-dx;
  x =a1*xtmp+a2*px/gammaz+dx;
  px=a3*xtmp*gammaz+a1*px;
}

void TrackBeam::applyCorrector(Beam *beam, double cx, double cy)
{ 

  for (int i=0; i<beam->beam.size();i++){
    for (int j=0; j<beam->beam.at(i).size();j++){
      beam->beam.at(i).at(j).px+=cx;
      beam->beam.at(i).at(j).py+=cy;
    }
  }
  return;
}


void TrackBeam::applyChicane(Beam *beam, double angle, double lb, double ld, double lt, double gamma0)
{ 
  // the tracking is done my applying the transfer matrix for the chicane and  backtracking for a drift over the length of the chicane
  // the effect of the R56 is applied here to the particle phase.  Then the normal tracking should do the momentum dependent change in the 
  // longitudinal position

  // the transfer matrix order is
  //  m -> bp -> ep -> d1 -> en -> bn -> d2 -> bn -> en-> d1 -> ep-> bp ->d3
  

  double m[4][4];
  double d1[4][4];
  double d2[4][4];
  double d3[4][4];
  double bp[4][4];
  double bn[4][4];
  double ep[4][4];
  double en[4][4];

  // construct the transfer matrix
  for (int i=0; i<4;i++){
    for (int j=0; j<4; j++){
      m[i][j]=0;
      d1[i][j]=0;
      d2[i][j]=0;
      d3[i][j]=0;
      bp[i][j]=0;
      bn[i][j]=0;
      ep[i][j]=0;
      en[i][j]=0;
    }
    m[i][i]=1;
    d1[i][i]=1;
    d2[i][i]=1;
    d3[i][i]=1;
    bp[i][i]=1;
    bn[i][i]=1;
    ep[i][i]=1;
    en[i][i]=1;
  }
  d1[0][1]=ld/cos(angle);  // drift between dipoles
  d1[2][3]=ld/cos(angle);   
  d2[0][1]=lt-4*lb-2*ld;   // drift in the middle
  d2[2][3]=lt-4*lb-2*ld;   
  d3[0][1]=-lt;            // negative drift over total chicane to get a zero element
  d3[2][3]=-lt;

  double R=lb/sin(angle);
  double Lpath=R*angle; 
  bp[2][3]=Lpath;  // positive deflection angle
  bp[0][0]=cos(angle);
  bp[0][1]=R*sin(angle);
  bp[1][0]=-sin(angle)/R;
  bp[1][1]=cos(angle);
 
  bn[2][3]=Lpath; // negative deflection angle
  bn[0][0]=cos(-angle);
  bn[0][1]=R*sin(-angle)*-1;
  bn[1][0]=-sin(-angle)/R*-1;
  bn[1][1]=cos(-angle);

  double efoc=tan(angle)/R;
  ep[1][0]=efoc;
  ep[3][2]=-efoc;
  en[1][0]=-efoc*-1;
  en[3][2]=efoc*-1;


  this->matmul(m,bp);
  this->matmul(m,ep);  
  this->matmul(m,d1);
  this->matmul(m,en);
  this->matmul(m,bn);
  this->matmul(m,d2);
  this->matmul(m,bn);
  this->matmul(m,en);
  this->matmul(m,d1);
  this->matmul(m,ep);
  this->matmul(m,bp);

  // transport matrix has been cross checked with Madx.  
  /*
  cout << "lt = " << lt << " angle = " << angle << " lb = " << lb << " ld = " << ld <<  endl;
  cout << m[0][0] << " " << m[0][1] << endl;
  cout << m[1][0] << " " << m[1][1] << endl;
  cout << m[2][2] << " " << m[2][3] << endl;
  cout << m[3][2] << " " << m[3][3] << endl;
  */

  this->matmul(m,d3);  // transport backwards because the main tracking still has to do the drift
  

  for (int i=0; i<beam->beam.size();i++){
    for (int j=0; j<beam->beam.at(i).size();j++){
      auto p=beam->beam.at(i)[j];
      double gammaz=sqrt(p.gamma*p.gamma-1- p.px*p.px - p.py*p.py); // = gamma*betaz=gamma*(1-(1+aw*aw)/gamma^2);

      double tmp=p.x;
      p.x =m[0][0]*tmp        +m[0][1]*p.px/gammaz;
      p.px=m[1][0]*tmp*gammaz +m[1][1]*p.px;
      tmp=p.y;
      p.y =m[2][2]*tmp        +m[2][3]*p.py/gammaz;
      p.py=m[3][2]*tmp*gammaz +m[3][3]*p.py;

    }
  }

  return;
}

void TrackBeam::matmul(double m[][4], double e[][4])
{
  double t[4][4];

  for (int i=0;i<4;i++){
    for (int j=0; j<4;j++){
      t[i][j]=0;
      for (int k=0;k<4;k++){
	t[i][j]+=e[i][k]*m[k][j];
      }
    }
  }
  for (int i=0;i<4;i++){
    for (int j=0; j<4;j++){
      m[i][j]=t[i][j];
    }
  }
}

void TrackBeam::applyR56(Beam *beam, Undulator *und, double lambda0)
{
  double angle,lb,ld,lt;

  double gamma0=und->getGammaRef();
  und->getChicaneParameters(&angle,&lb,&ld,&lt);
  if (angle==0) { return;}
  double R56=(4*lb/sin(angle)*(1-angle/tan(angle))+2*ld*tan(angle)/cos(angle))*angle;
  //    cout << "R56: " << R56 << endl;
  R56=R56*4*asin(1)/lambda0/gamma0;
  for (int i=0; i<beam->beam.size();i++){
    for (int j=0; j<beam->beam.at(i).size();j++){
      beam->beam.at(i).at(j).theta+=R56*(beam->beam.at(i).at(j).gamma-gamma0);
    }
  }
  return;

}
