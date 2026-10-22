#ifndef __GENESIS_FIELDSOLVERADI__
#define __GENESIS_FIELDSOLVERADI__

#include <vector>
#include <iostream>
#include <string>
#include <complex>


class Field;
class Beam;

#include "Particle.h"
#include "Undulator.h"
#include "FieldSolver.h"

using namespace std;


class FieldSolverADI : public FieldSolver{
 public:
    ~FieldSolverADI();
    void init(double,double,double,unsigned int);
    void advance(double, Field *, Beam *, Undulator *);
    void initSourceFilter(double,double,double,bool);

 private:
   unsigned int ngrid {0};
   double delz_save {0};
   complex<double> cstep;
   vector< complex< double > > c,cbet,cwet,crsource;
   static constexpr int lanes = 8;                 // rows solved together in the x-implicit half-step
   vector< complex< double > > rowbuf;             // stencil of the rows being swept
   vector< complex< double > > laneT;              // a block of rows, transposed
   vector< complex< double > > prevrow;            // the row above the current block, before it was overwritten

   void ADI(vector<complex< double > > &);
   void halfStepX(const complex<double> *src, complex<double> *u);
   void halfStepY(const complex<double> *src, complex<double> *u);
   
};

inline void FieldSolverADI::initSourceFilter(double x,double y,double z ,bool t) { return;}

#endif
