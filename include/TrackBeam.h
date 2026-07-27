#ifndef __GENESIS_TRACKBEAM__
#define __GENESIS_TRACKBEAM__

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <map>
#include <stdlib.h>


class Beam;
#include "Undulator.h"
#include "Particle.h"


using namespace std;

class TrackBeam{
 public:
   TrackBeam();
   virtual ~TrackBeam();
   void track(double, Beam *, Undulator *, bool);

   void applyCorrector(Beam *, double, double);
   void applyChicane(Beam *, double, double, double, double,double);
   void applyR56(Beam *, Undulator *, double);

 private:
   void matmul(double a[][4], double b[][4]);
};


#endif
