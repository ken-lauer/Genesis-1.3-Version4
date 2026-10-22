#ifndef __GENESIS_UNDTRANSVERSE__
#define __GENESIS_UNDTRANSVERSE__

// Transverse dependence of the undulator field for one z-step, fetched once
// per step so the per-particle kernels evaluate a polynomial on scalars.
// note kx is scaled as XKX*ku*ku in Lattice.cpp, gradx as ku*GRADX.
struct UndTransverse {
    double ax, ay, kx, ky, gradx, grady;

    double faw(double x, double y) const
    {
        const double dx = x - ax;
        const double dy = y - ay;
        return 1. + 0.5 * (kx * dx * dx + ky * dy * dy) + gradx * dx + grady * dy;
    }

    // square of the transverse dependence
    double faw2(double x, double y) const
    {
        const double dx = x - ax;
        const double dy = y - ay;
        return 1. + kx * dx * dx + ky * dy * dy + 2. * (gradx * dx + grady * dy);
    }
};

#endif
