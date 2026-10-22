#ifndef __GENESIS_SIMDMATH__
#define __GENESIS_SIMDMATH__

#include <cmath>
#include <cstdint>

// Vectorization helpers for the plain SoA particle loops. The kernels are
// ordinary scalar C++ over the component arrays; the compiler vectorizes the
// loops marked G4_SIMD_LOOP (-fopenmp-simd), with gathers and scatters kept in
// separate scalar loops. Build target `vec-audit` checks that every marked
// loop is actually vectorized.
#define G4_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define G4_SIMD_LOOP_WITH(...) G4_PRAGMA(omp simd __VA_ARGS__) _Pragma("clang loop vectorize(enable) interleave_count(4)")
#else
#define G4_SIMD_LOOP_WITH(...) G4_PRAGMA(omp simd __VA_ARGS__)
#endif
// G4_SIMD_LOOP_WITH(reduction(+:a,b)) for reductions, G4_SIMD_LOOP otherwise
#define G4_SIMD_LOOP G4_SIMD_LOOP_WITH()

// Functions called inside a G4_SIMD_LOOP body must inline, or the loop
// silently stays scalar.
#define G4_ALWAYS_INLINE [[gnu::always_inline]] inline

// Branch-free scalar sin/cos pair that compilers can inline into an
// auto-vectorized loop (the plain-loop variant of the particle kernels).
// Cephes-style: reduce by pi/2 with a two-part constant (exact for
// |x| < ~1e6, far beyond any theta*harmonic here), evaluate both polynomials
// on the reduced argument, pick by quadrant. Accuracy ~1-2 ulp, the same
// class as the xsimd kernel used by the batch path.
G4_ALWAYS_INLINE void sincos_poly(double x, double &s, double &c)
{
    // fdlibm's Cody-Waite split of pi/2: 33 bits + 53 bits, so n*pio2_1 is
    // exact for n < 2^20 and the reduction holds ~2^-86 of pi/2
    constexpr double pio2_1 = 1.57079632673412561417e+00;
    constexpr double pio2_1t = 6.07710050650619224932e-11;
    constexpr double twoopi = 6.36619772367581382433e-1;

    const double xa = std::fabs(x);
    const double n = std::nearbyint(xa * twoopi);
    const double xr = (xa - n * pio2_1) - n * pio2_1t;
    const double z = xr * xr;

    // sin(xr) = xr + xr*z*P(z), cos(xr) = 1 - z/2 + z*z*Q(z)
    double ps = 1.58962301576546568060e-10;
    ps = ps * z - 2.50507477628578072866e-8;
    ps = ps * z + 2.75573136213857245213e-6;
    ps = ps * z - 1.98412698295895385996e-4;
    ps = ps * z + 8.33333333332211858878e-3;
    ps = ps * z - 1.66666666666666307295e-1;
    const double sr = xr + xr * z * ps;

    double pc = -1.13585365213876817300e-11;
    pc = pc * z + 2.08757008419747316778e-9;
    pc = pc * z - 2.75573141792967388112e-7;
    pc = pc * z + 2.48015872888517045348e-5;
    pc = pc * z - 1.38888888888730564116e-3;
    pc = pc * z + 4.16666666666665929218e-2;
    const double cr = 1. - 0.5 * z + z * z * pc;

    const std::int64_t q = static_cast<std::int64_t>(n);
    const bool swap = (q & 1) != 0;          // quadrants 1,3: sin <-> cos
    const bool neg_s = (q & 2) != 0;         // quadrants 2,3: sin negative
    const bool neg_c = ((q + 1) & 2) != 0;   // quadrants 1,2: cos negative
    double sv = swap ? cr : sr;
    double cv = swap ? sr : cr;
    sv = neg_s ? -sv : sv;
    cv = neg_c ? -cv : cv;
    s = x < 0 ? -sv : sv;                    // sin is odd, cos is even
    c = cv;
}

#endif
