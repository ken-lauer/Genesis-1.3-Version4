#ifndef __GENESIS_SIMDBATCH__
#define __GENESIS_SIMDBATCH__

#include <complex>

#include <xsimd/xsimd.hpp>

// SIMD batch types shared by the vectorized solvers, tracker and diagnostics.
// The batch width follows the instruction set selected at compile time
// (e.g. -march=native): 2 doubles on NEON, 4 on AVX2, 8 on AVX-512.

using dbatch = xsimd::batch<double>;
using cbatch = xsimd::batch<std::complex<double>>;

constexpr int dbatch_width = static_cast<int>(dbatch::size);
constexpr int cbatch_width = static_cast<int>(cbatch::size);

#endif
