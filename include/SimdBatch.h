#ifndef __GENESIS_SIMDBATCH__
#define __GENESIS_SIMDBATCH__

#include <complex>
#include <cstddef>

#include <xsimd/xsimd.hpp>

#include "Particle.h"

// SIMD batch types shared by the vectorized solvers, tracker and diagnostics.
// The batch width follows the instruction set selected at compile time
// (e.g. -march=native): 2 doubles on NEON, 4 on AVX2, 8 on AVX-512.

using dbatch = xsimd::batch<double>;
using cbatch = xsimd::batch<std::complex<double>>;

constexpr int dbatch_width = static_cast<int>(dbatch::size);
constexpr int cbatch_width = static_cast<int>(cbatch::size);

// the Particles padding must let every kernel run whole batches over the tail
static_assert(particles_simd_pad % dbatch_width == 0,
              "Particles padding is not a multiple of the double batch width");
static_assert(particles_simd_pad % cbatch_width == 0,
              "Particles padding is not a multiple of the complex batch width");

// batch_bool with the first `count` lanes active (all lanes if count >= width).
// Use it to mask the final batch of a padded particle loop wherever tail lanes
// must not contribute: reductions and scatters. Plain loads/stores of the
// padded component arrays themselves never need it.
inline xsimd::batch_bool<double> tail_mask(std::size_t count)
{
    alignas(64) static constexpr double iota[particles_simd_pad] = {0., 1., 2., 3.,
                                                                    4., 5., 6., 7.};
    static_assert(particles_simd_pad == 8, "iota table must match the pad width");
    return dbatch::load_aligned(iota) < dbatch(static_cast<double>(count));
}

#endif
