#ifndef __GENESIS_SIMDBATCH__
#define __GENESIS_SIMDBATCH__

#include <array>
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

// Batched Field::getLLGridpoint: bilinear weights and lower-left grid index
// of each lane's cell, plus the on-grid mask. wx/wy/idx of off-grid lanes are
// garbage and must be skipped via the mask. idx is integer-valued but kept in
// doubles (exact well below 2^53); lane consumers cast per lane. The division
// by dgrid matches the scalar code bit for bit.
inline xsimd::batch_bool<double> grid_weights(const dbatch &x, const dbatch &y,
                                              double gridmax, double dgrid, double ngrid,
                                              dbatch &wx, dbatch &wy, dbatch &idx)
{
    const auto on = (x > -gridmax) && (x < gridmax) && (y > -gridmax) && (y < gridmax);
    const dbatch tx = (x + gridmax) / dgrid;
    const dbatch ty = (y + gridmax) / dgrid;
    const dbatch fx = xsimd::floor(tx);
    const dbatch fy = xsimd::floor(ty);
    wx = 1. + fx - tx;
    wy = 1. + fy - ty;
    idx = fx + fy * ngrid;
    return on;
}

// Reduction over the first np entries of padded particle arrays: body(ip)
// returns N batches of per-lane contributions, batch_sum returns their N
// horizontal sums. Whole batches accumulate unmasked (padding makes the loads
// valid); only the final partial batch is masked so tail lanes contribute
// nothing. Usage:
//
//   auto [s1, s2] = batch_sum<2>(np, [&](int ip) -> std::array<dbatch, 2> {
//       const dbatch x = dbatch::load_aligned(x_s + ip);
//       return {x, x * x};
//   });
template <std::size_t N, class F>
std::array<double, N> batch_sum(int np, F &&body)
{
    std::array<dbatch, N> acc;
    acc.fill(dbatch(0.));
    int ip = 0;
    for (; ip + dbatch_width <= np; ip += dbatch_width) {
        const std::array<dbatch, N> v = body(ip);
        for (std::size_t j = 0; j < N; j++) { acc[j] += v[j]; }
    }
    if (ip < np) {
        const auto m = tail_mask(np - ip);
        const std::array<dbatch, N> v = body(ip);
        for (std::size_t j = 0; j < N; j++) { acc[j] += xsimd::select(m, v[j], dbatch(0.)); }
    }
    std::array<double, N> out;
    for (std::size_t j = 0; j < N; j++) { out[j] = xsimd::reduce_add(acc[j]); }
    return out;
}

#endif
