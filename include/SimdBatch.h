#ifndef __GENESIS_SIMDBATCH__
#define __GENESIS_SIMDBATCH__

#include <array>
#include <complex>
#include <cstddef>
#include <utility>

#include <Eigen/Core>
#include <xsimd/xsimd.hpp>

#include "Particle.h"

// SIMD batch types shared by the vectorized solvers, tracker and diagnostics.
// Fixed-size Eigen arrays: Eigen maps each statement onto however many
// hardware packets the width needs and unrolls. A batch therefore spans
// SEVERAL packets on purpose - the unroll gives the serial RK4/ODE chains
// instruction-level parallelism a single-register batch cannot (measured
// ~17% on NEON vs single-packet widths).
//
// Kernels use the native Eigen idioms directly:
//   loads   const dbatch x = dbatch::MapAligned(p);   // or ::Map(p) unaligned
//   stores  dbatch::MapAligned(p) = expr;
//   math    x.sqrt(), x.sin(), mask.select(a, b), acc.sum(), ...
// MapAligned asserts AlignedMax = the packet alignment of the target ISA,
// which the Eigen::aligned_allocator arrays and alignas(64) staging buffers
// always satisfy. Bind loads to a concrete batch (not `auto`) so the value is
// materialized instead of aliasing memory that a later store overwrites.

// Hardware packet sizes for the instruction set selected at compile time
// (e.g. -march=native): doubles 2/4/8 and complex<double> 1/2/4 on
// NEON/AVX2/AVX-512. packet_traits sits in Eigen's internal namespace but is
// the stable, canonical source of these constants.
constexpr int dpacket_width = Eigen::internal::packet_traits<double>::size;
constexpr int cpacket_width =
    Eigen::internal::packet_traits<std::complex<double>>::size;

// Doubles: a full particles_simd_pad per batch - 4 packets on NEON, 2 on
// AVX2 and AVX-512 (16 doubles there). Complex: two packets per batch -
// enough to pair the mul-add chains without inflating tridagx's per-lane
// gather/scatter.
constexpr int dbatch_width = static_cast<int>(particles_simd_pad);
constexpr int cbatch_width = 2 * cpacket_width;

using dbatch = Eigen::Array<double, dbatch_width, 1>;
using cbatch = Eigen::Array<std::complex<double>, cbatch_width, 1>;
using dmask = Eigen::Array<bool, dbatch_width, 1>;

// the Particles padding must let every kernel run whole batches over the tail
static_assert(particles_simd_pad % dbatch_width == 0,
              "Particles padding is not a multiple of the double batch width");
static_assert(particles_simd_pad % cbatch_width == 0,
              "Particles padding is not a multiple of the complex batch width");
// and batches must be whole numbers of hardware packets
static_assert(
    dbatch_width % dpacket_width == 0,
    "double batch width is not a multiple of the hardware packet size");

// dmask with the first `count` lanes active (all lanes if count >= width).
// Use it to mask the final batch of a padded particle loop wherever tail lanes
// must not contribute: reductions and scatters. Plain loads/stores of the
// padded component arrays themselves never need it.
inline dmask tail_mask(std::size_t count) {
  alignas(64) static constexpr std::array<double, particles_simd_pad> iota = [] {
    std::array<double, particles_simd_pad> a{};
    for (std::size_t i = 0; i < a.size(); i++) {
      a[i] = static_cast<double>(i);
    }
    return a;
  }();
  return dbatch::MapAligned(iota.data()) < static_cast<double>(count);
}

// sin and cos of one batch of angles, sharing the argument reduction.
// xsimd's Cephes-style kernel (pure FMA polynomial, no division) is ~2.4x
// faster than the pair of Eigen Pade-based .sin()/.cos() calls, which each
// reduce the argument and evaluate BOTH rationals only to discard one.
// Accuracy is the usual Cephes ~1-2 ulp instead of Eigen's <1 ulp.
EIGEN_ALWAYS_INLINE void sincos(const dbatch &angle, dbatch &s, dbatch &c) {
  using xb = xsimd::batch<double>;
  static_assert(dbatch_width % static_cast<int>(xb::size) == 0,
                "batch width is not a multiple of the xsimd packet size");
  // dbatch is EIGEN_MAX_ALIGN_BYTES-aligned, which meets or exceeds the
  // xsimd packet alignment on every instruction set
  for (int i = 0; i < dbatch_width; i += static_cast<int>(xb::size)) {
    const auto [ss, cc] = xsimd::sincos(xb::load_aligned(angle.data() + i));
    ss.store_aligned(s.data() + i);
    cc.store_aligned(c.data() + i);
  }
}

namespace simd_detail {
template <class F, std::size_t N, std::size_t... I>
void lane_apply(F &&fn, int l, const double (&stage)[N][dbatch_width],
                std::index_sequence<I...>) {
  fn(l, stage[I][l]...);
}
} // namespace simd_detail

// Per-lane visitor over one batch of a padded particle loop: evaluates each
// batch expression into aligned staging, then calls fn(lane, v0, v1, ...)
// with the lanes' scalars, for every lane < count where on(lane) is set.
// This is the one supported way to feed batch results into scalar per-lane
// code (gathers from and scatters onto grids); pass count = dbatch_width to
// visit tail lanes too, or the remaining particle count to mask them off.
template <class F, class... Bs>
void for_each_lane(const dmask &on, int count, F &&fn, const Bs &...batches) {
  static_assert(sizeof...(Bs) > 0, "stage at least one batch");
  alignas(64) double stage[sizeof...(Bs)][dbatch_width];
  std::size_t j = 0;
  ((dbatch::MapAligned(stage[j++]) = batches), ...);
  const int lanes = count < dbatch_width ? count : dbatch_width;
  for (int l = 0; l < lanes; l++) {
    if (on(l)) {
      simd_detail::lane_apply(fn, l, stage, std::index_sequence_for<Bs...>{});
    }
  }
}

// Reduction over the first np entries of padded particle arrays: body(ip)
// returns N batches of per-lane contributions, batch_sum returns their N
// horizontal sums. Whole batches accumulate unmasked (padding makes the loads
// valid); only the final partial batch is masked so tail lanes contribute
// nothing. Usage:
//
//   auto [s1, s2] = batch_sum<2>(np, [&](int ip) -> std::array<dbatch, 2> {
//       const dbatch x = dbatch::MapAligned(x_s + ip);
//       return {x, x * x};
//   });
template <std::size_t N, class F>
std::array<double, N> batch_sum(int np, F &&body) {
  std::array<dbatch, N> acc;
  for (auto &a : acc) {
    a.setZero();
  }
  int ip = 0;
  for (; ip + dbatch_width <= np; ip += dbatch_width) {
    const std::array<dbatch, N> v = body(ip);
    for (std::size_t j = 0; j < N; j++) {
      acc[j] += v[j];
    }
  }
  if (ip < np) {
    const dmask m = tail_mask(np - ip);
    const std::array<dbatch, N> v = body(ip);
    for (std::size_t j = 0; j < N; j++) {
      acc[j] += m.select(v[j], 0.0);
    }
  }
  std::array<double, N> out;
  for (std::size_t j = 0; j < N; j++) {
    out[j] = acc[j].sum();
  }
  return out;
}

#endif
