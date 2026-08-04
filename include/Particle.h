#ifndef __GENESIS_PARTICLE__
#define __GENESIS_PARTICLE__

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

// standalone xsimd subheader (std includes only) - keeps the full xsimd.hpp
// out of the many translation units that see this header via Beam.h
#include <xsimd/memory/xsimd_aligned_allocator.hpp>

using namespace std;

typedef struct{
  double gamma;
  double theta;
  double x;
  double y;
  double px;
  double py;
} Particle;

// Component arrays are physically sized to a multiple of this (in doubles) so
// SIMD kernels can always run whole batches, tail included. Must be a multiple
// of every batch width in use - enforced by a static_assert in SimdBatch.h.
inline constexpr std::size_t particles_simd_pad = 8;

constexpr std::size_t padded_count(std::size_t n)
{
    return (n + particles_simd_pad - 1) / particles_simd_pad * particles_simd_pad;
}


// Structure-of-arrays storage for the particles of one beam slice.
//
// Element access returns lightweight proxies whose members are references
// into the six component arrays, so existing `slice[j].theta = ...` code
// compiles unchanged. Conventions (same as std::vector<bool>):
//  - copy-CONSTRUCTING a Ref rebinds (`auto p = slice[j]` refers to slot j),
//  - copy-ASSIGNING writes through (`slice[i] = slice[j]` copies the particle).
// Constness is shallow on Ref itself; real read-only protection comes from the
// const overloads, which hand out ConstRef with const double& members.
//
// Padding invariant: the arrays are physically sized to padded_count(size())
// and the slots in [size(), padded) replicate the last particle. SIMD kernels
// may therefore always run whole batches: tail lanes compute valid (if
// redundant) physics and may be stored back freely. Only operations with side
// effects beyond the arrays themselves - reductions, scatters onto grids -
// must mask the tail lanes (see tail_mask in SimdBatch.h).
class Particles {
public:
    struct Ref {
        double &gamma;
        double &theta;
        double &x;
        double &y;
        double &px;
        double &py;

        operator Particle() const { return Particle{gamma, theta, x, y, px, py}; }
        Ref(const Ref &) = default;
        Ref &operator=(const Particle &p)
        {
            gamma = p.gamma;
            theta = p.theta;
            x = p.x;
            y = p.y;
            px = p.px;
            py = p.py;
            return *this;
        }
        Ref &operator=(const Ref &o)
        {
            gamma = o.gamma;
            theta = o.theta;
            x = o.x;
            y = o.y;
            px = o.px;
            py = o.py;
            return *this;
        }
    };

    struct ConstRef {
        const double &gamma;
        const double &theta;
        const double &x;
        const double &y;
        const double &px;
        const double &py;

        ConstRef(const Ref &r)
            : gamma(r.gamma), theta(r.theta), x(r.x), y(r.y), px(r.px), py(r.py) {}
        ConstRef(const double &g, const double &t, const double &x_in, const double &y_in,
                 const double &px_in, const double &py_in)
            : gamma(g), theta(t), x(x_in), y(y_in), px(px_in), py(py_in) {}
        operator Particle() const { return Particle{gamma, theta, x, y, px, py}; }
        ConstRef &operator=(const ConstRef &) = delete;
    };

private:
    template <bool IsConst>
    class Iter {
        using C = std::conditional_t<IsConst, const Particles, Particles>;
        C *c_ {nullptr};
        std::size_t i_ {0};

    public:
        // proxy iterator in the std::vector<bool> tradition: `reference` is a
        // proxy value, good enough for range-for and sized copies
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Particle;
        using reference = std::conditional_t<IsConst, ConstRef, Ref>;
        using pointer = void;
        using difference_type = std::ptrdiff_t;

        Iter() = default;
        Iter(C *c, std::size_t i) : c_(c), i_(i) {}
        reference operator*() const { return (*c_)[i_]; }
        Iter &operator++() { ++i_; return *this; }
        Iter operator++(int) { Iter t = *this; ++i_; return t; }
        Iter &operator--() { --i_; return *this; }
        Iter &operator+=(difference_type d) { i_ += static_cast<std::size_t>(d); return *this; }
        friend Iter operator+(Iter it, difference_type d) { it += d; return it; }
        friend difference_type operator-(const Iter &a, const Iter &b)
        {
            return static_cast<difference_type>(a.i_) - static_cast<difference_type>(b.i_);
        }
        friend bool operator==(const Iter &a, const Iter &b) { return a.i_ == b.i_; }
        friend bool operator!=(const Iter &a, const Iter &b) { return a.i_ != b.i_; }
        friend bool operator<(const Iter &a, const Iter &b) { return a.i_ < b.i_; }
    };

public:
    using iterator = Iter<false>;
    using const_iterator = Iter<true>;
    using value_type = Particle;
    using size_type = std::size_t;

    Ref operator[](std::size_t i)
    {
        return Ref{gamma_[i], theta_[i], x_[i], y_[i], px_[i], py_[i]};
    }
    ConstRef operator[](std::size_t i) const
    {
        return ConstRef(gamma_[i], theta_[i], x_[i], y_[i], px_[i], py_[i]);
    }
    Ref at(std::size_t i)
    {
        if (i >= size()) { throw std::out_of_range("Particles::at"); }
        return (*this)[i];
    }
    ConstRef at(std::size_t i) const
    {
        if (i >= size()) { throw std::out_of_range("Particles::at"); }
        return (*this)[i];
    }

    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    std::size_t capacity() const { return gamma_.capacity(); }

    // note: multi-array updates are not atomic; a mid-way bad_alloc would
    // desync the six arrays (fatal for the simulation anyway)
    void resize(std::size_t n)
    {
        const std::size_t phys = padded_count(n);
        // former pad slots entering the logical range come back zeroed,
        // matching std::vector<Particle> resize semantics
        const std::size_t wipe_end = std::min(n, gamma_.size());
        const std::size_t wipe_begin = std::min(n_, wipe_end);
        for_each_array([&](dvec &a) {
            std::fill(a.begin() + wipe_begin, a.begin() + wipe_end, 0.);
            a.resize(phys);
        });
        n_ = n;
        repad();
    }
    void reserve(std::size_t n)
    {
        const std::size_t phys = padded_count(n);
        for_each_array([&](dvec &a) { a.reserve(phys); });
    }
    void shrink_to_fit()
    {
        for_each_array([&](dvec &a) { a.shrink_to_fit(); });
    }
    void clear()
    {
        for_each_array([&](dvec &a) { a.clear(); });
        n_ = 0;
    }
    void push_back(const Particle &p)
    {
        if (gamma_.size() == n_) {
            // no free pad slot: grow by one pad block, geometrically so that
            // particle-at-a-time filling (sorting, loading) stays amortized O(1)
            const std::size_t phys = n_ + particles_simd_pad;
            for_each_array([&](dvec &a) {
                if (phys > a.capacity()) { a.reserve(std::max(a.capacity() * 2, phys)); }
                a.resize(phys);
            });
        }
        gamma_[n_] = p.gamma; theta_[n_] = p.theta;
        x_[n_] = p.x; y_[n_] = p.y;
        px_[n_] = p.px; py_[n_] = p.py;
        ++n_;
        repad();
    }
    void pop_back()
    {
        --n_;
        const std::size_t phys = padded_count(n_);
        for_each_array([&](dvec &a) { a.resize(phys); });
        repad();
    }
    void swap(Particles &o) noexcept
    {
        gamma_.swap(o.gamma_); theta_.swap(o.theta_);
        x_.swap(o.x_); y_.swap(o.y_);
        px_.swap(o.px_); py_.swap(o.py_);
        std::swap(n_, o.n_);
    }
    friend void swap(Particles &a, Particles &b) noexcept { a.swap(b); }

    iterator begin() { return {this, 0}; }
    iterator end() { return {this, size()}; }
    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, size()}; }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    // raw component spans for the SIMD kernels (contiguous, 64-byte aligned)
    double *gamma() { return gamma_.data(); }
    double *theta() { return theta_.data(); }
    double *x() { return x_.data(); }
    double *y() { return y_.data(); }
    double *px() { return px_.data(); }
    double *py() { return py_.data(); }
    const double *gamma() const { return gamma_.data(); }
    const double *theta() const { return theta_.data(); }
    const double *x() const { return x_.data(); }
    const double *y() const { return y_.data(); }
    const double *px() const { return px_.data(); }
    const double *py() const { return py_.data(); }

private:
    // 64-byte alignment: cache line, and >= any current SIMD register width
    using dvec = std::vector<double, xsimd::aligned_allocator<double, 64>>;

    template <class F>
    void for_each_array(F &&f)
    {
        f(gamma_); f(theta_); f(x_); f(y_); f(px_); f(py_);
    }

    // restore the padding invariant: slots in [n_, padded) replicate particle n_-1
    void repad()
    {
        if (n_ == 0 || gamma_.size() == n_) { return; }
        for_each_array([&](dvec &a) { std::fill(a.begin() + n_, a.end(), a[n_ - 1]); });
    }

    dvec gamma_, theta_, x_, y_, px_, py_;
    std::size_t n_ {0};
};

#endif
