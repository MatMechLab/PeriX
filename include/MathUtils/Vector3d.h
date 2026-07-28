//****************************************************************
//* This file is part of the PeriX framework
//* Peridynamics framework for multiphysics simulation (X)
//* All rights reserved, Yang Bai/MM-Lab@CopyRight 2026-present
//* https://github.com/MatMechLab/PeriX
//* Licensed under GNU GPLv3, please see LICENSE for details
//* https://www.gnu.org/licenses/gpl-3.0.en.html
//****************************************************************
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++ Author  : Yang Bai
//+++ Date    : 2026.04.14
//+++ Function: defines the vector with only 3-components
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#pragma once

#include <cmath>
#include <cstdlib>
#include <type_traits>

class Vector3d {
public:
    // ------------------------------------------------------------
    // constructors / assignment
    // ------------------------------------------------------------
    constexpr Vector3d() noexcept
        : m_Vals{0.0, 0.0, 0.0} {}

    // keep implicit conversion for compatibility with existing code
    constexpr Vector3d(double val) noexcept
        : m_Vals{val, val, val} {}

    constexpr Vector3d(const Vector3d&) noexcept = default;
    constexpr Vector3d& operator=(const Vector3d&) noexcept = default;

    constexpr Vector3d& operator=(double val) noexcept {
        m_Vals[0] = val;
        m_Vals[1] = val;
        m_Vals[2] = val;
        return *this;
    }

    // ------------------------------------------------------------
    // element access (1-based, kept for compatibility)
    // ------------------------------------------------------------
    [[nodiscard]] constexpr double& operator()(int i) noexcept {
#ifndef NDEBUG
        if (static_cast<unsigned>(i - 1) >= 3u) std::abort();
#endif
        return m_Vals[i - 1];
    }

    [[nodiscard]] constexpr const double& operator()(int i) const noexcept {
#ifndef NDEBUG
        if (static_cast<unsigned>(i - 1) >= 3u) std::abort();
#endif
        return m_Vals[i - 1];
    }

    // optional raw accessors for hot loops
    [[nodiscard]] constexpr double* data() noexcept { return m_Vals; }
    [[nodiscard]] constexpr const double* data() const noexcept { return m_Vals; }

    // ------------------------------------------------------------
    // arithmetic with scalar
    // ------------------------------------------------------------
    [[nodiscard]] constexpr Vector3d operator+(double val) const noexcept {
        return Vector3d(m_Vals[0] + val, m_Vals[1] + val, m_Vals[2] + val, tag{});
    }

    constexpr Vector3d& operator+=(double val) noexcept {
        m_Vals[0] += val;
        m_Vals[1] += val;
        m_Vals[2] += val;
        return *this;
    }

    [[nodiscard]] constexpr Vector3d operator-(double val) const noexcept {
        return Vector3d(m_Vals[0] - val, m_Vals[1] - val, m_Vals[2] - val, tag{});
    }

    constexpr Vector3d& operator-=(double val) noexcept {
        m_Vals[0] -= val;
        m_Vals[1] -= val;
        m_Vals[2] -= val;
        return *this;
    }

    [[nodiscard]] constexpr Vector3d operator*(double val) const noexcept {
        return Vector3d(m_Vals[0] * val, m_Vals[1] * val, m_Vals[2] * val, tag{});
    }

    constexpr Vector3d& operator*=(double val) noexcept {
        m_Vals[0] *= val;
        m_Vals[1] *= val;
        m_Vals[2] *= val;
        return *this;
    }

    [[nodiscard]] inline Vector3d operator/(double val) const noexcept {
#ifndef NDEBUG
        if (std::abs(val) < 1.0e-15) std::abort();
#endif
        const double inv = 1.0 / val;
        return Vector3d(m_Vals[0] * inv, m_Vals[1] * inv, m_Vals[2] * inv, tag{});
    }

    // ------------------------------------------------------------
    // arithmetic with vector
    // ------------------------------------------------------------
    [[nodiscard]] constexpr Vector3d operator+(const Vector3d& a) const noexcept {
        return Vector3d(m_Vals[0] + a.m_Vals[0],
                        m_Vals[1] + a.m_Vals[1],
                        m_Vals[2] + a.m_Vals[2], tag{});
    }

    constexpr Vector3d& operator+=(const Vector3d& a) noexcept {
        m_Vals[0] += a.m_Vals[0];
        m_Vals[1] += a.m_Vals[1];
        m_Vals[2] += a.m_Vals[2];
        return *this;
    }

    [[nodiscard]] constexpr Vector3d operator-(const Vector3d& a) const noexcept {
        return Vector3d(m_Vals[0] - a.m_Vals[0],
                        m_Vals[1] - a.m_Vals[1],
                        m_Vals[2] - a.m_Vals[2], tag{});
    }

    constexpr Vector3d& operator-=(const Vector3d& a) noexcept {
        m_Vals[0] -= a.m_Vals[0];
        m_Vals[1] -= a.m_Vals[1];
        m_Vals[2] -= a.m_Vals[2];
        return *this;
    }

    // dot product
    [[nodiscard]] constexpr double operator*(const Vector3d& a) const noexcept {
        return m_Vals[0] * a.m_Vals[0]
             + m_Vals[1] * a.m_Vals[1]
             + m_Vals[2] * a.m_Vals[2];
    }

    [[nodiscard]] constexpr double odot(const Vector3d& a) const noexcept {
        return (*this) * a;
    }

    friend constexpr Vector3d operator*(double val, const Vector3d& a) noexcept {
        return Vector3d(val * a.m_Vals[0],
                        val * a.m_Vals[1],
                        val * a.m_Vals[2], tag{});
    }

    // ------------------------------------------------------------
    // norms
    // ------------------------------------------------------------
    [[nodiscard]] constexpr double normsq() const noexcept {
        return m_Vals[0] * m_Vals[0]
             + m_Vals[1] * m_Vals[1]
             + m_Vals[2] * m_Vals[2];
    }

    [[nodiscard]] inline double norm() const noexcept {
        return std::sqrt(normsq());
    }

private:
    struct tag {};
    constexpr Vector3d(double x, double y, double z, tag) noexcept
        : m_Vals{x, y, z} {}

private:
    double m_Vals[3];
};

static_assert(std::is_trivially_copyable_v<Vector3d>,
              "Vector3d should stay trivially copyable");