#pragma once

#include <cmath>
#include <limits>

namespace phy_engine::model::logic_level
{
    // Analog solves are allowed to finish a few ulps (and a small solver
    // residual) away from an ideal digital rail.  Keep that numerical error
    // out of the Ll..Hl hysteresis band without turning a genuine
    // intermediate voltage into a rail.
    inline constexpr double rail_tolerance(double Ll, double Hl) noexcept
    {
        if(!::std::isfinite(Ll) || !::std::isfinite(Hl) || !(Hl > Ll)) { return 0.0; }

        double const span{Hl - Ll};
        double const abs_ll{Ll < 0.0 ? -Ll : Ll};
        double const abs_hl{Hl < 0.0 ? -Hl : Hl};
        double const scale{abs_ll > abs_hl ? abs_ll : abs_hl};
        double const residual_tolerance{span * 1e-6};
        double const roundoff_tolerance{(scale > span ? scale : span) * 64.0 * ::std::numeric_limits<double>::epsilon()};
        double const candidate{residual_tolerance > roundoff_tolerance ? residual_tolerance : roundoff_tolerance};

        // Even for unusually offset or very narrow rails, leave at least 75%
        // of the configured threshold interval as an unambiguous hysteresis
        // band.  Invalid/reversed rails retain the old exact comparisons.
        double const maximum{span * 0.125};
        return candidate < maximum ? candidate : maximum;
    }

    inline constexpr bool is_high(double voltage, double Ll, double Hl) noexcept
    {
        return ::std::isfinite(voltage) && voltage >= Hl - rail_tolerance(Ll, Hl);
    }

    inline constexpr bool is_low(double voltage, double Ll, double Hl) noexcept
    {
        return ::std::isfinite(voltage) && voltage <= Ll + rail_tolerance(Ll, Hl);
    }
}  // namespace phy_engine::model::logic_level
