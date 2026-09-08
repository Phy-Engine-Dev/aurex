#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

#include <fast_io/fast_io_dsal/string_view.h>

#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Irreversible, two-terminal electrothermal protection with two
    // high-impedance voltage-sense pins. Current and power feed a first-order
    // thermal RC; voltage remains an immediate dielectric/breakdown limit.
    // DC represents the steady-state thermal operating point, whereas TR
    // advances temperature exactly once per completed time step. A zero limit
    // disables that criterion. Once tripped, the path stays open.
    struct rated_protection
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"Rated protection"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"RATED_PROTECTION"};

        double Ron{1e-9};
        double Roff{1e12};
        double Imax{};
        double Vmax{};
        double Pmax{};
        bool initial_broken{};
        double ambient_temp_c{25.0};
        double rating_reference_temp_c{25.0};
        double trip_temp_c{150.0};
        double thermal_resistance_k_per_w{1.0};
        double thermal_capacitance_j_per_k{10.0};

        bool broken{};
        ::std::uint8_t trip_mask{}; // 1=current, 2=voltage, 4=power, 8=pre-broken
        double trip_current{};
        double trip_voltage{};
        double trip_power{};
        double temperature_c{25.0};
        bool transient{};
        pin pins[4]{{{u8"line_in"}}, {{u8"line_out"}}, {{u8"sense+"}}, {{u8"sense-"}}};
        branch branches{};

        [[nodiscard]] bool valid() const noexcept
        {
            return ::std::isfinite(Ron) && Ron > 0.0 && ::std::isfinite(Roff) && Roff > Ron &&
                   ::std::isfinite(Imax) && Imax >= 0.0 && ::std::isfinite(Vmax) && Vmax >= 0.0 &&
                   ::std::isfinite(Pmax) && Pmax >= 0.0 &&
                   ::std::isfinite(ambient_temp_c) && ::std::isfinite(rating_reference_temp_c) &&
                   ::std::isfinite(trip_temp_c) && trip_temp_c > ambient_temp_c &&
                   trip_temp_c > rating_reference_temp_c &&
                   ::std::isfinite(thermal_resistance_k_per_w) && thermal_resistance_k_per_w > 0.0 &&
                   ::std::isfinite(thermal_capacitance_j_per_k) && thermal_capacitance_j_per_k > 0.0;
        }

        [[nodiscard]] double sensed_voltage() const noexcept
        {
            auto const positive{pins[2].nodes};
            auto const negative{pins[3].nodes};
            if(positive == nullptr || negative == nullptr) { return 0.0; }
            return (positive->node_information.an.voltage - negative->node_information.an.voltage).real();
        }

        [[nodiscard]] double sensed_current() const noexcept { return branches.current.real(); }
        [[nodiscard]] double sensed_power() const noexcept { return sensed_voltage() * sensed_current(); }

        [[nodiscard]] ::std::uint8_t overload_mask(bool include_thermal) const noexcept
        {
            auto const current{::std::abs(sensed_current())};
            auto const voltage{::std::abs(sensed_voltage())};
            auto const power{::std::abs(sensed_power())};
            ::std::uint8_t mask{};
            if(include_thermal && Imax > 0.0 && current > Imax) { mask |= 1u; }
            if(Vmax > 0.0 && voltage > Vmax) { mask |= 2u; }
            if(include_thermal && Pmax > 0.0 && power > Pmax) { mask |= 4u; }
            return mask;
        }

        void latch_overload(::std::uint8_t mask) noexcept
        {
            if(broken) { return; }
            if(mask != 0u)
            {
                trip_current = sensed_current();
                trip_voltage = sensed_voltage();
                trip_power = sensed_power();
                broken = true;
                trip_mask |= mask;
            }
        }

        [[nodiscard]] double thermal_load_ratio() const noexcept
        {
            double ratio{};
            if(Imax > 0.0)
            {
                auto const normalized{::std::abs(sensed_current()) / Imax};
                ratio = normalized * normalized;
            }
            if(Pmax > 0.0)
            {
                ratio = ::std::max(ratio, ::std::abs(sensed_power()) / Pmax);
            }
            return ratio;
        }

        void advance_thermal(double dt) noexcept
        {
            if(broken || !(::std::isfinite(dt) && dt > 0.0)) { return; }
            // The saved rating is calibrated at rating_reference_temp_c.
            // Changing ambient therefore changes available thermal headroom.
            auto const rise{trip_temp_c - rating_reference_temp_c};
            auto const steady{ambient_temp_c + rise * thermal_load_ratio()};
            auto const tau{thermal_resistance_k_per_w * thermal_capacitance_j_per_k};
            auto const decay{::std::exp(-dt / tau)};
            temperature_c = steady + (temperature_c - steady) * decay;
            if(!::std::isfinite(temperature_c))
            {
                temperature_c = trip_temp_c;
            }
            if(temperature_c >= trip_temp_c)
            {
                auto mask{overload_mask(true)};
                if((mask & (1u | 4u)) == 0u)
                {
                    // Numerical crossing was driven by the just-completed
                    // current/power sample; retain an explicit thermal cause.
                    mask |= Imax > 0.0 ? 1u : 4u;
                }
                latch_overload(mask & (1u | 4u));
            }
        }
    };

    static_assert(model<rated_protection>);

    inline bool set_attribute_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                     ::std::size_t index, variant value) noexcept
    {
        if(index == 5)
        {
            if(value.type != variant_type::boolean) { return false; }
            p.initial_broken = value.boolean;
            return true;
        }
        if(value.type != variant_type::d || !::std::isfinite(value.d)) { return false; }
        switch(index)
        {
            case 0: if(value.d <= 0.0) return false; p.Ron = value.d; return true;
            case 1: if(value.d <= p.Ron) return false; p.Roff = value.d; return true;
            case 2: if(value.d < 0.0) return false; p.Imax = value.d; return true;
            case 3: if(value.d < 0.0) return false; p.Vmax = value.d; return true;
            case 4: if(value.d < 0.0) return false; p.Pmax = value.d; return true;
            case 6: p.ambient_temp_c = value.d; return true;
            case 7: p.rating_reference_temp_c = value.d; return true;
            case 8: p.trip_temp_c = value.d; return true;
            case 9: if(value.d <= 0.0) return false; p.thermal_resistance_k_per_w = value.d; return true;
            case 10: if(value.d <= 0.0) return false; p.thermal_capacitance_j_per_k = value.d; return true;
            default: return false;
        }
    }

    inline variant get_attribute_define(model_reserve_type_t<rated_protection>, rated_protection const& p,
                                        ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return {.d{p.Ron}, .type{variant_type::d}};
            case 1: return {.d{p.Roff}, .type{variant_type::d}};
            case 2: return {.d{p.Imax}, .type{variant_type::d}};
            case 3: return {.d{p.Vmax}, .type{variant_type::d}};
            case 4: return {.d{p.Pmax}, .type{variant_type::d}};
            case 5: return {.boolean{p.initial_broken}, .type{variant_type::boolean}};
            case 6: return {.d{p.ambient_temp_c}, .type{variant_type::d}};
            case 7: return {.d{p.rating_reference_temp_c}, .type{variant_type::d}};
            case 8: return {.d{p.trip_temp_c}, .type{variant_type::d}};
            case 9: return {.d{p.thermal_resistance_k_per_w}, .type{variant_type::d}};
            case 10: return {.d{p.thermal_capacitance_j_per_k}, .type{variant_type::d}};
            case 11: return {.boolean{p.broken}, .type{variant_type::boolean}};
            case 12: return {.ui8{p.trip_mask}, .type{variant_type::ui8}};
            case 13: return {.d{p.sensed_current()}, .type{variant_type::d}};
            case 14: return {.d{p.sensed_voltage()}, .type{variant_type::d}};
            case 15: return {.d{p.sensed_power()}, .type{variant_type::d}};
            case 16: return {.d{p.trip_current}, .type{variant_type::d}};
            case 17: return {.d{p.trip_voltage}, .type{variant_type::d}};
            case 18: return {.d{p.trip_power}, .type{variant_type::d}};
            case 19: return {.d{p.temperature_c}, .type{variant_type::d}};
            case 20: return {.d{p.thermal_capacitance_j_per_k *
                                ::std::max(0.0, p.temperature_c - p.ambient_temp_c)},
                             .type{variant_type::d}};
            default: return {};
        }
    }

    inline constexpr ::fast_io::u8string_view
        get_attribute_name_define(model_reserve_type_t<rated_protection>, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return u8"Ron";
            case 1: return u8"Roff";
            case 2: return u8"Imax";
            case 3: return u8"Vmax";
            case 4: return u8"Pmax";
            case 5: return u8"InitialBroken";
            case 6: return u8"AmbientTempC";
            case 7: return u8"RatingReferenceTempC";
            case 8: return u8"TripTempC";
            case 9: return u8"ThermalResistanceKPerW";
            case 10: return u8"ThermalCapacitanceJPerK";
            case 11: return u8"Broken";
            case 12: return u8"TripMask";
            case 13: return u8"Current";
            case 14: return u8"Voltage";
            case 15: return u8"Power";
            case 16: return u8"TripCurrent";
            case 17: return u8"TripVoltage";
            case 18: return u8"TripPower";
            case 19: return u8"TemperatureC";
            case 20: return u8"ThermalEnergyJ";
            default: return {};
        }
    }

    inline bool init_define(model_reserve_type_t<rated_protection>, rated_protection& p) noexcept
    {
        if(!p.valid()) { return false; }
        p.broken = p.initial_broken;
        p.trip_mask = p.initial_broken ? 8u : 0u;
        p.temperature_c = p.ambient_temp_c;
        p.transient = false;
        return true;
    }

    inline bool prepare_foundation_define(model_reserve_type_t<rated_protection>, rated_protection& p) noexcept
    {
        return p.valid();
    }

    inline void rated_protection_stamp(rated_protection const& p, ::phy_engine::MNA::MNA& mna) noexcept
    {
        auto const a{p.pins[0].nodes};
        auto const b{p.pins[1].nodes};
        if(a == nullptr || b == nullptr) { return; }
        auto const k{p.branches.index};
        mna.B_ref(a->node_index, k) += 1.0;
        mna.B_ref(b->node_index, k) -= 1.0;
        mna.C_ref(k, a->node_index) += 1.0;
        mna.C_ref(k, b->node_index) -= 1.0;
        mna.D_ref(k, k) -= p.broken ? p.Roff : p.Ron;
    }

    inline bool iterate_dc_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                  ::phy_engine::MNA::MNA& mna) noexcept
    {
        if(!p.valid()) { return false; }
        p.transient = false;
        p.latch_overload(p.overload_mask(true));
        rated_protection_stamp(p, mna);
        return true;
    }

    inline bool prepare_tr_define(model_reserve_type_t<rated_protection>, rated_protection& p) noexcept
    {
        if(!p.valid()) { return false; }
        p.transient = true;
        return true;
    }

    inline bool step_changed_tr_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                       [[maybe_unused]] double old_step, double new_step) noexcept
    {
        if(!(::std::isfinite(new_step) && new_step > 0.0)) { return false; }
        p.advance_thermal(new_step);
        return p.valid() && ::std::isfinite(p.temperature_c);
    }

    inline bool iterate_tr_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                  ::phy_engine::MNA::MNA& mna, [[maybe_unused]] double time) noexcept
    {
        if(!p.valid()) { return false; }
        p.transient = true;
        p.latch_overload(p.overload_mask(false));
        rated_protection_stamp(p, mna);
        return true;
    }

    inline bool iterate_trop_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                    ::phy_engine::MNA::MNA& mna) noexcept
    {
        if(!p.valid()) { return false; }
        rated_protection_stamp(p, mna);
        return true;
    }

    inline bool iterate_ac_define(model_reserve_type_t<rated_protection>, rated_protection& p,
                                  ::phy_engine::MNA::MNA& mna, [[maybe_unused]] double omega) noexcept
    {
        if(!p.valid()) { return false; }
        rated_protection_stamp(p, mna);
        return true;
    }

    inline bool check_convergence_define(model_reserve_type_t<rated_protection>, rated_protection const& p) noexcept
    {
        return p.broken || p.overload_mask(!p.transient) == 0u;
    }

    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<rated_protection>, rated_protection& p) noexcept
    {
        return {p.pins, 4};
    }

    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<rated_protection>, rated_protection& p) noexcept
    {
        return {__builtin_addressof(p.branches), 1};
    }
}
