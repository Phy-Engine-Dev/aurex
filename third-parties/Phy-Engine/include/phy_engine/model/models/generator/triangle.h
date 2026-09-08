#pragma once
#include <numbers>
#include <cmath>
#include <fast_io/fast_io_dsal/string_view.h>
#include "../../model_refs/base.h"

namespace phy_engine::model
{
    struct triangle_gen
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"Triangle Wave Generator"};

        inline static constexpr ::phy_engine::model::model_device_type device_type{::phy_engine::model::model_device_type::linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"TRIANGLE"};

        double Vh{5.0};
        double Vl{0.0};
        double freq{1e3};
        double phase{0.0};  // radians
        double duty{0.5};   // fraction of period spent rising
        double series_resistance{0.0}; // ohms; legacy source remains ideal

        ::phy_engine::model::pin pins[2]{{{u8"+"}}, {{u8"-"}}};
        ::phy_engine::model::branch branches{};
    };

    static_assert(::phy_engine::model::model<triangle_gen>);

    inline constexpr bool set_attribute_define(::phy_engine::model::model_reserve_type_t<triangle_gen>,
                                               triangle_gen& g,
                                               ::std::size_t idx,
                                               ::phy_engine::model::variant vi) noexcept
    {
        switch(idx)
        {
            case 0:
                if(vi.type != ::phy_engine::model::variant_type::d) [[unlikely]] { return false; }
                g.Vh = vi.d;
                return true;
            case 1:
                if(vi.type != ::phy_engine::model::variant_type::d) [[unlikely]] { return false; }
                g.Vl = vi.d;
                return true;
            case 2:
                if(vi.type != ::phy_engine::model::variant_type::d) [[unlikely]] { return false; }
                g.freq = vi.d;
                return true;
            case 3:
                if(vi.type != ::phy_engine::model::variant_type::d) [[unlikely]] { return false; }
                g.phase = vi.d;
                return true;
            case 4:
                if(vi.type != variant_type::d || !std::isfinite(vi.d) || vi.d <= 0 || vi.d >= 1) return false;
                g.duty = vi.d;
                return true;
            case 5:
                if(vi.type != variant_type::d || !std::isfinite(vi.d) || vi.d < 0) return false;
                g.series_resistance = vi.d;
                return true;
            default: return false;
        }
        return false;
    }

    static_assert(::phy_engine::model::defines::has_set_attribute<triangle_gen>);

    inline constexpr ::phy_engine::model::variant
        get_attribute_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, triangle_gen const& g, ::std::size_t idx) noexcept
    {
        switch(idx)
        {
            case 0: return {.d{g.Vh}, .type{::phy_engine::model::variant_type::d}};
            case 1: return {.d{g.Vl}, .type{::phy_engine::model::variant_type::d}};
            case 2: return {.d{g.freq}, .type{::phy_engine::model::variant_type::d}};
            case 3: return {.d{g.phase}, .type{::phy_engine::model::variant_type::d}};
            case 4: return {.d{g.duty}, .type{variant_type::d}};
            case 5: return {.d{g.series_resistance}, .type{variant_type::d}};
            default: return {};
        }
        return {};
    }

    static_assert(::phy_engine::model::defines::has_get_attribute<triangle_gen>);

    inline constexpr ::fast_io::u8string_view get_attribute_name_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, ::std::size_t idx) noexcept
    {
        switch(idx)
        {
            case 0: return u8"Vh";
            case 1: return u8"Vl";
            case 2: return u8"freq";
            case 3: return u8"phase";
            case 4: return u8"duty";
            case 5: return u8"Rseries";
            default: return {};
        }
        return {};
    }

    static_assert(::phy_engine::model::defines::has_get_attribute_name<triangle_gen>);

    inline bool prepare_foundation_define(model_reserve_type_t<triangle_gen>, triangle_gen const& g) noexcept
    {
        return std::isfinite(g.Vh) && std::isfinite(g.Vl) && std::isfinite(g.phase)
            && std::isfinite(g.freq) && g.freq > 0 && std::isfinite(1.0 / g.freq)
            && std::isfinite(g.duty) && g.duty > 0 && g.duty < 1
            && std::isfinite(g.series_resistance) && g.series_resistance >= 0;
    }

    // phase=0 starts at Vl; negative phases wrap into the same periodic waveform.
    inline double triangle_voltage(triangle_gen const& g, double time) noexcept
    {
        double const period = 1.0 / g.freq;
        double phase = std::fmod(time, period) / period + std::fmod(g.phase / (2 * std::numbers::pi), 1.0);
        phase -= std::floor(phase);
        double const fraction = phase < g.duty ? phase / g.duty : (1.0 - phase) / (1.0 - g.duty);
        return (1.0 - fraction) * g.Vl + fraction * g.Vh;
    }

    inline bool
        iterate_tr_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, triangle_gen const& g, ::phy_engine::MNA::MNA& mna, double tTime) noexcept
    {
        auto const node_P{g.pins[0].nodes};
        auto const node_M{g.pins[1].nodes};
        if(node_P && node_M) [[likely]]
        {
            double const val = triangle_voltage(g, tTime);
            if(!std::isfinite(val)) return false;
            auto const k{g.branches.index};
            mna.B_ref(node_P->node_index, k) = 1.0;
            mna.B_ref(node_M->node_index, k) = -1.0;
            mna.C_ref(k, node_P->node_index) = 1.0;
            mna.C_ref(k, node_M->node_index) = -1.0;
            mna.E_ref(k) = val;
            mna.D_ref(k, k) -= g.series_resistance;
        }
        return true;
    }

    static_assert(::phy_engine::model::defines::can_iterate_tr<triangle_gen>);

    inline bool
        iterate_dc_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, triangle_gen const& g, ::phy_engine::MNA::MNA& mna) noexcept
    {
        // For DC operating point, use the waveform value at t=0.
        return iterate_tr_define(::phy_engine::model::model_reserve_type_t<triangle_gen>{}, g, mna, 0.0);
    }

    static_assert(::phy_engine::model::defines::can_iterate_dc<triangle_gen>);

    inline constexpr bool
        iterate_ac_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, triangle_gen const& g, ::phy_engine::MNA::MNA& mna, [[maybe_unused]] double omega) noexcept
    {
        // No defined small-signal AC excitation for time-domain generators; treat as AC=0V.
        auto const node_P{g.pins[0].nodes};
        auto const node_M{g.pins[1].nodes};
        if(node_P && node_M) [[likely]]
        {
            auto const k{g.branches.index};
            mna.B_ref(node_P->node_index, k) = 1.0;
            mna.B_ref(node_M->node_index, k) = -1.0;
            mna.C_ref(k, node_P->node_index) = 1.0;
            mna.C_ref(k, node_M->node_index) = -1.0;
            mna.D_ref(k, k) -= g.series_resistance;
            // mna.E_ref(k) += 0.0;
        }
        return true;
    }

    static_assert(::phy_engine::model::defines::can_iterate_ac<triangle_gen>);

    inline constexpr ::phy_engine::model::pin_view generate_pin_view_define(::phy_engine::model::model_reserve_type_t<triangle_gen>, triangle_gen& g) noexcept
    {
        return {g.pins, 2};
    }

    static_assert(::phy_engine::model::defines::can_generate_pin_view<triangle_gen>);

    inline constexpr ::phy_engine::model::branch_view generate_branch_view_define(::phy_engine::model::model_reserve_type_t<triangle_gen>,
                                                                                  triangle_gen& g) noexcept
    {
        return {__builtin_addressof(g.branches), 1};
    }

    static_assert(::phy_engine::model::defines::can_generate_branch_view<triangle_gen>);
}  // namespace phy_engine::model
