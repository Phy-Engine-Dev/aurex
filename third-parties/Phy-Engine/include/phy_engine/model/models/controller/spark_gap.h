#pragma once

#include <cmath>
#include <complex>

#include <fast_io/fast_io_dsal/string_view.h>

#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Two-state spark gap. Breakdown is evaluated during nonlinear iteration;
    // hold-current extinction is evaluated once at each transient boundary so
    // an off/on decision cannot chatter inside a single MNA solve.
    struct spark_gap
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"Spark gap"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"SPARK_GAP"};

        double breakdown_v{1000.0};
        double arc_r{1.0};
        double holding_current{0.0};
        double off_r{1e12};
        bool initial_conducting{};
        bool conducting{};
        pin pins[2]{{{u8"A"}}, {{u8"B"}}};
        branch branches{};

        [[nodiscard]] bool valid() const noexcept
        {
            return ::std::isfinite(breakdown_v) && breakdown_v > 0.0 &&
                   ::std::isfinite(arc_r) && arc_r > 0.0 &&
                   ::std::isfinite(holding_current) && holding_current >= 0.0 &&
                   ::std::isfinite(off_r) && off_r > arc_r;
        }

        [[nodiscard]] double voltage() const noexcept
        {
            if(pins[0].nodes == nullptr || pins[1].nodes == nullptr) { return 0.0; }
            return (pins[0].nodes->node_information.an.voltage -
                    pins[1].nodes->node_information.an.voltage).real();
        }
        [[nodiscard]] double current() const noexcept { return branches.current.real(); }
    };

    static_assert(model<spark_gap>);

    inline bool set_attribute_define(model_reserve_type_t<spark_gap>, spark_gap& g,
                                     ::std::size_t index, variant value) noexcept
    {
        if(index == 4)
        {
            if(value.type != variant_type::boolean) { return false; }
            g.initial_conducting = value.boolean;
            return true;
        }
        if(value.type != variant_type::d || !::std::isfinite(value.d)) { return false; }
        switch(index)
        {
            case 0: if(value.d <= 0.0) return false; g.breakdown_v = value.d; return true;
            case 1: if(value.d <= 0.0) return false; g.arc_r = value.d; return true;
            case 2: if(value.d < 0.0) return false; g.holding_current = value.d; return true;
            case 3: if(value.d <= g.arc_r) return false; g.off_r = value.d; return true;
            default: return false;
        }
    }

    inline variant get_attribute_define(model_reserve_type_t<spark_gap>, spark_gap const& g,
                                        ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return {.d{g.breakdown_v}, .type{variant_type::d}};
            case 1: return {.d{g.arc_r}, .type{variant_type::d}};
            case 2: return {.d{g.holding_current}, .type{variant_type::d}};
            case 3: return {.d{g.off_r}, .type{variant_type::d}};
            case 4: return {.boolean{g.initial_conducting}, .type{variant_type::boolean}};
            case 5: return {.boolean{g.conducting}, .type{variant_type::boolean}};
            case 6: return {.d{g.current()}, .type{variant_type::d}};
            case 7: return {.d{g.voltage()}, .type{variant_type::d}};
            case 8: return {.d{g.voltage() * g.current()}, .type{variant_type::d}};
            default: return {};
        }
    }

    inline constexpr ::fast_io::u8string_view
        get_attribute_name_define(model_reserve_type_t<spark_gap>, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return u8"BreakdownVoltage";
            case 1: return u8"ArcResistance";
            case 2: return u8"HoldingCurrent";
            case 3: return u8"OffResistance";
            case 4: return u8"InitialConducting";
            case 5: return u8"Conducting";
            case 6: return u8"Current";
            case 7: return u8"Voltage";
            case 8: return u8"Power";
            default: return {};
        }
    }

    inline bool init_define(model_reserve_type_t<spark_gap>, spark_gap& g) noexcept
    {
        if(!g.valid()) { return false; }
        g.conducting = g.initial_conducting;
        return true;
    }

    inline bool prepare_foundation_define(model_reserve_type_t<spark_gap>, spark_gap const& g) noexcept
    {
        return g.valid();
    }

    inline void spark_gap_stamp(spark_gap const& g, MNA::MNA& mna) noexcept
    {
        if(g.pins[0].nodes == nullptr || g.pins[1].nodes == nullptr) { return; }
        auto const k{g.branches.index};
        mna.B_ref(g.pins[0].nodes->node_index, k) += 1.0;
        mna.B_ref(g.pins[1].nodes->node_index, k) -= 1.0;
        mna.C_ref(k, g.pins[0].nodes->node_index) += 1.0;
        mna.C_ref(k, g.pins[1].nodes->node_index) -= 1.0;
        mna.D_ref(k, k) -= g.conducting ? g.arc_r : g.off_r;
    }

    inline bool iterate_dc_define(model_reserve_type_t<spark_gap>, spark_gap& g,
                                  MNA::MNA& mna) noexcept
    {
        if(!g.valid()) { return false; }
        if(!g.conducting && ::std::abs(g.voltage()) >= g.breakdown_v) { g.conducting = true; }
        spark_gap_stamp(g, mna);
        return true;
    }

    inline bool step_changed_tr_define(model_reserve_type_t<spark_gap>, spark_gap& g,
                                       [[maybe_unused]] double old_step,
                                       [[maybe_unused]] double new_step) noexcept
    {
        if(g.conducting && ::std::abs(g.current()) < g.holding_current) { g.conducting = false; }
        return true;
    }

    inline bool iterate_tr_define(model_reserve_type_t<spark_gap>, spark_gap& g,
                                  MNA::MNA& mna, [[maybe_unused]] double time) noexcept
    {
        return iterate_dc_define(model_reserve_type<spark_gap>, g, mna);
    }

    inline bool iterate_trop_define(model_reserve_type_t<spark_gap>, spark_gap& g,
                                    MNA::MNA& mna) noexcept
    {
        return iterate_dc_define(model_reserve_type<spark_gap>, g, mna);
    }

    inline bool iterate_ac_define(model_reserve_type_t<spark_gap>, spark_gap const& g,
                                  MNA::MNA& mna, [[maybe_unused]] double omega) noexcept
    {
        if(!g.valid()) { return false; }
        spark_gap_stamp(g, mna);
        return true;
    }

    inline bool check_convergence_define(model_reserve_type_t<spark_gap>, spark_gap const& g) noexcept
    {
        return g.conducting || ::std::abs(g.voltage()) < g.breakdown_v;
    }

    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<spark_gap>, spark_gap& g) noexcept
    {
        return {g.pins, 2};
    }

    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<spark_gap>, spark_gap& g) noexcept
    {
        return {__builtin_addressof(g.branches), 1};
    }
}
