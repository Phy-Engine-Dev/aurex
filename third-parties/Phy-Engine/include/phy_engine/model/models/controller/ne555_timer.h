#pragma once

#include <cmath>

#include <fast_io/fast_io_dsal/string_view.h>

#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Behavioral bipolar 555 core. The public eight-terminal topology is
    // retained: comparators update an SR latch, OUT is a finite-resistance
    // voltage driver, and DIS is an open-collector discharge path.
    struct ne555_timer
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"NE555 timer"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"NE555"};

        double low_v{};
        double high_v{3.0};
        double output_r{1e-3};
        double discharge_on_r{1e-3};
        double discharge_off_r{1e12};
        bool internal_control{true};
        bool internal_reset_pullup{true};

        bool latched_high{};
        // A fixed TR time point may commit at most one causal SR-latch edge.
        // Newton then re-solves the resulting topology without feeding an
        // unconverged candidate voltage back into the latch.
        bool tr_transition_available{true};
        // Limits the state-commit hook to TR; DC/OP keep their original loop.
        bool tr_iteration_active{};
        double trigger_threshold{};
        double threshold_threshold{};
        pin pins[8]{{{u8"vcc"}}, {{u8"dis"}}, {{u8"thr"}}, {{u8"ctrl"}},
                    {{u8"trig"}}, {{u8"out"}}, {{u8"reset"}}, {{u8"ground"}}};
        branch branches[2]{}; // OUT->GND, DIS->GND

        [[nodiscard]] bool valid() const noexcept
        {
            return ::std::isfinite(low_v) && ::std::isfinite(high_v) && low_v <= high_v &&
                   ::std::isfinite(output_r) && output_r > 0.0 &&
                   ::std::isfinite(discharge_on_r) && discharge_on_r > 0.0 &&
                   ::std::isfinite(discharge_off_r) && discharge_off_r > discharge_on_r;
        }

        [[nodiscard]] double voltage(::std::size_t index) const noexcept
        {
            auto const node{pins[index].nodes};
            return node == nullptr ? 0.0 : node->node_information.an.voltage.real();
        }

        [[nodiscard]] bool desired_latch() noexcept
        {
            double const ground{voltage(7)};
            double const supply{voltage(0) - ground};
            double const upper{internal_control ? (2.0 / 3.0) * supply : voltage(3) - ground};
            threshold_threshold = upper;
            trigger_threshold = .5 * upper;
            bool const reset_high{internal_reset_pullup || (voltage(6) - ground) > .5 * supply};
            if(!(supply > 0.0) || !reset_high) { return false; }
            double const trigger{voltage(4) - ground};
            double const threshold{voltage(2) - ground};
            if(trigger < trigger_threshold) { return true; }
            if(threshold > threshold_threshold) { return false; }
            return latched_high;
        }
    };

    static_assert(model<ne555_timer>);

    inline bool set_attribute_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer,
                                     ::std::size_t index, variant value) noexcept
    {
        if(index == 5 || index == 6)
        {
            if(value.type != variant_type::boolean) { return false; }
            if(index == 5) timer.internal_control = value.boolean;
            else timer.internal_reset_pullup = value.boolean;
            return true;
        }
        if(value.type != variant_type::d || !::std::isfinite(value.d)) { return false; }
        switch(index)
        {
            case 0: timer.low_v = value.d; return true;
            case 1: timer.high_v = value.d; return true;
            case 2: if(value.d <= 0.0) return false; timer.output_r = value.d; return true;
            case 3: if(value.d <= 0.0) return false; timer.discharge_on_r = value.d; return true;
            case 4: if(value.d <= timer.discharge_on_r) return false; timer.discharge_off_r = value.d; return true;
            default: return false;
        }
    }

    inline variant get_attribute_define(model_reserve_type_t<ne555_timer>, ne555_timer const& timer,
                                        ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return {.d{timer.low_v}, .type{variant_type::d}};
            case 1: return {.d{timer.high_v}, .type{variant_type::d}};
            case 2: return {.d{timer.output_r}, .type{variant_type::d}};
            case 3: return {.d{timer.discharge_on_r}, .type{variant_type::d}};
            case 4: return {.d{timer.discharge_off_r}, .type{variant_type::d}};
            case 5: return {.boolean{timer.internal_control}, .type{variant_type::boolean}};
            case 6: return {.boolean{timer.internal_reset_pullup}, .type{variant_type::boolean}};
            case 7: return {.boolean{timer.latched_high}, .type{variant_type::boolean}};
            case 8: return {.d{timer.trigger_threshold}, .type{variant_type::d}};
            case 9: return {.d{timer.threshold_threshold}, .type{variant_type::d}};
            case 10: return {.d{timer.branches[0].current.real()}, .type{variant_type::d}};
            case 11: return {.d{timer.branches[1].current.real()}, .type{variant_type::d}};
            default: return {};
        }
    }

    inline constexpr ::fast_io::u8string_view
        get_attribute_name_define(model_reserve_type_t<ne555_timer>, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return u8"LowV"; case 1: return u8"HighV";
            case 2: return u8"OutputR"; case 3: return u8"DischargeOnR";
            case 4: return u8"DischargeOffR"; case 5: return u8"InternalControl";
            case 6: return u8"InternalResetPullup"; case 7: return u8"LatchedHigh";
            case 8: return u8"TriggerThreshold"; case 9: return u8"ThresholdThreshold";
            case 10: return u8"OutputCurrent"; case 11: return u8"DischargeCurrent";
            default: return {};
        }
    }

    inline bool prepare_foundation_define(model_reserve_type_t<ne555_timer>, ne555_timer const& timer) noexcept
    {
        return timer.valid();
    }

    inline void ne555_branch_stamp(ne555_timer const& timer, MNA::MNA& mna,
                                   ::std::size_t branch_index, ::std::size_t positive_pin,
                                   double resistance, double source_voltage) noexcept
    {
        auto const positive{timer.pins[positive_pin].nodes};
        auto const ground{timer.pins[7].nodes};
        if(positive == nullptr || ground == nullptr) { return; }
        auto const k{timer.branches[branch_index].index};
        mna.B_ref(positive->node_index, k) += 1.0;
        mna.B_ref(ground->node_index, k) -= 1.0;
        mna.C_ref(k, positive->node_index) += 1.0;
        mna.C_ref(k, ground->node_index) -= 1.0;
        mna.D_ref(k, k) -= resistance;
        mna.E_ref(k) += source_voltage;
    }

    inline void ne555_conductance_stamp(ne555_timer const& timer, MNA::MNA& mna,
                                        ::std::size_t a_index, ::std::size_t b_index,
                                        double conductance) noexcept
    {
        auto const a{timer.pins[a_index].nodes};
        auto const b{timer.pins[b_index].nodes};
        if(a == nullptr || b == nullptr || a == b) { return; }
        auto const ai{a->node_index};
        auto const bi{b->node_index};
        mna.G_ref(ai, ai) += conductance;
        mna.G_ref(ai, bi) -= conductance;
        mna.G_ref(bi, ai) -= conductance;
        mna.G_ref(bi, bi) += conductance;
    }

    inline void ne555_input_stamp(ne555_timer const& timer, MNA::MNA& mna) noexcept
    {
        // The comparator inputs are high impedance, but a truly infinite
        // impedance leaves an otherwise unwired public pin as an empty MNA
        // row. Keep a SPICE-style 1 Tohm input leakage in the device itself
        // so a native NE555 remains solvable even when global GMIN is zero.
        // This is far below the loading of ordinary 555 timing networks.
        constexpr double input_g{1e-12};
        ne555_conductance_stamp(timer, mna, 2, 7, input_g); // THR
        ne555_conductance_stamp(timer, mna, 4, 7, input_g); // TRIG

        // A bipolar 555 always retains its three internal 5 kohm divider
        // resistors.  Wiring CTRL does not remove that divider: a bypass
        // capacitor must charge to 2/3 VCC, while a low-impedance external
        // source may still override it.  The CTRL node sees one 5 kohm
        // resistor to VCC and two series 5 kohm resistors to ground.
        constexpr double divider_resistance{5000.0};
        ne555_conductance_stamp(timer, mna, 3, 0, 1.0 / divider_resistance);
        ne555_conductance_stamp(timer, mna, 3, 7, 1.0 / (2.0 * divider_resistance));

        if(timer.internal_reset_pullup)
        {
            ne555_conductance_stamp(timer, mna, 6, 0, input_g);
        }
        else
        {
            ne555_conductance_stamp(timer, mna, 6, 7, input_g);
        }
    }

    inline bool iterate_dc_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer,
                                  MNA::MNA& mna) noexcept
    {
        if(!timer.valid()) { return false; }
        timer.tr_iteration_active = false;
        timer.latched_high = timer.desired_latch();
        ne555_input_stamp(timer, mna);
        ne555_branch_stamp(timer, mna, 0, 5, timer.output_r,
                           timer.latched_high ? timer.high_v : timer.low_v);
        ne555_branch_stamp(timer, mna, 1, 1,
                           timer.latched_high ? timer.discharge_off_r : timer.discharge_on_r, 0.0);
        return true;
    }

    inline bool iterate_tr_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer,
                                  MNA::MNA& mna, [[maybe_unused]] double time) noexcept
    {
        if(!timer.valid()) { return false; }
        timer.tr_iteration_active = true;
        ne555_input_stamp(timer, mna);
        ne555_branch_stamp(timer, mna, 0, 5, timer.output_r,
                           timer.latched_high ? timer.high_v : timer.low_v);
        ne555_branch_stamp(timer, mna, 1, 1,
                           timer.latched_high ? timer.discharge_off_r : timer.discharge_on_r, 0.0);
        return true;
    }

    inline bool check_convergence_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer) noexcept
    {
        if(!timer.tr_iteration_active) { return true; }
        bool const desired{timer.desired_latch()};
        if(desired == timer.latched_high || !timer.tr_transition_available) { return true; }
        timer.latched_high = desired;
        timer.tr_transition_available = false;
        return false;
    }

    inline bool step_changed_tr_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer,
                                       [[maybe_unused]] double last_step,
                                       [[maybe_unused]] double new_step) noexcept
    {
        timer.tr_transition_available = true;
        return true;
    }

    inline bool iterate_trop_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer,
                                    MNA::MNA& mna) noexcept
    {
        return iterate_dc_define(model_reserve_type<ne555_timer>, timer, mna);
    }

    inline bool iterate_ac_define(model_reserve_type_t<ne555_timer>, ne555_timer const& timer,
                                  MNA::MNA& mna, [[maybe_unused]] double omega) noexcept
    {
        if(!timer.valid()) { return false; }
        ne555_input_stamp(timer, mna);
        ne555_branch_stamp(timer, mna, 0, 5, timer.output_r, 0.0);
        ne555_branch_stamp(timer, mna, 1, 1,
                           timer.latched_high ? timer.discharge_off_r : timer.discharge_on_r, 0.0);
        return true;
    }

    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer) noexcept
    {
        return {timer.pins, 8};
    }

    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<ne555_timer>, ne555_timer& timer) noexcept
    {
        return {timer.branches, 2};
    }
}
