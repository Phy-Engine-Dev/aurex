#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include <fast_io/fast_io.h>

#include <phy_engine/circuits/circuit.h>
#include <phy_engine/model/models/controller/ne555_timer.h>
#include <phy_engine/model/models/linear/VDC.h>
#include <phy_engine/model/models/linear/capacitor.h>
#include <phy_engine/model/models/linear/resistance.h>
#include <phy_engine/model/models/non-linear/PN_junction.h>
#include <phy_engine/netlist/impl.h>

namespace
{
    struct observation
    {
        bool solved{};
        ::std::size_t completed_steps{};
        ::std::size_t edges{};
        double first_fall_time{};
        double first_rise_time{};
        double before_fall_v{};
        double after_fall_v{};
        double after_rise_v{};
        double minimum_timing_v{1e100};
        double maximum_timing_v{-1e100};
    };

    observation simulate(double dt, double duration)
    {
        ::phy_engine::circult circuit{};
        circuit.set_analyze_type(::phy_engine::analyze_type::TR);
        auto& setting{circuit.get_analyze_setting()};
        setting.tr.t_step = dt;
        setting.tr.t_stop = dt;
        auto& netlist{circuit.get_netlist()};

        auto [supply, supply_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = 15.25})};

        ::phy_engine::model::ne555_timer configured{};
        configured.internal_control = true;
        configured.internal_reset_pullup = false;
        configured.high_v = 3.0;
        configured.output_r = 1e-3;
        configured.discharge_on_r = 1e-3;
        configured.discharge_off_r = 1e12;
        auto [timer, timer_pos]{add_model(netlist, ::std::move(configured))};

        auto [charge, charge_pos]{add_model(netlist, ::phy_engine::model::resistance{.r = 1000.0})};
        auto [discharge, discharge_pos]{add_model(netlist, ::phy_engine::model::resistance{.r = 10000.0})};
        auto [fast_lead, fast_lead_pos]{add_model(netlist, ::phy_engine::model::resistance{.r = 1e-5})};
        auto [esr, esr_pos]{add_model(netlist, ::phy_engine::model::resistance{.r = 5.0})};
        auto [timing_cap, timing_cap_pos]{
            add_model(netlist, ::phy_engine::model::capacitor{.m_kZimag = 100e-9})};

        ::phy_engine::model::PN_junction diode_model{};
        diode_model.Is = 9.177923434724038e-6;
        diode_model.N = 2.0;
        diode_model.Bv_set = false;
        auto [charge_diode, charge_diode_pos]{add_model(netlist, diode_model)};
        auto [discharge_diode, discharge_diode_pos]{add_model(netlist, diode_model)};

        auto& vcc{create_node(netlist)};
        auto& discharge_pin{create_node(netlist)};
        auto& timing{create_node(netlist)};
        auto& control{create_node(netlist)};
        auto& output{create_node(netlist)};
        auto& discharge_path{create_node(netlist)};
        auto& charge_path{create_node(netlist)};
        auto& capacitor_internal{create_node(netlist)};
        auto& ground{netlist.ground_node};

        add_to_node(netlist, *supply, 0, vcc);
        add_to_node(netlist, *supply, 1, ground);
        add_to_node(netlist, *timer, 0, vcc);
        add_to_node(netlist, *timer, 1, discharge_pin);
        add_to_node(netlist, *timer, 2, timing);
        add_to_node(netlist, *timer, 3, control);
        add_to_node(netlist, *timer, 4, timing);
        add_to_node(netlist, *timer, 5, output);
        add_to_node(netlist, *timer, 6, vcc);
        add_to_node(netlist, *timer, 7, ground);
        add_to_node(netlist, *charge, 0, vcc);
        add_to_node(netlist, *charge, 1, discharge_pin);
        add_to_node(netlist, *fast_lead, 0, timing);
        add_to_node(netlist, *fast_lead, 1, charge_path);
        add_to_node(netlist, *charge_diode, 0, charge_path);
        add_to_node(netlist, *charge_diode, 1, discharge_pin);
        add_to_node(netlist, *discharge_diode, 0, discharge_pin);
        add_to_node(netlist, *discharge_diode, 1, discharge_path);
        add_to_node(netlist, *discharge, 0, discharge_path);
        add_to_node(netlist, *discharge, 1, timing);
        add_to_node(netlist, *esr, 0, timing);
        add_to_node(netlist, *esr, 1, capacitor_internal);
        add_to_node(netlist, *timing_cap, 0, capacitor_internal);
        add_to_node(netlist, *timing_cap, 1, ground);

        observation result{};
        bool previous_latch{timer->ptr->get_attribute(7).boolean};
        double previous_timing_v{};
        auto const steps{static_cast<::std::size_t>(::std::llround(duration / dt))};
        for(::std::size_t step{1}; step <= steps; ++step)
        {
            if(!circuit.analyze()) { return result; }
            result.completed_steps = step;

            double const timing_v{timing.node_information.an.voltage.real()};
            double const output_v{output.node_information.an.voltage.real()};
            if(!::std::isfinite(timing_v) || !::std::isfinite(output_v)) { return result; }
            result.minimum_timing_v = ::std::min(result.minimum_timing_v, timing_v);
            result.maximum_timing_v = ::std::max(result.maximum_timing_v, timing_v);

            bool const latch{timer->ptr->get_attribute(7).boolean};
            double const expected_output{latch ? 3.0 : 0.0};
            if(::std::abs(output_v - expected_output) > 1e-5) { return result; }

            if(latch != previous_latch)
            {
                ++result.edges;
                if(previous_latch && !latch && result.first_fall_time == 0.0)
                {
                    result.first_fall_time = circuit.tr_duration;
                    result.before_fall_v = previous_timing_v;
                    result.after_fall_v = timing_v;
                }
                else if(!previous_latch && latch && result.first_fall_time != 0.0 &&
                        result.first_rise_time == 0.0)
                {
                    result.first_rise_time = circuit.tr_duration;
                    result.after_rise_v = timing_v;
                }
                previous_latch = latch;
            }
            previous_timing_v = timing_v;
        }
        result.solved = true;
        return result;
    }

    bool near(double actual, double expected, double tolerance) noexcept
    {
        return ::std::isfinite(actual) && ::std::abs(actual - expected) <= tolerance;
    }
}

int main()
{
    // Reduced from the real imported Buck: two opposed PN steering paths,
    // a 10 micro-ohm lead, and a 100 nF timing capacitor with 5 ohm ESR.
    // The old model oscillated between topologies at the first commutation.
    constexpr double duration{2.5e-3};
    constexpr double steps[]{2e-6, 1e-6, 0.5e-6};
    for(double const dt: steps)
    {
        auto const result{simulate(dt, duration)};
        auto const expected_steps{static_cast<::std::size_t>(::std::llround(duration / dt))};
        bool const valid{
            result.solved && result.completed_steps == expected_steps &&
            result.edges >= 3 &&
            near(result.first_fall_time, 1.229e-3, 2.5 * dt) &&
            near(result.first_rise_time - result.first_fall_time, dt, 1e-12) &&
            result.before_fall_v > 10.0 && result.before_fall_v < 10.3 &&
            result.after_fall_v > 0.45 && result.after_fall_v < 0.75 &&
            result.minimum_timing_v > -3.0 &&
            result.maximum_timing_v > 10.0 && result.maximum_timing_v < 15.25};
        if(!valid)
        {
            ::fast_io::io::perr(
                "ne555_diode_steering_tr: dt=", dt,
                " solved=", result.solved,
                " steps=", result.completed_steps,
                " edges=", result.edges,
                " first_fall=", result.first_fall_time,
                " first_rise=", result.first_rise_time,
                " before/after fall=", result.before_fall_v, "/", result.after_fall_v,
                " after_rise=", result.after_rise_v,
                " range=[", result.minimum_timing_v, ",", result.maximum_timing_v, "]\n");
            return 1;
        }
    }
    return 0;
}
