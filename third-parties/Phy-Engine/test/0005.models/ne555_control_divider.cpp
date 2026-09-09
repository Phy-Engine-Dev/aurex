#include <cmath>

#include <fast_io/fast_io.h>

#include <phy_engine/circuits/circuit.h>
#include <phy_engine/model/models/controller/ne555_timer.h>
#include <phy_engine/model/models/linear/VDC.h>
#include <phy_engine/model/models/linear/capacitor.h>
#include <phy_engine/model/models/linear/resistance.h>
#include <phy_engine/netlist/impl.h>

namespace
{
    enum class control_kind { unwired, bypass_capacitor, driven };

    struct observation
    {
        bool solved{};
        bool latched_high{};
        double control_v{};
        double output_v{};
        double trigger_threshold{};
        double threshold_threshold{};
    };

    observation solve(control_kind kind, double control_v, double trigger_v, double threshold_v)
    {
        ::phy_engine::circult circuit{};
        circuit.set_analyze_type(::phy_engine::analyze_type::DC);
        auto& netlist{circuit.get_netlist()};

        auto [supply, supply_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = 5.0})};
        auto [trigger, trigger_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = trigger_v})};
        auto [threshold, threshold_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = threshold_v})};
        auto [load, load_pos]{add_model(netlist, ::phy_engine::model::resistance{.r = 1000.0})};

        ::phy_engine::model::ne555_timer configured{};
        configured.internal_control = kind == control_kind::unwired;
        configured.internal_reset_pullup = false;
        auto [timer, timer_pos]{add_model(netlist, ::std::move(configured))};

        auto& vcc{create_node(netlist)};
        auto& discharge{create_node(netlist)};
        auto& threshold_node{create_node(netlist)};
        auto& control{create_node(netlist)};
        auto& trigger_node{create_node(netlist)};
        auto& output{create_node(netlist)};
        auto& ground{netlist.ground_node};

        add_to_node(netlist, *supply, 0, vcc);
        add_to_node(netlist, *supply, 1, ground);
        add_to_node(netlist, *trigger, 0, trigger_node);
        add_to_node(netlist, *trigger, 1, ground);
        add_to_node(netlist, *threshold, 0, threshold_node);
        add_to_node(netlist, *threshold, 1, ground);
        add_to_node(netlist, *load, 0, output);
        add_to_node(netlist, *load, 1, ground);

        add_to_node(netlist, *timer, 0, vcc);
        add_to_node(netlist, *timer, 1, discharge);
        add_to_node(netlist, *timer, 2, threshold_node);
        add_to_node(netlist, *timer, 3, control);
        add_to_node(netlist, *timer, 4, trigger_node);
        add_to_node(netlist, *timer, 5, output);
        add_to_node(netlist, *timer, 6, vcc);
        add_to_node(netlist, *timer, 7, ground);

        if(kind == control_kind::bypass_capacitor)
        {
            auto [bypass, bypass_pos]{add_model(netlist,
                ::phy_engine::model::capacitor{.m_kZimag = 10e-9})};
            add_to_node(netlist, *bypass, 0, control);
            add_to_node(netlist, *bypass, 1, ground);
        }
        else if(kind == control_kind::driven)
        {
            auto [driver, driver_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = control_v})};
            add_to_node(netlist, *driver, 0, control);
            add_to_node(netlist, *driver, 1, ground);
        }

        observation result{};
        result.solved = circuit.analyze();
        if(!result.solved) { return result; }
        result.latched_high = timer->ptr->get_attribute(7).boolean;
        result.trigger_threshold = timer->ptr->get_attribute(8).d;
        result.threshold_threshold = timer->ptr->get_attribute(9).d;
        result.control_v = control.node_information.an.voltage.real();
        result.output_v = output.node_information.an.voltage.real();
        return result;
    }

    bool near(double actual, double expected, double tolerance = 1e-6) noexcept
    {
        return ::std::isfinite(actual) && ::std::abs(actual - expected) <= tolerance;
    }
}  // namespace

int main()
{
    // An unused CTRL pin still exposes the physical 5k-5k-5k divider node.
    auto const unwired{solve(control_kind::unwired, 0.0, 0.5, 1.0)};
    if(!unwired.solved || !unwired.latched_high ||
       !near(unwired.control_v, 10.0 / 3.0) || !near(unwired.output_v, 3.0, 1e-5) ||
       !near(unwired.trigger_threshold, 5.0 / 3.0) || !near(unwired.threshold_threshold, 10.0 / 3.0))
    {
        ::fast_io::io::perr("ne555_control_divider: unwired CTRL mismatch\n");
        return 1;
    }

    // A common CTRL bypass capacitor is open at DC and must settle to 2/3 VCC;
    // merely connecting it must not remove the internal divider or lock OUT low.
    auto const bypassed{solve(control_kind::bypass_capacitor, 0.0, 0.5, 1.0)};
    if(!bypassed.solved || !bypassed.latched_high ||
       !near(bypassed.control_v, 10.0 / 3.0) || !near(bypassed.output_v, 3.0, 1e-5) ||
       !near(bypassed.trigger_threshold, 5.0 / 3.0) || !near(bypassed.threshold_threshold, 10.0 / 3.0))
    {
        ::fast_io::io::perr("ne555_control_divider: bypassed CTRL mismatch\n");
        return 2;
    }

    // A low-impedance external CTRL source remains authoritative despite the
    // retained divider, and both comparator thresholds follow it.
    auto const driven_high{solve(control_kind::driven, 2.0, 0.5, 1.5)};
    auto const driven_low{solve(control_kind::driven, 2.0, 1.5, 2.5)};
    if(!driven_high.solved || !driven_high.latched_high || !near(driven_high.control_v, 2.0) ||
       !near(driven_high.trigger_threshold, 1.0) || !near(driven_high.threshold_threshold, 2.0) ||
       !driven_low.solved || driven_low.latched_high || !near(driven_low.control_v, 2.0) ||
       !near(driven_low.trigger_threshold, 1.0) || !near(driven_low.threshold_threshold, 2.0))
    {
        ::fast_io::io::perr("ne555_control_divider: driven CTRL mismatch\n");
        return 3;
    }

    return 0;
}
