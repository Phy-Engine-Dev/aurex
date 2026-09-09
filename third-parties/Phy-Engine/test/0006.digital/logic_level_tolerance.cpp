#include <fast_io/fast_io.h>

#include <phy_engine/model/model_refs/logic_level.h>
#include <phy_engine/phy_engine.h>

namespace
{
    using dns = ::phy_engine::model::digital_node_statement_t;

    dns observe(double voltage)
    {
        ::phy_engine::circult circuit{};
        circuit.set_analyze_type(::phy_engine::analyze_type::DC);
        auto& netlist{circuit.get_netlist()};

        auto [source, source_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = voltage})};
        auto [sink, sink_pos]{add_model(netlist, ::phy_engine::model::OUTPUT{.Ll = 0.0, .Hl = 3.0, .Tsu = 0.0, .Th = 0.0})};
        auto& signal{create_node(netlist)};
        add_to_node(netlist, *source, 0, signal);
        add_to_node(netlist, *sink, 0, signal);
        add_to_node(netlist, *source, 1, netlist.ground_node);

        if(!circuit.analyze()) { return dns::X; }
        circuit.digital_clk();
        auto const value{sink->ptr->get_attribute(0)};
        return value.type == ::phy_engine::model::variant_type::digital ? value.digital : dns::X;
    }

    bool counter_accepts_near_high_rail()
    {
        ::phy_engine::circult circuit{};
        circuit.set_analyze_type(::phy_engine::analyze_type::DC);
        auto& netlist{circuit.get_netlist()};

        auto [source, source_pos]{add_model(netlist, ::phy_engine::model::VDC{.V = 2.9999997})};
        auto [counter, counter_pos]{add_model(netlist, ::phy_engine::model::COUNTER4{.Ll = 0.0, .Hl = 3.0})};
        auto& clock{create_node(netlist)};
        add_to_node(netlist, *source, 0, clock);
        add_to_node(netlist, *counter, 4, clock);
        add_to_node(netlist, *source, 1, netlist.ground_node);

        if(!circuit.analyze()) { return false; }
        circuit.digital_clk();
        auto const value{counter->ptr->get_attribute(0)};
        return value.type == ::phy_engine::model::variant_type::ui8 && value.ui8 == 1;
    }
}  // namespace

int main()
{
    using ::phy_engine::model::logic_level::is_high;
    using ::phy_engine::model::logic_level::is_low;

    // The real failing rail from the imported 555 pulse counter.
    if(!is_high(2.9999997, 0.0, 3.0) || observe(2.9999997) != dns::H || !counter_accepts_near_high_rail())
    {
        ::fast_io::io::perr("logic_level_tolerance: near-high rail was not recognized\n");
        return 1;
    }

    if(!is_low(0.0000003, 0.0, 3.0) || observe(0.0000003) != dns::L)
    {
        ::fast_io::io::perr("logic_level_tolerance: near-low rail was not recognized\n");
        return 2;
    }

    // Ten microvolts below a 3 V high rail is outside the 1 ppm tolerance;
    // ordinary mid-band values must remain indeterminate.
    if(is_high(2.99999, 0.0, 3.0) || is_low(0.00001, 0.0, 3.0) || observe(2.99999) != dns::X || observe(1.5) != dns::X)
    {
        ::fast_io::io::perr("logic_level_tolerance: a non-rail voltage was classified as digital\n");
        return 3;
    }

    // A large common-mode offset must not erase the configured 3 V band.
    if(!is_high(-999999997.0000004, -1000000000.0, -999999997.0) ||
       is_high(-999999998.5, -1000000000.0, -999999997.0))
    {
        ::fast_io::io::perr("logic_level_tolerance: offset-rail scaling is unsafe\n");
        return 4;
    }

    return 0;
}
