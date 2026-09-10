#pragma once
#include <cstddef>
#include <vector>
#include <fast_io/fast_io_dsal/string_view.h>
#include "../../model/pin/pin.h"

namespace phy_engine::digital
{
    enum class settle_reason : unsigned char { settled, event_budget, multiple_drivers };
    struct node_activity { model::node_t* node{}; std::size_t events{}; };
    struct settle_status
    {
        bool settled{};
        settle_reason reason{settle_reason::settled};
        double time_s{};
        std::size_t processed_events{}, pending_nodes{}, multiple_driver_nodes{}, undriven_nodes{}, undriven_clock_nodes{};
        std::vector<model::node_t*> pending{}, conflicts{}, undriven{};
        std::vector<node_activity> hot{};
    };

    // Native primitive interface contracts, not pin-name guesses. Directly
    // tied digital outputs have no last-writer-wins semantics: callers must
    // use distinct driver nets and RESOLVE2 for explicit four-state resolution.
    // -1 is an unknown interface, 0 a load, 1 a driver.
    inline constexpr int pin_role(fast_io::u8string_view name, std::size_t pin) noexcept
    {
        if(name==u8"INPUT" || name==u8"EIGHT_BIT_INPUT") return 1;
        if(name==u8"OUTPUT" || name==u8"EIGHT_BIT_DISPLAY" || name==u8"VERILOG_PORTS") return 0;
        if(name==u8"NOT" || name==u8"YES" || name==u8"IS_UNKNOWN" || name==u8"SCHMITT_TRIGGER" || name==u8"TICK_DELAY") return pin==1;
        if(name==u8"AND" || name==u8"OR" || name==u8"XOR" || name==u8"NAND" || name==u8"NOR" || name==u8"XNOR" || name==u8"IMP" || name==u8"NIMP" || name==u8"CASE_EQ" || name==u8"RESOLVE2" || name==u8"TRI") return pin==2;
        if(name==u8"DFF" || name==u8"DLATCH" || name==u8"TFF" || name==u8"T_BAR_FF") return pin==2;
        if(name==u8"DFF_ARSTN" || name==u8"JKFF") return pin==3;
        if(name==u8"HALF_ADDER" || name==u8"HALF_SUB") return pin>=2;
        if(name==u8"FULL_ADDER" || name==u8"FULL_SUB") return pin>=3;
        if(name==u8"MUL2") return pin>=4;
        if(name==u8"COUNTER4" || name==u8"RANDOM_GENERATOR4") return pin<4;
        return -1;
    }
}
