#pragma once
#include <set>

#include "../../model/node/node.h"

namespace phy_engine::digital
{
    struct digital_node_update_state
    {
        ::phy_engine::model::node_t* node{};
        ::phy_engine::model::digital_node_statement_t update_statement{};
    };

    struct digital_node_update_table
    {
        // Nodes that must be processed every digital tick (e.g. hybrid analog/digital nodes).
        ::std::set<::phy_engine::model::node_t*> always_tables{};

        // Pending nodes scheduled for update in the current tick.
        ::std::set<::phy_engine::model::node_t*> tables{};
    };

    struct need_operate_analog_node_t
    {
        double voltage{};
        ::phy_engine::model::node_t* need_to_operate_analog_node{};
        // Stable identity of the emitting model wrapper. A combinational
        // model may be revisited more than once during one settled tick; its
        // last value replaces its own earlier value, while two distinct
        // drivers on the same analog node remain a real contention error.
        void const* driver{};
    };
}  // namespace phy_engine::digital
