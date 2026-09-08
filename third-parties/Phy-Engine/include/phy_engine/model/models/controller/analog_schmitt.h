#pragma once
#include <algorithm>
#include <cmath>
#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Analog, hysteretic voltage driver. Input current is zero and output
    // impedance is ideal. Slew is explicitly V/s; zero means instantaneous.
    // No claim is made that an external application's undocumented "slew"
    // property uses this unit. Hysteresis is retained only across solved steps,
    // not across Newton trial iterations.
    struct analog_schmitt
    {
        inline static constexpr fast_io::u8string_view model_name{u8"Analog Schmitt Trigger"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr fast_io::u8string_view identification_name{u8"ASCHMITT"};
        double Vth_low{1.6666666666666667}, Vth_high{3.3333333333333335};
        bool inverted{};
        double Ll{0}, Hl{5}, slew_v_per_s{};
        bool committed_state{}, trial_state{};
        double committed_output{}, trial_output{}, dt{};
        bool transient{}, initialized{};
        pin pins[3]{{{u8"in"}},{{u8"out"}},{{u8"ref"}}};
        branch branches{};
    };
    static_assert(model<analog_schmitt>);
    inline bool set_attribute_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s,std::size_t n,variant v) noexcept
    {
        if(n==2 && v.type==variant_type::boolean) { s.inverted=v.boolean; return true; }
        if(v.type!=variant_type::d || !std::isfinite(v.d)) return false;
        switch(n) { case 0:s.Vth_low=v.d;return true; case 1:s.Vth_high=v.d;return true;
                    case 2:if(v.d!=0 && v.d!=1) return false;s.inverted=v.d!=0;return true;
                    case 3:s.Ll=v.d;return true; case 4:s.Hl=v.d;return true;
                    case 5:if(v.d<0) return false;s.slew_v_per_s=v.d;return true; default:return false; }
    }
    inline variant get_attribute_define(model_reserve_type_t<analog_schmitt>,analog_schmitt const& s,std::size_t n) noexcept
    {
        double value{};
        switch(n) { case 0:value=s.Vth_low;break;case 1:value=s.Vth_high;break;case 2:value=s.inverted?1:0;break;
                    case 3:value=s.Ll;break;case 4:value=s.Hl;break;case 5:value=s.slew_v_per_s;break;
                    case 16:value=s.trial_state?1:0;break;case 17:value=s.trial_output;break;default:return {}; }
        return {.d{value},.type{variant_type::d}};
    }
    inline constexpr fast_io::u8string_view get_attribute_name_define(model_reserve_type_t<analog_schmitt>,std::size_t n) noexcept
    {
        switch(n) { case 0:return u8"Vth_low";case 1:return u8"Vth_high";case 2:return u8"inverted";
                    case 3:return u8"Ll";case 4:return u8"Hl";case 5:return u8"slew_V_per_s";
                    case 16:return u8"hysteresis_state";case 17:return u8"output_V";default:return {}; }
    }
    inline bool prepare_foundation_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s) noexcept
    {
        if(!std::isfinite(s.Vth_low) || !std::isfinite(s.Vth_high) || s.Vth_low>s.Vth_high
            || !std::isfinite(s.Ll) || !std::isfinite(s.Hl) || !std::isfinite(s.slew_v_per_s) || s.slew_v_per_s<0) return false;
        s.committed_state=s.trial_state=false;
        s.committed_output=s.trial_output=s.inverted?s.Hl:s.Ll;
        s.dt=0;s.transient=false;s.initialized=true;
        return true;
    }
    inline bool prepare_tr_define(model_reserve_type_t<analog_schmitt> tag,analog_schmitt& s) noexcept
    {
        if(!s.initialized && !prepare_foundation_define(tag,s)) return false;
        s.transient=true;return true;
    }
    inline bool step_changed_tr_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s,[[maybe_unused]] double old_dt,double dt) noexcept
    {
        if(!std::isfinite(dt) || dt<=0) return false;
        s.committed_state=s.trial_state;s.committed_output=s.trial_output;s.dt=dt;
        return true;
    }
    inline bool analog_schmitt_decide(analog_schmitt const& s,double input) noexcept
    {
        if(input>=s.Vth_high) return true;
        if(input<=s.Vth_low) return false;
        return s.committed_state;
    }
    inline double analog_schmitt_target(analog_schmitt const& s,bool state) noexcept
    {
        double const target=(state!=s.inverted)?s.Hl:s.Ll;
        if(!s.transient || s.slew_v_per_s==0 || s.Ll==s.Hl) return target;
        double const step=s.slew_v_per_s*s.dt;
        return std::clamp(target,s.committed_output-step,s.committed_output+step);
    }
    inline bool analog_schmitt_stamp(analog_schmitt const& s,MNA::MNA& mna,double value) noexcept
    {
        if(!s.pins[1].nodes || !s.pins[2].nodes || !std::isfinite(value)) return false;
        auto o=s.pins[1].nodes->node_index,r=s.pins[2].nodes->node_index,k=s.branches.index;
        mna.B_ref(o,k)+=1;mna.B_ref(r,k)-=1;mna.C_ref(k,o)+=1;mna.C_ref(k,r)-=1;mna.E_ref(k)+=value;
        return true;
    }
    inline bool iterate_dc_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s,MNA::MNA& mna) noexcept
    {
        if(!s.pins[0].nodes || !s.pins[2].nodes) return false;
        double const input=s.pins[0].nodes->node_information.an.voltage.real()-s.pins[2].nodes->node_information.an.voltage.real();
        if(!std::isfinite(input)) return false;
        s.trial_state=analog_schmitt_decide(s,input);
        s.trial_output=analog_schmitt_target(s,s.trial_state);
        return analog_schmitt_stamp(s,mna,s.trial_output);
    }
    inline bool iterate_ac_define(model_reserve_type_t<analog_schmitt>,analog_schmitt const& s,MNA::MNA& mna,[[maybe_unused]] double omega) noexcept
    { return analog_schmitt_stamp(s,mna,0); }
    inline bool check_convergence_define(model_reserve_type_t<analog_schmitt>,analog_schmitt const& s) noexcept
    {
        for(auto const& p:s.pins) if(!p.nodes) return false;
        double const input=s.pins[0].nodes->node_information.an.voltage.real()-s.pins[2].nodes->node_information.an.voltage.real();
        double const output=s.pins[1].nodes->node_information.an.voltage.real()-s.pins[2].nodes->node_information.an.voltage.real();
        double const expected=analog_schmitt_target(s,analog_schmitt_decide(s,input));
        return std::isfinite(input) && std::isfinite(output) && std::abs(output-expected)<=1e-8+1e-6*std::abs(expected);
    }
    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s) noexcept { return {s.pins,3}; }
    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<analog_schmitt>,analog_schmitt& s) noexcept { return {&s.branches,1}; }
}
