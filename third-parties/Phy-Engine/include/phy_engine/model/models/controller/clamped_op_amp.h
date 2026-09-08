#pragma once
#include <algorithm>
#include <cmath>
#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Memoryless, finite-gain behavioral amplifier, not a powered silicon model.
    // V(out)-V(ref)=clamp(mu*(V(+)-V(-)), Vmin, Vmax).
    // Inputs draw zero current, output is ideal; no bandwidth, slew or current limit.
    // Legacy `op_amp` remains unchanged for callers requesting an unlimited amplifier.
    struct clamped_op_amp
    {
        inline static constexpr fast_io::u8string_view model_name{u8"Clamped OpAmp"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr fast_io::u8string_view identification_name{u8"OPAMP_CLAMP"};
        double mu{1e5}, Vmin{-15}, Vmax{15};
        double ac_gain{};
        pin pins[4]{{{u8"+"}}, {{u8"-"}}, {{u8"out"}}, {{u8"ref"}}};
        branch branches{};
    };
    static_assert(model<clamped_op_amp>);
    inline bool set_attribute_define(model_reserve_type_t<clamped_op_amp>, clamped_op_amp& a, std::size_t n, variant v) noexcept
    {
        if(v.type != variant_type::d || !std::isfinite(v.d)) return false;
        switch(n) { case 0: if(v.d < 0) return false; a.mu=v.d; return true;
                    case 1: a.Vmin=v.d; return true; case 2: a.Vmax=v.d; return true; default: return false; }
    }
    inline variant get_attribute_define(model_reserve_type_t<clamped_op_amp>, clamped_op_amp const& a, std::size_t n) noexcept
    {
        switch(n) { case 0:return {.d{a.mu},.type{variant_type::d}}; case 1:return {.d{a.Vmin},.type{variant_type::d}};
                    case 2:return {.d{a.Vmax},.type{variant_type::d}}; default:return {}; }
    }
    inline constexpr fast_io::u8string_view get_attribute_name_define(model_reserve_type_t<clamped_op_amp>,std::size_t n) noexcept
    {
        switch(n) { case 0:return u8"mu"; case 1:return u8"Vmin"; case 2:return u8"Vmax"; default:return {}; }
    }
    inline bool prepare_foundation_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp const& a) noexcept
    {
        return std::isfinite(a.mu) && a.mu>=0 && std::isfinite(a.Vmin) && std::isfinite(a.Vmax) && a.Vmin<=a.Vmax;
    }
    inline bool clamped_op_amp_stamp(clamped_op_amp const& a,MNA::MNA& mna,double gain,double rhs) noexcept
    {
        auto p=a.pins[0].nodes, n=a.pins[1].nodes, o=a.pins[2].nodes, r=a.pins[3].nodes;
        if(!p || !n || !o || !r) return false; // implicit ground is an import policy, not hidden here
        auto k=a.branches.index;
        mna.B_ref(o->node_index,k)+=1; mna.B_ref(r->node_index,k)-=1;
        mna.C_ref(k,o->node_index)+=1; mna.C_ref(k,r->node_index)-=1;
        mna.C_ref(k,p->node_index)-=gain; mna.C_ref(k,n->node_index)+=gain;
        mna.E_ref(k)+=rhs;
        return true;
    }
    inline bool iterate_dc_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp& a,MNA::MNA& mna) noexcept
    {
        if(!a.pins[0].nodes || !a.pins[1].nodes) return false;
        double const input=a.pins[0].nodes->node_information.an.voltage.real()-a.pins[1].nodes->node_information.an.voltage.real();
        double const raw=a.mu*input;
        if(!std::isfinite(raw)) return false;
        double const output=std::clamp(raw,a.Vmin,a.Vmax);
        a.ac_gain=(raw>=a.Vmin && raw<=a.Vmax && a.Vmin!=a.Vmax)?a.mu:0;
        return clamped_op_amp_stamp(a,mna,a.ac_gain,output-a.ac_gain*input);
    }
    inline bool iterate_ac_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp const& a,MNA::MNA& mna,[[maybe_unused]] double omega) noexcept
    { return clamped_op_amp_stamp(a,mna,a.ac_gain,0); }
    inline bool check_convergence_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp const& a) noexcept
    {
        for(auto const& p:a.pins) if(!p.nodes) return false;
        double const raw=a.mu*(a.pins[0].nodes->node_information.an.voltage.real()-a.pins[1].nodes->node_information.an.voltage.real());
        double const output=a.pins[2].nodes->node_information.an.voltage.real()-a.pins[3].nodes->node_information.an.voltage.real();
        double const expected=std::clamp(raw,a.Vmin,a.Vmax);
        return std::isfinite(raw) && std::isfinite(output) && std::abs(output-expected)<=1e-8+1e-6*std::abs(expected);
    }
    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp& a) noexcept { return {a.pins,4}; }
    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<clamped_op_amp>,clamped_op_amp& a) noexcept { return {&a.branches,1}; }
}
