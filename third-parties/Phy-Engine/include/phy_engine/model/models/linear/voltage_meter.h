#pragma once
#include <cmath>
#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Finite-input-resistance voltage probe. This is not an ideal 0 V source:
    // it loads the measured circuit by 1/Rinput, and exposes fresh signed terminal
    // currents. Meter dial/mode conversion belongs to the file-format adapter.
    struct voltage_meter
    {
        inline static constexpr fast_io::u8string_view model_name{u8"Voltage Meter"};
        inline static constexpr model_device_type device_type{model_device_type::linear};
        inline static constexpr fast_io::u8string_view identification_name{u8"VMETER"};
        double Rinput{1e9};
        pin pins[2]{{{u8"+"}},{{u8"-"}}};
    };
    static_assert(model<voltage_meter>);
    inline bool set_attribute_define(model_reserve_type_t<voltage_meter>,voltage_meter& m,std::size_t n,variant v) noexcept
    {
        if(n!=0 || v.type!=variant_type::d || !std::isfinite(v.d) || v.d<=0 || !std::isfinite(1/v.d)) return false;
        m.Rinput=v.d; return true;
    }
    inline variant get_attribute_define(model_reserve_type_t<voltage_meter>,voltage_meter const& m,std::size_t n) noexcept
    {
        if(n==0) return {.d{m.Rinput},.type{variant_type::d}};
        if(n<16 || n>19 || !m.pins[0].nodes || !m.pins[1].nodes) return {};
        auto i=(m.pins[0].nodes->node_information.an.voltage-m.pins[1].nodes->node_information.an.voltage)/m.Rinput;
        double value=n<18?i.real():i.imag();
        if(n%2) value=-value;
        return {.d{value},.type{variant_type::d}};
    }
    inline constexpr fast_io::u8string_view get_attribute_name_define(model_reserve_type_t<voltage_meter>,std::size_t n) noexcept
    {
        switch(n) { case 0:return u8"Rinput"; case 16:return u8"Iplus_real"; case 17:return u8"Iminus_real";
                    case 18:return u8"Iplus_imag"; case 19:return u8"Iminus_imag"; default:return {}; }
    }
    inline bool prepare_foundation_define(model_reserve_type_t<voltage_meter>,voltage_meter const& m) noexcept
    { return std::isfinite(m.Rinput) && m.Rinput>0 && std::isfinite(1/m.Rinput); }
    inline bool iterate_dc_define(model_reserve_type_t<voltage_meter>,voltage_meter const& m,MNA::MNA& mna) noexcept
    {
        if(!m.pins[0].nodes || !m.pins[1].nodes) return false;
        auto p=m.pins[0].nodes->node_index,n=m.pins[1].nodes->node_index;
        double const g=1/m.Rinput;
        mna.G_ref(p,p)+=g; mna.G_ref(n,n)+=g; mna.G_ref(p,n)-=g; mna.G_ref(n,p)-=g;
        return true;
    }
    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<voltage_meter>,voltage_meter& m) noexcept { return {m.pins,2}; }
}
