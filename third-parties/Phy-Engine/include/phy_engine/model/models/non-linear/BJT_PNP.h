#pragma once
#include <fast_io/fast_io_dsal/string_view.h>
#include "bjt_ebers_moll.h"

namespace phy_engine::model
{
    struct BJT_PNP : details::bjt_ebers_moll_state
    {
        inline static constexpr fast_io::u8string_view model_name{u8"PNP BJT"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr fast_io::u8string_view identification_name{u8"QPN"};
        inline static constexpr double polarity{-1};
        pin pins[3]{{{u8"B"}},{{u8"C"}},{{u8"E"}}};
    };
    static_assert(model<BJT_PNP>);
    inline bool set_attribute_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q,std::size_t n,variant value) noexcept
    { return details::bjt_set(q,n,value); }
    inline variant get_attribute_define(model_reserve_type_t<BJT_PNP>,BJT_PNP const& q,std::size_t n) noexcept
    { return details::bjt_get(q,n); }
    inline constexpr fast_io::u8string_view get_attribute_name_define(model_reserve_type_t<BJT_PNP>,std::size_t n) noexcept
    { return details::bjt_attribute_name(n); }
    inline bool prepare_foundation_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q) noexcept
    { return details::bjt_prepare(q); }
    inline bool iterate_dc_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q,MNA::MNA& mna) noexcept
    { return details::bjt_stamp(q,mna); }
    inline bool iterate_tr_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q,MNA::MNA& mna,[[maybe_unused]] double time) noexcept
    { return details::bjt_stamp(q,mna); }
    inline bool iterate_ac_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q,MNA::MNA& mna,[[maybe_unused]] double omega) noexcept
    { return details::bjt_ac(q,mna); }
    inline bool check_convergence_define(model_reserve_type_t<BJT_PNP>,BJT_PNP const& q) noexcept
    { return details::bjt_converged(q); }
    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<BJT_PNP>,BJT_PNP& q) noexcept
    { return {q.pins,3}; }
}
