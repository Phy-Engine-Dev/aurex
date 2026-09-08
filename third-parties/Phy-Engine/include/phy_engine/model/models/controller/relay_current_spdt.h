#pragma once
#include <cmath>
#include <complex>
#include <fast_io/fast_io_dsal/string_view.h>
#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Generic engineering relay, not a reverse-engineered PhysicsLab solver.
    // Five external pins in PL order: contact0, common1, contact2, coil+3, coil-4.
    // Public convention: contact0=NC and contact2=NO. Coil is a real series R-L
    // branch. Magnitude of solved coil current drives Schmitt pickup/dropout.
    // Contacts are two explicit finite resistances; no bounce or arc model.
    // Transient coil integration is backward Euler (stable for tiny coil L).
    // Mechanical delays are quantized at actual transient solve time points.
    struct relay_current_spdt
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"Current Operated SPDT Relay"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"RELAYI"};

        double L{0.2};
        double R{20.0};
        double Ipull{0.02};
        double Idrop{0.016};
        double Ron{0.01};
        double Roff{1e12};
        double operate_delay{};
        double release_delay{};
        bool initial_engaged{};
        bool engaged{};

        pin pins[5]{{{u8"NC"}}, {{u8"COM"}}, {{u8"NO"}}, {{u8"COIL+"}}, {{u8"COIL-"}}};
        // Positive branch current: coil+->coil-, COM->NC, COM->NO.
        branch branches[3]{};

        double tr_req{};
        double tr_history{};
        bool in_transient{};
        bool check_dc_state{};
        bool step_engaged{};
        int pending{-1};
        int step_pending{-1};
        double pending_since{};
        double step_pending_since{};
        double trial_time{};

        [[nodiscard]] bool valid() const noexcept
        {
            return ::std::isfinite(L) && L >= 0.0 && ::std::isfinite(R) && R > 0.0 && ::std::isfinite(Ipull) && Ipull > 0.0 && ::std::isfinite(Idrop) &&
                   Idrop >= 0.0 && Idrop < Ipull && ::std::isfinite(Ron) && Ron > 0.0 && ::std::isfinite(Roff) && Roff > Ron &&
                   ::std::isfinite(operate_delay) && operate_delay >= 0.0 && ::std::isfinite(release_delay) && release_delay >= 0.0;
        }

        [[nodiscard]] bool requested_state(bool state) const noexcept
        {
            double const current{::std::abs(branches[0].current.real())};
            return state ? current > Idrop : current >= Ipull;
        }

        struct transition
        {
            bool state;
            int pending;
            double since;
        };

        [[nodiscard]] transition transition_at(double time) const noexcept
        {
            bool const desired{requested_state(step_engaged)};
            if(desired == step_engaged) { return {step_engaged, -1, 0.0}; }
            int const target{desired ? 1 : 0};
            double const since{step_pending == target ? step_pending_since : time};
            double const delay{desired ? operate_delay : release_delay};
            if(time - since >= delay) { return {desired, -1, 0.0}; }
            return {step_engaged, target, since};
        }
    };

    static_assert(model<relay_current_spdt>);

    inline bool set_attribute_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r, ::std::size_t index, variant value) noexcept
    {
        if(value.type != variant_type::d || !::std::isfinite(value.d)) { return false; }
        // Validate each field without requiring property-assignment order.
        switch(index)
        {
            case 0:
                if(value.d < 0.0) { return false; }
                r.L = value.d;
                return true;
            case 1:
                if(value.d <= 0.0) { return false; }
                r.R = value.d;
                return true;
            case 2:
                if(value.d <= 0.0) { return false; }
                r.Ipull = value.d;
                return true;
            case 3:
                if(value.d < 0.0) { return false; }
                r.Idrop = value.d;
                return true;
            case 4:
                if(value.d <= 0.0) { return false; }
                r.Ron = value.d;
                return true;
            case 5:
                if(value.d <= 0.0) { return false; }
                r.Roff = value.d;
                return true;
            case 6:
                if(value.d < 0.0) { return false; }
                r.operate_delay = value.d;
                return true;
            case 7:
                if(value.d < 0.0) { return false; }
                r.release_delay = value.d;
                return true;
            case 8:
                if(value.d != 0.0 && value.d != 1.0) { return false; }
                r.initial_engaged = value.d == 1.0;
                return true;
            default: return false;
        }
    }

    inline variant get_attribute_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt const& r, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return {.d{r.L}, .type{variant_type::d}};
            case 1: return {.d{r.R}, .type{variant_type::d}};
            case 2: return {.d{r.Ipull}, .type{variant_type::d}};
            case 3: return {.d{r.Idrop}, .type{variant_type::d}};
            case 4: return {.d{r.Ron}, .type{variant_type::d}};
            case 5: return {.d{r.Roff}, .type{variant_type::d}};
            case 6: return {.d{r.operate_delay}, .type{variant_type::d}};
            case 7: return {.d{r.release_delay}, .type{variant_type::d}};
            case 8: return {.d{r.initial_engaged ? 1.0 : 0.0}, .type{variant_type::d}};
            case 9: return {.d{r.engaged ? 1.0 : 0.0}, .type{variant_type::d}};
            case 10: return {.d{r.branches[0].current.real()}, .type{variant_type::d}};
            default: return {};
        }
    }

    inline constexpr ::fast_io::u8string_view get_attribute_name_define(model_reserve_type_t<relay_current_spdt>, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return u8"L";
            case 1: return u8"R";
            case 2: return u8"Ipull";
            case 3: return u8"Idrop";
            case 4: return u8"Ron";
            case 5: return u8"Roff";
            case 6: return u8"OperateDelay";
            case 7: return u8"ReleaseDelay";
            case 8: return u8"InitialEngaged";
            case 9: return u8"Engaged";
            case 10: return u8"CoilCurrent";
            default: return {};
        }
    }

    inline bool init_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r) noexcept
    {
        if(!r.valid()) { return false; }
        r.engaged = r.initial_engaged;
        r.pending = -1;
        return true;
    }

    inline bool prepare_foundation_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r) noexcept { return r.valid(); }

    inline void relay_spdt_stamp_branch(relay_current_spdt const& r,
                                        ::phy_engine::MNA::MNA& mna,
                                        unsigned a,
                                        unsigned b,
                                        unsigned branch_index,
                                        ::std::complex<double> impedance,
                                        double history = 0.0) noexcept
    {
        auto const pa{r.pins[a].nodes};
        auto const pb{r.pins[b].nodes};
        if(pa == nullptr || pb == nullptr) { return; }
        auto const k{r.branches[branch_index].index};
        // += is important when two original pins are wired to the same node.
        mna.B_ref(pa->node_index, k) += 1.0;
        mna.B_ref(pb->node_index, k) -= 1.0;
        mna.C_ref(k, pa->node_index) += 1.0;
        mna.C_ref(k, pb->node_index) -= 1.0;
        mna.D_ref(k, k) -= impedance;
        mna.E_ref(k) += history;
    }

    inline void relay_spdt_stamp_contacts(relay_current_spdt const& r, ::phy_engine::MNA::MNA& mna) noexcept
    {
        relay_spdt_stamp_branch(r, mna, 1, 0, 1, r.engaged ? r.Roff : r.Ron);
        relay_spdt_stamp_branch(r, mna, 1, 2, 2, r.engaged ? r.Ron : r.Roff);
    }

    inline bool iterate_dc_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r, ::phy_engine::MNA::MNA& mna) noexcept
    {
        if(!r.valid()) { return false; }
        r.in_transient = false;
        r.check_dc_state = true;
        r.engaged = r.requested_state(r.engaged);
        r.pending = -1;
        relay_spdt_stamp_branch(r, mna, 3, 4, 0, r.R);
        relay_spdt_stamp_contacts(r, mna);
        return true;
    }

    inline bool iterate_ac_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r, ::phy_engine::MNA::MNA& mna, double omega) noexcept
    {
        if(!r.valid() || !::std::isfinite(omega) || omega < 0.0) { return false; }
        // Small-signal AC holds the preceding operating-point contact state;
        // an AC phasor magnitude is not an instantaneous actuator current.
        r.in_transient = false;
        r.check_dc_state = false;
        relay_spdt_stamp_branch(r, mna, 3, 4, 0, {r.R, omega * r.L});
        relay_spdt_stamp_contacts(r, mna);
        return true;
    }

    inline bool step_changed_tr_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r, [[maybe_unused]] double old_step, double step) noexcept
    {
        if(!(::std::isfinite(step) && step > 0.0)) { return false; }
        r.tr_req = r.L / step;
        r.tr_history = -r.tr_req * r.branches[0].current.real();
        r.step_engaged = r.engaged;
        r.step_pending = r.pending;
        r.step_pending_since = r.pending_since;
        return ::std::isfinite(r.tr_req) && ::std::isfinite(r.tr_history) && ::std::isfinite(r.R + r.tr_req);
    }

    inline bool iterate_tr_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r, ::phy_engine::MNA::MNA& mna, double time) noexcept
    {
        if(!r.valid() || !::std::isfinite(time) || time < 0.0) { return false; }
        r.in_transient = true;
        r.trial_time = time;
        // Re-evaluate from the same prior time-step state on every Newton
        // iteration. Iteration count must never accumulate mechanical time.
        auto const transition{r.transition_at(time)};
        r.engaged = transition.state;
        r.pending = transition.pending;
        r.pending_since = transition.since;
        relay_spdt_stamp_branch(r, mna, 3, 4, 0, r.R + r.tr_req, r.tr_history);
        relay_spdt_stamp_contacts(r, mna);
        return true;
    }

    inline bool check_convergence_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt const& r) noexcept
    {
        if(r.in_transient)
        {
            auto const next{r.transition_at(r.trial_time)};
            return next.state == r.engaged && next.pending == r.pending && next.since == r.pending_since;
        }
        return !r.check_dc_state || r.requested_state(r.engaged) == r.engaged;
    }

    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r) noexcept { return {r.pins, 5}; }

    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<relay_current_spdt>, relay_current_spdt& r) noexcept
    { return {r.branches, 3}; }
}  // namespace phy_engine::model
