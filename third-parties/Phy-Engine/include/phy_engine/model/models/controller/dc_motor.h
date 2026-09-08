#pragma once

#include <cmath>
#include <complex>

#include <fast_io/fast_io_dsal/string_view.h>

#include "../../model_refs/base.h"

namespace phy_engine::model
{
    // Lumped permanent-magnet DC motor. Electrical and mechanical state are
    // coupled through Kt and Ke. Transient integration deliberately exposes
    // one physical state (omega) and uses a trapezoidal armature companion.
    struct dc_motor
    {
        inline static constexpr ::fast_io::u8string_view model_name{u8"DC motor"};
        inline static constexpr model_device_type device_type{model_device_type::non_linear};
        inline static constexpr ::fast_io::u8string_view identification_name{u8"DC_MOTOR"};

        double resistance{1.0};
        double inductance{1e-3};
        double torque_constant{0.01};
        double inertia{1e-5};
        double load_torque{};
        double back_emf_constant{0.01};
        double viscous_friction{1e-5};
        double initial_omega{};

        double omega{};
        double tr_step{};
        double tr_req{};
        double tr_ueq{};
        pin pins[2]{{{u8"+"}}, {{u8"-"}}};
        branch branches{};

        [[nodiscard]] bool valid() const noexcept
        {
            return ::std::isfinite(resistance) && resistance > 0.0 &&
                   ::std::isfinite(inductance) && inductance > 0.0 &&
                   ::std::isfinite(torque_constant) && torque_constant > 0.0 &&
                   ::std::isfinite(inertia) && inertia > 0.0 &&
                   ::std::isfinite(load_torque) && load_torque >= 0.0 &&
                   ::std::isfinite(back_emf_constant) && back_emf_constant > 0.0 &&
                   ::std::isfinite(viscous_friction) && viscous_friction > 0.0 &&
                   ::std::isfinite(initial_omega);
        }
        [[nodiscard]] double voltage() const noexcept
        {
            if(pins[0].nodes == nullptr || pins[1].nodes == nullptr) { return 0.0; }
            return (pins[0].nodes->node_information.an.voltage -
                    pins[1].nodes->node_information.an.voltage).real();
        }
        [[nodiscard]] double current() const noexcept { return branches.current.real(); }
    };

    static_assert(model<dc_motor>);

    inline bool set_attribute_define(model_reserve_type_t<dc_motor>, dc_motor& m,
                                     ::std::size_t index, variant value) noexcept
    {
        if(value.type != variant_type::d || !::std::isfinite(value.d)) { return false; }
        switch(index)
        {
            case 0: if(value.d <= 0.0) return false; m.resistance = value.d; return true;
            case 1: if(value.d <= 0.0) return false; m.inductance = value.d; return true;
            case 2: if(value.d <= 0.0) return false; m.torque_constant = value.d; return true;
            case 3: if(value.d <= 0.0) return false; m.inertia = value.d; return true;
            case 4: if(value.d < 0.0) return false; m.load_torque = value.d; return true;
            case 5: if(value.d <= 0.0) return false; m.back_emf_constant = value.d; return true;
            case 6: if(value.d <= 0.0) return false; m.viscous_friction = value.d; return true;
            case 7: m.initial_omega = value.d; return true;
            default: return false;
        }
    }

    inline variant get_attribute_define(model_reserve_type_t<dc_motor>, dc_motor const& m,
                                        ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return {.d{m.resistance}, .type{variant_type::d}};
            case 1: return {.d{m.inductance}, .type{variant_type::d}};
            case 2: return {.d{m.torque_constant}, .type{variant_type::d}};
            case 3: return {.d{m.inertia}, .type{variant_type::d}};
            case 4: return {.d{m.load_torque}, .type{variant_type::d}};
            case 5: return {.d{m.back_emf_constant}, .type{variant_type::d}};
            case 6: return {.d{m.viscous_friction}, .type{variant_type::d}};
            case 7: return {.d{m.initial_omega}, .type{variant_type::d}};
            case 8: return {.d{m.omega}, .type{variant_type::d}};
            case 9: return {.d{m.current()}, .type{variant_type::d}};
            case 10: return {.d{m.voltage()}, .type{variant_type::d}};
            case 11: return {.d{m.torque_constant * m.current()}, .type{variant_type::d}};
            case 12: return {.d{m.back_emf_constant * m.omega}, .type{variant_type::d}};
            case 13: return {.d{m.torque_constant * m.current() * m.omega}, .type{variant_type::d}};
            default: return {};
        }
    }

    inline constexpr ::fast_io::u8string_view
        get_attribute_name_define(model_reserve_type_t<dc_motor>, ::std::size_t index) noexcept
    {
        switch(index)
        {
            case 0: return u8"Resistance";
            case 1: return u8"Inductance";
            case 2: return u8"TorqueConstant";
            case 3: return u8"Inertia";
            case 4: return u8"LoadTorque";
            case 5: return u8"BackEmfConstant";
            case 6: return u8"ViscousFriction";
            case 7: return u8"InitialOmega";
            case 8: return u8"Omega";
            case 9: return u8"Current";
            case 10: return u8"Voltage";
            case 11: return u8"Torque";
            case 12: return u8"BackEmf";
            case 13: return u8"MechanicalPower";
            default: return {};
        }
    }

    inline bool init_define(model_reserve_type_t<dc_motor>, dc_motor& m) noexcept
    {
        if(!m.valid()) { return false; }
        m.omega = m.initial_omega;
        return true;
    }

    inline bool prepare_foundation_define(model_reserve_type_t<dc_motor>, dc_motor const& m) noexcept
    {
        return m.valid();
    }

    inline void motor_stamp(dc_motor const& m, MNA::MNA& mna,
                            double series_r, double rhs) noexcept
    {
        if(m.pins[0].nodes == nullptr || m.pins[1].nodes == nullptr) { return; }
        auto const k{m.branches.index};
        mna.B_ref(m.pins[0].nodes->node_index, k) += 1.0;
        mna.B_ref(m.pins[1].nodes->node_index, k) -= 1.0;
        mna.C_ref(k, m.pins[0].nodes->node_index) += 1.0;
        mna.C_ref(k, m.pins[1].nodes->node_index) -= 1.0;
        mna.D_ref(k, k) -= series_r;
        mna.E_ref(k) += rhs;
    }

    inline bool iterate_dc_define(model_reserve_type_t<dc_motor>, dc_motor& m,
                                  MNA::MNA& mna) noexcept
    {
        if(!m.valid()) { return false; }
        auto const drive{m.torque_constant * m.current()};
        if(::std::abs(drive) <= m.load_torque)
        {
            // Static load holds the shaft. This is important for a disconnected
            // fan: a positive load magnitude must never make it spin backward.
            m.omega = 0.0;
            motor_stamp(m, mna, m.resistance, 0.0);
        }
        else
        {
            auto const direction{::std::copysign(1.0, drive)};
            auto const electromechanical_r{m.back_emf_constant * m.torque_constant /
                                           m.viscous_friction};
            m.omega = (drive - direction * m.load_torque) / m.viscous_friction;
            motor_stamp(m, mna, m.resistance + electromechanical_r,
                        -m.back_emf_constant * direction * m.load_torque /
                        m.viscous_friction);
        }
        return true;
    }

    inline bool step_changed_tr_define(model_reserve_type_t<dc_motor>, dc_motor& m,
                                       [[maybe_unused]] double old_step, double new_step) noexcept
    {
        m.tr_step = new_step;
        if(new_step <= 0.0) { return true; }
        auto const old_omega{m.omega};
        auto const old_current{m.current()};
        auto const old_inductor_v{m.voltage() - m.resistance * old_current -
                                   m.back_emf_constant * old_omega};
        m.tr_req = 2.0 * m.inductance / new_step;
        m.tr_ueq = -old_inductor_v - m.tr_req * old_current;

        auto const drive{m.torque_constant * old_current};
        if(m.omega == 0.0 && ::std::abs(drive) <= m.load_torque)
        {
            m.omega = 0.0;
            return true;
        }
        auto const reference{m.omega != 0.0 ? m.omega : drive};
        auto const load{::std::copysign(m.load_torque, reference)};
        auto const steady{(drive - load) / m.viscous_friction};
        auto const decay{::std::exp(-m.viscous_friction * new_step / m.inertia)};
        m.omega = steady + (old_omega - steady) * decay;
        if(old_omega != 0.0 && ::std::signbit(old_omega) != ::std::signbit(m.omega) &&
           ::std::abs(drive) <= m.load_torque)
        {
            m.omega = 0.0;
        }
        return ::std::isfinite(m.omega);
    }

    inline bool iterate_tr_define(model_reserve_type_t<dc_motor>, dc_motor& m,
                                  MNA::MNA& mna, [[maybe_unused]] double time) noexcept
    {
        if(!m.valid()) { return false; }
        if(m.tr_step <= 0.0)
        {
            return iterate_dc_define(model_reserve_type<dc_motor>, m, mna);
        }
        motor_stamp(m, mna, m.resistance + m.tr_req,
                    m.back_emf_constant * m.omega + m.tr_ueq);
        return true;
    }

    inline bool iterate_trop_define(model_reserve_type_t<dc_motor>, dc_motor& m,
                                    MNA::MNA& mna) noexcept
    {
        return iterate_dc_define(model_reserve_type<dc_motor>, m, mna);
    }

    inline bool iterate_ac_define(model_reserve_type_t<dc_motor>, dc_motor const& m,
                                  MNA::MNA& mna, double omega) noexcept
    {
        if(!m.valid()) { return false; }
        // Small-signal locked-rotor armature impedance. Mechanical AC response
        // is intentionally not implied by a saved PhysicsLab fan record.
        if(m.pins[0].nodes == nullptr || m.pins[1].nodes == nullptr) { return true; }
        auto const k{m.branches.index};
        mna.B_ref(m.pins[0].nodes->node_index, k) += 1.0;
        mna.B_ref(m.pins[1].nodes->node_index, k) -= 1.0;
        mna.C_ref(k, m.pins[0].nodes->node_index) += 1.0;
        mna.C_ref(k, m.pins[1].nodes->node_index) -= 1.0;
        mna.D_ref(k, k) -= ::std::complex<double>{m.resistance, omega * m.inductance};
        return true;
    }

    inline constexpr pin_view generate_pin_view_define(model_reserve_type_t<dc_motor>, dc_motor& m) noexcept
    {
        return {m.pins, 2};
    }

    inline constexpr branch_view generate_branch_view_define(model_reserve_type_t<dc_motor>, dc_motor& m) noexcept
    {
        return {__builtin_addressof(m.branches), 1};
    }
}
