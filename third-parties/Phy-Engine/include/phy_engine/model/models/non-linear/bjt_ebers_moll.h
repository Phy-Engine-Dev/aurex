#pragma once
#include <algorithm>
#include <cmath>
#include "../../model_refs/base.h"

namespace phy_engine::model::details
{
    // Quasi-static Ebers-Moll transport model; independent implementation.
    // https://ngspice.sourceforge.io/docs/ngspice-manual.pdf (BJT chapter)
    // https://github.com/ngspice/ngspice/blob/master/src/spicelib/devices/bjt/bjtload.c
    // Reference equations with qb=1, no leakage/parasitic/charge-storage extensions:
    // Ic=If-(1+1/Br)*Ir; Ib=If/Bf+Ir/Br; Ie=-(Ic+Ib).
    // Legacy PE Is was a BE base-current scale, not SPICE transport IS.
    // Preserve forward-active compatibility with transport_Is=Is*BetaF.
    // No Early effect, breakdown, charge storage, parasitic resistance or
    // calibrated temperature scaling of Is/Beta. Temp sets thermal voltage.
    struct bjt_ebers_moll_state
    {
        double Is{1e-16}, N{1.0}, BetaF{100.0}, Temp{27.0}, Area{1.0};
        double BetaR{1.0}, Nr{1.0};
        double Ut{};
        double last_be{}, last_bc{};
        double jacobian[3][3]{};
    };

    inline bool bjt_valid(bjt_ebers_moll_state const& q) noexcept
    {
        return std::isfinite(q.Is) && q.Is>0 && std::isfinite(q.N) && q.N>0
            && std::isfinite(q.BetaF) && q.BetaF>0 && std::isfinite(q.BetaR) && q.BetaR>0
            && std::isfinite(q.Nr) && q.Nr>0 && std::isfinite(q.Temp) && q.Temp>-273.15
            && std::isfinite(q.Area) && q.Area>0 && std::isfinite(q.Is*q.BetaF*q.Area);
    }

    struct bjt_junction { double current{}, conductance{}; };
    inline bjt_junction bjt_diode(double voltage, double thermal, double scale) noexcept
    {
        // A C1 tangent continuation prevents exp overflow during Newton's
        // trial steps. Ordinary silicon operating voltages use exact expm1.
        double const x=voltage/thermal;
        double const ex=std::exp(std::min(x,60.0));
        return {scale*(x>60.0?ex*(1.0+x-60.0)-1.0:std::expm1(x)),
                scale*ex/thermal};
    }

    inline double bjt_limit(double voltage, double previous, double thermal, double scale) noexcept
    {
        // Junction-voltage limiting is numerical iteration control, not an
        // output clamp. check_convergence requires the unlimited solution.
        double const critical=thermal*std::log(thermal/(std::sqrt(2.0)*scale));
        if(voltage>critical && std::abs(voltage-previous)>2*thermal)
        {
            if(previous>0)
            {
                double const argument=1+(voltage-previous)/thermal;
                return argument>0?previous+thermal*std::log(argument):critical;
            }
            return thermal*std::log(std::max(voltage/thermal,1.0));
        }
        return voltage;
    }

    struct bjt_evaluation
    {
        double current[3]{};
        double du[3]{};
        double dv[3]{};
    };
    inline bjt_evaluation bjt_evaluate(bjt_ebers_moll_state const& q, double be, double bc) noexcept
    {
        double const scale=q.Is*q.BetaF*q.Area;
        auto f=bjt_diode(be,q.N*q.Ut,scale);
        auto r=bjt_diode(bc,q.Nr*q.Ut,scale);
        bjt_evaluation out{};
        out.current[0]=f.current/q.BetaF+r.current/q.BetaR;
        out.current[1]=f.current-(1+1/q.BetaR)*r.current;
        out.current[2]=-(out.current[0]+out.current[1]);
        out.du[0]=f.conductance/q.BetaF;
        out.du[1]=f.conductance;
        out.du[2]=-(out.du[0]+out.du[1]);
        out.dv[0]=r.conductance/q.BetaR;
        out.dv[1]=-(1+1/q.BetaR)*r.conductance;
        out.dv[2]=-(out.dv[0]+out.dv[1]);
        return out;
    }

    template<class Q> inline bool bjt_prepare(Q& q) noexcept
    {
        if(!bjt_valid(q)) return false;
        q.Ut=1.380650524e-23*(q.Temp+273.15)/1.6021765314e-19;
        q.last_be=q.last_bc=0;
        return true;
    }

    template<class Q> inline bool bjt_stamp(Q& q, MNA::MNA& mna) noexcept
    {
        if(!bjt_valid(q)) return false;
        auto b=q.pins[0].nodes;auto c=q.pins[1].nodes;auto e=q.pins[2].nodes;
        if(!b||!c||!e) return false;
        constexpr double sign=Q::polarity;
        double be=sign*(b->node_information.an.voltage.real()-e->node_information.an.voltage.real());
        double bc=sign*(b->node_information.an.voltage.real()-c->node_information.an.voltage.real());
        if(!std::isfinite(be)||!std::isfinite(bc)) return false;
        double scale=q.Is*q.BetaF*q.Area;
        be=bjt_limit(be,q.last_be,q.N*q.Ut,scale);
        bc=bjt_limit(bc,q.last_bc,q.Nr*q.Ut,scale);
        q.last_be=be;q.last_bc=bc;
        auto eval=bjt_evaluate(q,be,bc);
        std::size_t nodes[]{b->node_index,c->node_index,e->node_index};
        for(std::size_t row=0;row<3;++row)
        {
            q.jacobian[row][0]=eval.du[row]+eval.dv[row];
            q.jacobian[row][1]=-eval.dv[row];
            q.jacobian[row][2]=-eval.du[row];
            double const rhs=sign*(eval.current[row]-eval.du[row]*be-eval.dv[row]*bc);
            if(!std::isfinite(rhs)) return false;
            for(std::size_t col=0;col<3;++col)
            {
                if(!std::isfinite(q.jacobian[row][col])) return false;
                mna.G_ref(nodes[row],nodes[col])+=q.jacobian[row][col];
            }
            mna.I_ref(nodes[row])-=rhs;
        }
        return true;
    }

    template<class Q> inline bool bjt_converged(Q const& q) noexcept
    {
        auto b=q.pins[0].nodes;auto c=q.pins[1].nodes;auto e=q.pins[2].nodes;
        if(!b||!c||!e) return false;
        double const be=Q::polarity*(b->node_information.an.voltage.real()-e->node_information.an.voltage.real());
        double const bc=Q::polarity*(b->node_information.an.voltage.real()-c->node_information.an.voltage.real());
        return std::isfinite(be)&&std::isfinite(bc)&&std::abs(be-q.last_be)<1e-8&&std::abs(bc-q.last_bc)<1e-8;
    }

    template<class Q> inline bool bjt_ac(Q& q, MNA::MNA& mna) noexcept
    {
        for(std::size_t i=0;i<3;++i)
            for(std::size_t j=0;j<3;++j)
            {
                if(!q.pins[i].nodes||!q.pins[j].nodes) return false;
                mna.G_ref(q.pins[i].nodes->node_index,q.pins[j].nodes->node_index)+=q.jacobian[i][j];
            }
        return true;
    }

    template<class Q> inline variant bjt_get(Q const& q, std::size_t index) noexcept
    {
        double value{};
        switch(index)
        {
            case 0:value=q.Is;break;case 1:value=q.N;break;case 2:value=q.BetaF;break;
            case 3:value=q.Temp;break;case 4:value=q.Area;break;case 5:value=q.BetaR;break;
            case 6:value=q.Nr;break;
            case 16:case 17:case 18:case 19:case 20:
            {
                auto b=q.pins[0].nodes;auto c=q.pins[1].nodes;auto e=q.pins[2].nodes;
                if(!b||!c||!e||q.Ut<=0) return {};
                double const be=Q::polarity*(b->node_information.an.voltage.real()-e->node_information.an.voltage.real());
                double const bc=Q::polarity*(b->node_information.an.voltage.real()-c->node_information.an.voltage.real());
                if(index<19) value=Q::polarity*bjt_evaluate(q,be,bc).current[index-16];
                else value=Q::polarity*(index==19?be:bc);
                break;
            }
            default:return {};
        }
        return {.d{value},.type{variant_type::d}};
    }

    template<class Q> inline bool bjt_set(Q& q, std::size_t index, variant value) noexcept
    {
        if(value.type!=variant_type::d||!std::isfinite(value.d)) return false;
        if(index==3?value.d<=-273.15:value.d<=0) return false;
        switch(index)
        {
            case 0:q.Is=value.d;return true;case 1:q.N=value.d;return true;case 2:q.BetaF=value.d;return true;
            case 3:q.Temp=value.d;return true;case 4:q.Area=value.d;return true;case 5:q.BetaR=value.d;return true;
            case 6:q.Nr=value.d;return true;default:return false;
        }
    }

    inline constexpr fast_io::u8string_view bjt_attribute_name(std::size_t index) noexcept
    {
        switch(index)
        {
            case 0:return u8"Is";case 1:return u8"N";case 2:return u8"BetaF";case 3:return u8"Temp";
            case 4:return u8"Area";case 5:return u8"BetaR";case 6:return u8"Nr";
            case 16:return u8"I_B";case 17:return u8"I_C";case 18:return u8"I_E";
            case 19:return u8"V_BE";case 20:return u8"V_BC";default:return {};
        }
    }
}
