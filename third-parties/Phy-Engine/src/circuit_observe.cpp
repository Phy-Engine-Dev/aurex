#include <phy_engine/phy_engine.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>

// Bounded observation ABI. Unlike the legacy real-only sampler, this preserves
// AC phase and four-state digital values and never reads analog union storage
// for an exclusively digital node.
extern "C" int circuit_sample_complex(void* circuit_ptr, std::size_t* vec_pos,
    std::size_t* chunk_pos, std::size_t comp_size, std::size_t capacity,
    double* vr, double* vi, std::size_t* vo, double* ir, double* ii,
    std::size_t* io, std::uint8_t* digital)
{
    if (!circuit_ptr || !vec_pos || !chunk_pos || !vr || !vi || !vo || !ir || !ii || !io || !digital) return 1;
    auto* circuit = static_cast<phy_engine::circult*>(circuit_ptr);
    auto& netlist = circuit->get_netlist();
    vo[0] = io[0] = 0;
    for (std::size_t c = 0; c < comp_size; ++c)
    {
        auto* model = phy_engine::netlist::get_model(netlist, phy_engine::netlist::model_pos{vec_pos[c], chunk_pos[c]});
        if (!model || !model->ptr) return 2;
        auto pins = model->ptr->generate_pin_view();
        auto branches = model->ptr->generate_branch_view();
        if (pins.size > capacity - vo[c] || branches.size > capacity - io[c]) return 3;
        for (std::size_t p = 0; p < pins.size; ++p)
        {
            auto* node = pins.pins[p].nodes;
            auto n = vo[c] + p;
            vr[n] = vi[n] = 0;
            digital[n] = 2;
            if (!node) continue;
            if (node->num_of_analog_node != 0)
            {
                vr[n] = node->node_information.an.voltage.real();
                vi[n] = node->node_information.an.voltage.imag();
                if(!std::isfinite(vr[n]) || !std::isfinite(vi[n])) return 4;
            }
            else
            {
                switch (node->node_information.dn.state)
                {
                    case phy_engine::model::digital_node_statement_t::L: digital[n] = 0; break;
                    case phy_engine::model::digital_node_statement_t::H: digital[n] = 1; break;
                    case phy_engine::model::digital_node_statement_t::Z: digital[n] = 3; break;
                    default: digital[n] = 2;
                }
            }
        }
        for (std::size_t b = 0; b < branches.size; ++b)
        {
            ir[io[c] + b] = branches.branches[b].current.real();
            ii[io[c] + b] = branches.branches[b].current.imag();
            if(!std::isfinite(ir[io[c] + b]) || !std::isfinite(ii[io[c] + b])) return 4;
        }
        vo[c + 1] = vo[c] + pins.size;
        io[c + 1] = io[c] + branches.size;
    }
    return 0;
}

// Read-only typed model attributes (including behavior-state flags). The
// caller chooses catalog-declared indices; no guessed memory layout is read.
extern "C" int circuit_get_model_scalar(void* circuit_ptr, std::size_t vec_pos,
    std::size_t chunk_pos, std::size_t attribute, double* output)
{
    if(!circuit_ptr || !output) return 1;
    auto* circuit=static_cast<phy_engine::circult*>(circuit_ptr);
    auto* model=phy_engine::netlist::get_model(circuit->get_netlist(),phy_engine::netlist::model_pos{vec_pos,chunk_pos});
    if(!model || !model->ptr) return 2;
    auto value=model->ptr->get_attribute(attribute);
    if(value.type==phy_engine::model::variant_type::d) *output=value.d;
    else if(value.type==phy_engine::model::variant_type::boolean) *output=value.boolean?1:0;
    else if(value.type==phy_engine::model::variant_type::ui8) *output=value.ui8;
    else return 3;
    return std::isfinite(*output)?0:4;
}

extern "C" int circuit_get_model_digital(void* circuit_ptr, std::size_t vec_pos,
    std::size_t chunk_pos, std::size_t attribute, std::uint8_t* output)
{
    if(!circuit_ptr || !output) return 1;
    auto* circuit=static_cast<phy_engine::circult*>(circuit_ptr);
    auto* model=phy_engine::netlist::get_model(circuit->get_netlist(),phy_engine::netlist::model_pos{vec_pos,chunk_pos});
    if(!model || !model->ptr) return 2;
    auto value=model->ptr->get_attribute(attribute);
    if(value.type!=phy_engine::model::variant_type::digital) return 3;
    switch(value.digital)
    {
        case phy_engine::model::digital_node_statement_t::L: *output=0; break;
        case phy_engine::model::digital_node_statement_t::H: *output=1; break;
        case phy_engine::model::digital_node_statement_t::X: *output=2; break;
        case phy_engine::model::digital_node_statement_t::Z: *output=3; break;
        default: return 4;
    }
    return 0;
}

// A bounded exact-endpoint TR driver using the same native prepare/stamp/
// nonlinear solve path. Unlike the legacy accumulated floating-point while
// loop, 0.5s / 50us is exactly 10000 solves and ends at precisely base+0.5s.
using circuit_trace_callback = int (*)(void*, double, std::size_t);
using circuit_control_callback = int (*)(void*, double, std::size_t);

// Version 1: each completed TR solve advances digital models exactly once,
// before observation, including steps without a callback. No final extra tick.
extern "C" int circuit_transient_digital_propagation_version() { return 1; }

// Configured entry points use 1..64 complete digital ticks per TR solve.
// Ticks advance state (e.g. TICK_DELAY); they are not a convergence tolerance
// or a physical-time interval. The existing entry points retain count=1.
extern "C" int circuit_transient_digital_propagation_configured_version() { return 1; }

// Version 1 calls the control callback immediately before every requested TR
// step. The callback receives that solve's target time and one-based step
// number, so changing a switch/resistance/source affects that exact solve.
extern "C" int circuit_transient_control_version() { return 1; }

static bool circuit_has_hybrid_node(phy_engine::circult const& circuit)
{
    auto hybrid = [](phy_engine::model::node_t const& node) {
        return node.num_of_analog_node != 0 && node.num_of_analog_node != node.pins.size();
    };
    if(hybrid(circuit.nl.ground_node)) return true;
    for(auto const& block:circuit.nl.nodes)
        for(auto node=block.begin;node!=block.curr;++node)
            if(hybrid(*node)) return true;
    return false;
}

// One configured digital propagation followed by an MNA solve. The digital
// tick reads the preceding analog state, emits finite ideal Ll/Hl drives, and
// prepare() allocates those source branches before the analog solve. Repeating
// this operation is an explicit propagation count, never a hidden time step.
enum class mixed_tick_result : unsigned char
{
    ok,
    invalid_drive,
    conflicting_drive,
    analog_solve_failed,
};

static mixed_tick_result circuit_mixed_tick(phy_engine::circult& circuit)
{
    circuit.digital_clk();
    // A single combinational settle can revisit one model and emit its old
    // then final voltage. Keep only that driver's final value. Without model
    // identity, this harmless transition was indistinguishable from two
    // independent outputs fighting on one analog node.
    decltype(circuit.digital_out) final_by_driver{};
    for(auto const drive : circuit.digital_out)
    {
        if(!drive.need_to_operate_analog_node || !drive.driver || !std::isfinite(drive.voltage))
            return mixed_tick_result::invalid_drive;
        bool replaced{};
        for(auto& prior : final_by_driver)
        {
            if(prior.driver == drive.driver &&
               prior.need_to_operate_analog_node == drive.need_to_operate_analog_node)
            {
                prior = drive;
                replaced = true;
                break;
            }
        }
        if(!replaced) final_by_driver.push_back(drive);
    }
    decltype(circuit.digital_out) unique{};
    for(auto const drive : final_by_driver)
    {
        bool found{};
        for(auto const prior : unique)
        {
            if(prior.need_to_operate_analog_node != drive.need_to_operate_analog_node) continue;
            if(prior.voltage != drive.voltage) return mixed_tick_result::conflicting_drive;
            found = true;
            break;
        }
        if(!found) unique.push_back(drive);
    }
    circuit.digital_out = ::std::move(unique);
    circuit.prepare();
    return circuit.solve() ? mixed_tick_result::ok : mixed_tick_result::analog_solve_failed;
}

static int mixed_tick_error_code(mixed_tick_result result) noexcept
{
    switch(result)
    {
        case mixed_tick_result::invalid_drive: return 5;
        case mixed_tick_result::conflicting_drive: return 6;
        case mixed_tick_result::analog_solve_failed: return 7;
        default: return 0;
    }
}

static bool circuit_solution_finite(phy_engine::circult const& circuit)
{
    for(auto node:circuit.size_t_to_node_p)
    {
        auto v=node->node_information.an.voltage;
        if(!std::isfinite(v.real())||!std::isfinite(v.imag())) return false;
    }
    for(auto branch:circuit.size_t_to_branch_p)
        if(!std::isfinite(branch->current.real())||!std::isfinite(branch->current.imag())) return false;
    return true;
}

// Static mixed analysis is a bounded zero-time fixed point. Stateful digital
// loops that do not settle are rejected rather than reported as a DC result.
extern "C" int circuit_run_mixed_dc(void* circuit_ptr, std::uint32_t analyze_type)
{
    if(!circuit_ptr) return 1;
    if(analyze_type > 1) return 2;
    auto& circuit = *static_cast<phy_engine::circult*>(circuit_ptr);
    circuit.set_analyze_type(static_cast<phy_engine::analyze_type>(analyze_type));
    circuit.prepare();
    decltype(circuit.digital_out) previous{};
    for(std::size_t iteration{}; iteration != 64; ++iteration)
    {
        auto const tick{circuit_mixed_tick(circuit)};
        if(tick != mixed_tick_result::ok) return mixed_tick_error_code(tick);
        if(!circuit_solution_finite(circuit)) return 8;
        bool same = previous.size() == circuit.digital_out.size();
        if(same)
        {
            for(std::size_t i{}; i != previous.size(); ++i)
                if(previous[i].need_to_operate_analog_node != circuit.digital_out[i].need_to_operate_analog_node
                   || previous[i].voltage != circuit.digital_out[i].voltage) { same = false; break; }
        }
        if(same) return 0;
        previous = circuit.digital_out;
    }
    return 4;
}

static int circuit_run_transient_trace_controlled_impl(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, std::size_t sample_every,
    std::size_t digital_steps_per_tr_step, circuit_trace_callback callback,
    void* user, circuit_control_callback control, void* control_user,
    double* actual_stop, std::size_t* actual_steps,
    std::size_t* actual_samples, std::size_t* actual_digital_steps)
{
    if(!circuit_ptr || !actual_stop || !actual_steps || !actual_samples || !actual_digital_steps) return 1;
    auto* circuit=static_cast<phy_engine::circult*>(circuit_ptr);
    double base=circuit->tr_duration;
    *actual_stop=base;*actual_steps=0;*actual_samples=0;*actual_digital_steps=0;
    if(digital_steps_per_tr_step==0 || digital_steps_per_tr_step>64) return 2;
    if(!std::isfinite(step)||!std::isfinite(stop)||step<=0||stop<=0||step>stop||max_steps==0||max_steps>100000) return 2;
    double ratio=stop/step,nearest=std::round(ratio);
    if(!std::isfinite(ratio)||ratio>static_cast<double>(max_steps)+1e-8) return 2;
    if(std::abs(ratio-nearest)<=32*std::numeric_limits<double>::epsilon()*std::max(1.0,ratio)) ratio=nearest;
    auto count=static_cast<std::size_t>(std::ceil(ratio));
    if(count==0||count>max_steps) return 2;
    if(count>std::numeric_limits<std::size_t>::max()/digital_steps_per_tr_step) return 2;
    // Sample only completed native solves. No invented t=0 operating point.
    // The final point is always included, without duplicating a regular sample.
    if(callback && (sample_every==0 || (count/sample_every+(count%sample_every!=0))>201)) return 2;
    if(!std::isfinite(base)||!std::isfinite(base+stop)||base+stop<=base) return 2;
    bool const mixed = circuit_has_hybrid_node(*circuit);
    circuit->set_analyze_type(phy_engine::analyze_type::TR);
    circuit->analyzer_setting.tr.t_step=step;
    circuit->analyzer_setting.tr.t_stop=stop;
    circuit->prepare();
    for(std::size_t i=0;i<count;++i)
    {
        double target=base+(i+1==count?stop:std::min(stop,static_cast<double>(i+1)*step));
        double prior=circuit->tr_duration;
        if(control && control(control_user,target,i+1)!=0) return 9;
        circuit->update_tr_step(target-prior);
        circuit->tr_duration=target;
        if(mixed)
        {
            for(std::size_t tick=0; tick<digital_steps_per_tr_step; ++tick)
            {
                auto const result{circuit_mixed_tick(*circuit)};
                if(result != mixed_tick_result::ok){circuit->tr_duration=prior;return mixed_tick_error_code(result);}
            }
        }
        else
        {
            if(!circuit->solve()){circuit->tr_duration=prior;return 3;}
            for(std::size_t tick=0;tick<digital_steps_per_tr_step;++tick) circuit->digital_clk();
        }
        if(!circuit_solution_finite(*circuit)){circuit->tr_duration=prior;return 3;}
        *actual_digital_steps+=digital_steps_per_tr_step;
        *actual_stop=target;*actual_steps=i+1;
        if(callback && ((i+1)%sample_every==0 || i+1==count))
        {
            ++*actual_samples;
            if(callback(user,circuit->tr_duration,i+1)!=0) return 4;
        }
    }
    return 0;
}

extern "C" int circuit_run_transient_trace_controlled(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, std::size_t sample_every,
    std::size_t digital_steps_per_tr_step, circuit_trace_callback callback,
    void* user, circuit_control_callback control, void* control_user,
    double* actual_stop, std::size_t* actual_steps,
    std::size_t* actual_samples, std::size_t* actual_digital_steps)
{
    if(!control) return 2;
    return circuit_run_transient_trace_controlled_impl(circuit_ptr,step,stop,max_steps,
        sample_every,digital_steps_per_tr_step,callback,user,control,control_user,
        actual_stop,actual_steps,actual_samples,actual_digital_steps);
}

extern "C" int circuit_run_transient_trace_configured(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, std::size_t sample_every,
    std::size_t digital_steps_per_tr_step, circuit_trace_callback callback,
    void* user, double* actual_stop, std::size_t* actual_steps,
    std::size_t* actual_samples, std::size_t* actual_digital_steps)
{
    return circuit_run_transient_trace_controlled_impl(circuit_ptr,step,stop,max_steps,
        sample_every,digital_steps_per_tr_step,callback,user,nullptr,nullptr,
        actual_stop,actual_steps,actual_samples,actual_digital_steps);
}

extern "C" int circuit_run_transient_trace(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, std::size_t sample_every,
    circuit_trace_callback callback, void* user, double* actual_stop,
    std::size_t* actual_steps, std::size_t* actual_samples)
{
    std::size_t digital_steps{};
    return circuit_run_transient_trace_configured(circuit_ptr,step,stop,max_steps,
        sample_every,1,callback,user,actual_stop,actual_steps,actual_samples,&digital_steps);
}

extern "C" int circuit_run_transient_bounded_configured(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, std::size_t digital_steps_per_tr_step,
    double* actual_stop, std::size_t* actual_steps, std::size_t* actual_digital_steps)
{
    std::size_t samples{};
    return circuit_run_transient_trace_configured(circuit_ptr,step,stop,max_steps,
        0,digital_steps_per_tr_step,nullptr,nullptr,actual_stop,actual_steps,
        &samples,actual_digital_steps);
}

extern "C" int circuit_run_transient_bounded(void* circuit_ptr, double step,
    double stop, std::size_t max_steps, double* actual_stop, std::size_t* actual_steps)
{
    std::size_t samples{};
    return circuit_run_transient_trace(circuit_ptr,step,stop,max_steps,0,nullptr,
        nullptr,actual_stop,actual_steps,&samples);
}
