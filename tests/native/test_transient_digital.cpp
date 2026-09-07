#include <phy_engine/phy_engine.h>
#include <phy_engine/model/models/digital/logical/tick_delay.h>
#include <dlfcn.h>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using dns = phy_engine::model::digital_node_statement_t;
using node_t = phy_engine::model::node_t;
#define CHECK(condition) do { if(!(condition)) throw std::runtime_error(std::string(__func__) + ": " #condition); } while(false)

struct api
{
    using callback = int (*)(void*, double, std::size_t);
    using trace_fn = int (*)(void*, double, double, std::size_t, std::size_t,
                            callback, void*, double*, std::size_t*, std::size_t*);
    using bounded_fn = int (*)(void*, double, double, std::size_t, double*, std::size_t*);
    using configured_trace_fn = int (*)(void*, double, double, std::size_t, std::size_t,
        std::size_t, callback, void*, double*, std::size_t*, std::size_t*, std::size_t*);
    using configured_bounded_fn = int (*)(void*, double, double, std::size_t,
        std::size_t, double*, std::size_t*, std::size_t*);
    void* library{};
    trace_fn trace{};
    bounded_fn bounded{};
    int (*version)(){};
    configured_trace_fn configured_trace{};
    configured_bounded_fn configured_bounded{};
    int (*configured_version)(){};
    explicit api(char const* path)
    {
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if(!library) throw std::runtime_error(dlerror());
        trace = reinterpret_cast<trace_fn>(dlsym(library, "circuit_run_transient_trace"));
        bounded = reinterpret_cast<bounded_fn>(dlsym(library, "circuit_run_transient_bounded"));
        version = reinterpret_cast<int (*)()>(dlsym(library, "circuit_transient_digital_propagation_version"));
        configured_trace = reinterpret_cast<configured_trace_fn>(dlsym(library, "circuit_run_transient_trace_configured"));
        configured_bounded = reinterpret_cast<configured_bounded_fn>(dlsym(library, "circuit_run_transient_bounded_configured"));
        configured_version = reinterpret_cast<int (*)()>(dlsym(library, "circuit_transient_digital_propagation_configured_version"));
        CHECK(trace && bounded);
    }
    ~api() { if(library) dlclose(library); }
};

struct network
{
    phy_engine::circult circuit{};
    auto& netlist() { return circuit.get_netlist(); }
    node_t& node() { return create_node(netlist()); }
    auto* input(dns value, node_t& n)
    {
        auto [m, pos] = add_model(netlist(), phy_engine::model::INPUT{.outputA=value});
        add_to_node(netlist(), *m, 0, n);
        return m;
    }
    void output(node_t& n)
    {
        auto [m, pos] = add_model(netlist(), phy_engine::model::OUTPUT{});
        add_to_node(netlist(), *m, 0, n);
    }
    void invert(node_t& in, node_t& out)
    {
        auto [m, pos] = add_model(netlist(), phy_engine::model::NOT{});
        add_to_node(netlist(), *m, 0, in);
        add_to_node(netlist(), *m, 1, out);
    }
};

void set_input(auto* input, dns state)
{
    CHECK(input->ptr->set_attribute(0, {.digital=state, .type=phy_engine::model::variant_type::digital}));
}

struct observations
{
    std::vector<node_t*> nodes;
    std::vector<std::vector<dns>> rows;
    std::vector<double> times;
    std::vector<std::size_t> steps;
    std::function<void(std::size_t)> after;
    std::exception_ptr error;
    static int capture(void* data, double time, std::size_t step)
    {
        auto& self = *static_cast<observations*>(data);
        try
        {
            std::vector<dns> row;
            for(auto* node:self.nodes) row.push_back(node->node_information.dn.state);
            self.rows.push_back(std::move(row));
            self.times.push_back(time);
            self.steps.push_back(step);
            if(self.after) self.after(step);
            return 0;
        }
        catch(...) { self.error=std::current_exception(); return 1; }
    }
};

void trace(api& engine, network& n, std::size_t count, std::size_t every, observations& samples)
{
    double base=n.circuit.tr_duration, actual{};
    std::size_t completed{}, captured{};
    int rc=engine.trace(&n.circuit, 0.25, count*0.25, count, every,
                        observations::capture, &samples, &actual, &completed, &captured);
    if(samples.error) std::rethrow_exception(samples.error);
    CHECK(rc==0);
    CHECK(completed==count && captured==samples.rows.size());
    CHECK(actual==base+count*0.25 && actual==n.circuit.tr_duration);
    CHECK(!samples.times.empty() && samples.times.back()==actual);
    for(std::size_t i=0;i<samples.times.size();++i)
        CHECK(samples.times[i]==base+samples.steps[i]*0.25);
}

void capability(api& engine)
{
    CHECK(engine.version && engine.version()==1);
    CHECK(engine.version()==1); // Read-only probe, repeatable without a circuit.
}

void fresh_not_each_frame(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    n.input(dns::H,in); n.invert(in,out); n.output(out);
    observations s{.nodes={&in,&out}};
    trace(engine,n,3,1,s);
    CHECK(s.rows.size()==3);
    for(auto const& row:s.rows) CHECK((row==std::vector<dns>{dns::H,dns::L}));
}

void input_changes_not_stale(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::L,in); n.invert(in,out); n.output(out);
    observations s{.nodes={&in,&out}};
    s.after=[&](std::size_t step) { if(step<4) set_input(input, step%2 ? dns::H : dns::L); };
    trace(engine,n,4,1,s);
    CHECK((s.rows==std::vector<std::vector<dns>>{{dns::L,dns::H},{dns::H,dns::L},{dns::L,dns::H},{dns::H,dns::L}}));
}

void dff_edges_and_hold(api& engine)
{
    network n;
    auto& d=n.node(); auto& clk=n.node(); auto& q=n.node();
    auto* data=n.input(dns::L,d); auto* clock=n.input(dns::L,clk);
    auto [ff,pos]=add_model(n.netlist(),phy_engine::model::DFF{});
    add_to_node(n.netlist(),*ff,0,d); add_to_node(n.netlist(),*ff,1,clk); add_to_node(n.netlist(),*ff,2,q);
    n.output(q);
    std::vector<std::pair<dns,dns>> inputs{{dns::L,dns::L},{dns::H,dns::L},{dns::H,dns::H},
        {dns::L,dns::H},{dns::L,dns::L},{dns::L,dns::H}};
    observations s{.nodes={&q}};
    s.after=[&](std::size_t step) {
        if(step<inputs.size()) { set_input(data,inputs[step].first); set_input(clock,inputs[step].second); }
    };
    trace(engine,n,inputs.size(),1,s);
    CHECK((s.rows==std::vector<std::vector<dns>>{{dns::L},{dns::L},{dns::H},{dns::H},{dns::H},{dns::L}}));
}

void unsampled_steps_exactly_once(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::L,in);
    auto [delay,pos]=add_model(n.netlist(),phy_engine::model::TICK_DELAY{2});
    add_to_node(n.netlist(),*delay,0,in); add_to_node(n.netlist(),*delay,1,out); n.output(out);
    observations warm{.nodes={&out}}; trace(engine,n,1,1,warm);
    CHECK(warm.rows[0][0]==dns::L);
    set_input(input,dns::H);
    observations s{.nodes={&out}}; trace(engine,n,3,2,s);
    CHECK((s.steps==std::vector<std::size_t>{2,3}));
    CHECK((s.rows==std::vector<std::vector<dns>>{{dns::L},{dns::H}}));
}

void bounded_no_callback_exactly_once(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::L,in);
    auto [delay,pos]=add_model(n.netlist(),phy_engine::model::TICK_DELAY{2});
    add_to_node(n.netlist(),*delay,0,in); add_to_node(n.netlist(),*delay,1,out); n.output(out);
    observations warm{.nodes={&out}}; trace(engine,n,1,1,warm);
    set_input(input,dns::H);
    double actual{}; std::size_t steps{};
    CHECK(engine.bounded(&n.circuit,0.25,0.5,2,&actual,&steps)==0);
    CHECK(actual==0.75 && steps==2 && out.node_information.dn.state==dns::L);
    CHECK(engine.bounded(&n.circuit,0.25,0.25,1,&actual,&steps)==0);
    CHECK(actual==1.0 && steps==1 && out.node_information.dn.state==dns::H);
}

void four_state_not_forced_zero(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::X,in); n.invert(in,out); n.output(out);
    observations x{.nodes={&in,&out}}; trace(engine,n,2,1,x);
    for(auto const& row:x.rows) CHECK((row==std::vector<dns>{dns::X,dns::X}));
    set_input(input,dns::Z);
    observations z{.nodes={&in,&out}}; trace(engine,n,2,1,z);
    for(auto const& row:z.rows) CHECK((row==std::vector<dns>{dns::Z,dns::X}));
}

node_t& analog_branch(network& n)
{
    auto& voltage=n.node(); auto& ground=n.netlist().ground_node;
    auto [source,spos]=add_model(n.netlist(),phy_engine::model::VDC{.V=5.0});
    auto [load,rpos]=add_model(n.netlist(),phy_engine::model::resistance{.r=10.0});
    add_to_node(n.netlist(),*source,0,voltage); add_to_node(n.netlist(),*source,1,ground);
    add_to_node(n.netlist(),*load,0,voltage); add_to_node(n.netlist(),*load,1,ground);
    return voltage;
}

void pure_analog_allowed(api& engine)
{
    network n; auto& v=analog_branch(n);
    observations s;
    s.after=[&](std::size_t) { CHECK(std::abs(v.node_information.an.voltage.real()-5.0)<1e-12); };
    trace(engine,n,3,1,s);
}

void disconnected_analog_digital_allowed(api& engine)
{
    network n; auto& v=analog_branch(n);
    auto& in=n.node(); auto& out=n.node();
    n.input(dns::H,in); n.invert(in,out); n.output(out);
    observations s{.nodes={&in,&out}};
    s.after=[&](std::size_t) { CHECK(std::abs(v.node_information.an.voltage.real()-5.0)<1e-12); };
    trace(engine,n,3,1,s);
    for(auto const& row:s.rows) CHECK((row==std::vector<dns>{dns::H,dns::L}));
}

void hybrid_digital_drive_is_solved(api& engine)
{
    network n;
    auto& mixed=n.node(); auto& ground=n.netlist().ground_node;
    n.input(dns::H,mixed);
    auto [load,pos]=add_model(n.netlist(),phy_engine::model::resistance{.r=10.0});
    add_to_node(n.netlist(),*load,0,mixed); add_to_node(n.netlist(),*load,1,ground);
    observations s;
    s.after=[&](std::size_t) { CHECK(std::abs(mixed.node_information.an.voltage.real()-5.0)<1e-12); };
    double actual=-1; std::size_t completed=99,captured=99;
    CHECK(engine.trace(&n.circuit,0.25,0.75,3,1,observations::capture,&s,&actual,&completed,&captured)==0);
    CHECK(actual==0.75 && completed==3 && captured==3);
    CHECK(std::abs(mixed.node_information.an.voltage.real()-5.0)<1e-12);
}

void configured_trace(api& engine, network& n, std::size_t count, std::size_t every,
                      std::size_t ticks, observations& samples)
{
    CHECK(engine.configured_trace);
    double base=n.circuit.tr_duration, actual{};
    std::size_t completed{}, captured{}, digital_steps{};
    int rc=engine.configured_trace(&n.circuit,0.25,count*0.25,count,every,ticks,
        observations::capture,&samples,&actual,&completed,&captured,&digital_steps);
    if(samples.error) std::rethrow_exception(samples.error);
    CHECK(rc==0 && completed==count && captured==samples.rows.size());
    CHECK(actual==base+count*0.25 && actual==n.circuit.tr_duration);
    CHECK(digital_steps==count*ticks);
    CHECK(!samples.times.empty() && samples.times.back()==actual);
    for(std::size_t i=0;i<samples.times.size();++i)
        CHECK(samples.times[i]==base+samples.steps[i]*0.25);
}

void configured_capability(api& engine)
{
    CHECK(engine.configured_version && engine.configured_version()==1);
    CHECK(engine.configured_trace && engine.configured_bounded);
    CHECK(engine.version && engine.version()==1);
}

void configured_three_ticks_unsampled(api& engine)
{
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::L,in);
    auto [delay,pos]=add_model(n.netlist(),phy_engine::model::TICK_DELAY{6});
    add_to_node(n.netlist(),*delay,0,in); add_to_node(n.netlist(),*delay,1,out); n.output(out);
    observations warm{.nodes={&out}}; trace(engine,n,1,1,warm);
    set_input(input,dns::H);
    observations s{.nodes={&out}}; configured_trace(engine,n,3,2,3,s);
    CHECK((s.steps==std::vector<std::size_t>{2,3}));
    CHECK((s.rows==std::vector<std::vector<dns>>{{dns::L},{dns::H}}));
    CHECK((s.times==std::vector<double>{0.75,1.0})); // N does not alter physical time.
}

void configured_bounded_three_ticks(api& engine)
{
    CHECK(engine.configured_bounded);
    network n;
    auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::L,in);
    auto [delay,pos]=add_model(n.netlist(),phy_engine::model::TICK_DELAY{6});
    add_to_node(n.netlist(),*delay,0,in); add_to_node(n.netlist(),*delay,1,out); n.output(out);
    observations warm{.nodes={&out}}; trace(engine,n,1,1,warm);
    set_input(input,dns::H);
    double actual{}; std::size_t steps{},digital_steps{};
    CHECK(engine.configured_bounded(&n.circuit,0.25,0.5,2,3,&actual,&steps,&digital_steps)==0);
    CHECK(actual==0.75 && steps==2 && digital_steps==6 && out.node_information.dn.state==dns::L);
    CHECK(engine.configured_bounded(&n.circuit,0.25,0.25,1,3,&actual,&steps,&digital_steps)==0);
    CHECK(actual==1.0 && steps==1 && digital_steps==3 && out.node_information.dn.state==dns::H);
}

void configured_one_equals_legacy(api& engine)
{
    std::vector<observations> results;
    for(bool use_configured:{false,true})
    {
        network n;
        auto& in=n.node(); auto& out=n.node();
        auto* input=n.input(dns::L,in);
        auto [delay,pos]=add_model(n.netlist(),phy_engine::model::TICK_DELAY{2});
        add_to_node(n.netlist(),*delay,0,in); add_to_node(n.netlist(),*delay,1,out); n.output(out);
        observations warm{.nodes={&out}}; trace(engine,n,1,1,warm);
        set_input(input,dns::H);
        observations s{.nodes={&out}};
        if(use_configured) configured_trace(engine,n,3,1,1,s); else trace(engine,n,3,1,s);
        s.nodes.clear(); results.push_back(std::move(s));
    }
    CHECK(results[0].rows==results[1].rows && results[0].times==results[1].times && results[0].steps==results[1].steps);
    CHECK((results[1].rows==std::vector<std::vector<dns>>{{dns::L},{dns::L},{dns::H}}));
}

void configured_invalid_no_execution(api& engine)
{
    CHECK(engine.configured_trace && engine.configured_bounded);
    for(std::size_t ticks:{std::size_t{0},std::size_t{65},std::numeric_limits<std::size_t>::max()})
    {
        network n; auto& input=n.node(); n.input(dns::H,input);
        observations s;
        double actual=-1; std::size_t steps=99,captured=99,digital_steps=99;
        CHECK(engine.configured_trace(&n.circuit,0.25,0.75,3,1,ticks,observations::capture,&s,
            &actual,&steps,&captured,&digital_steps)==2);
        CHECK(actual==0 && steps==0 && captured==0 && digital_steps==0 && s.rows.empty());
        CHECK(!n.circuit.has_prepare && n.circuit.tr_duration==0);
        CHECK(engine.configured_bounded(&n.circuit,0.25,0.75,3,ticks,&actual,&steps,&digital_steps)==2);
        CHECK(actual==0 && steps==0 && digital_steps==0 && !n.circuit.has_prepare);
    }
}

void configured_overflow_guards(api& engine)
{
    CHECK(engine.configured_trace);
    network n; auto& input=n.node(); n.input(dns::H,input);
    observations s;
    auto check=[&](double step,double stop,std::size_t limit) {
        double actual=-1; std::size_t completed=99,captured=99,ticks=99;
        CHECK(engine.configured_trace(&n.circuit,step,stop,limit,1,64,observations::capture,&s,
            &actual,&completed,&captured,&ticks)==2);
        CHECK(actual==0 && completed==0 && captured==0 && ticks==0 && s.rows.empty());
        CHECK(!n.circuit.has_prepare && n.circuit.tr_duration==0);
    };
    check(0.25,0.75,std::numeric_limits<std::size_t>::max());
    check(std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max(),100000);
    check(0.25,std::numeric_limits<double>::max(),100000);
}

void configured_upper_bound_and_four_state(api& engine)
{
    network n; auto& in=n.node(); auto& out=n.node();
    auto* input=n.input(dns::Z,in); n.invert(in,out); n.output(out);
    observations s{.nodes={&in,&out}}; configured_trace(engine,n,1,1,64,s);
    CHECK((s.rows[0]==std::vector<dns>{dns::Z,dns::X}));
    set_input(input,dns::X);
    observations x{.nodes={&in,&out}}; configured_trace(engine,n,1,1,3,x);
    CHECK((x.rows[0]==std::vector<dns>{dns::X,dns::X}));
}

void configured_hybrid_drive_is_solved(api& engine)
{
    CHECK(engine.configured_trace && engine.configured_bounded);
    network n;
    auto& mixed=n.node(); auto& ground=n.netlist().ground_node;
    n.input(dns::H,mixed);
    auto [load,pos]=add_model(n.netlist(),phy_engine::model::resistance{.r=10.0});
    add_to_node(n.netlist(),*load,0,mixed); add_to_node(n.netlist(),*load,1,ground);
    observations s;
    s.after=[&](std::size_t) { CHECK(std::abs(mixed.node_information.an.voltage.real()-5.0)<1e-12); };
    double actual=-1; std::size_t completed=99,captured=99,ticks=99;
    CHECK(engine.configured_trace(&n.circuit,0.25,0.75,3,1,3,observations::capture,&s,
        &actual,&completed,&captured,&ticks)==0);
    CHECK(actual==0.75 && completed==3 && captured==3 && ticks==9);
    CHECK(std::abs(mixed.node_information.an.voltage.real()-5.0)<1e-12);
}

void configured_analog_unchanged(api& engine)
{
    for(bool with_digital:{false,true})
    {
        network n; auto& v=analog_branch(n);
        observations s;
        if(with_digital)
        {
            auto& in=n.node(); auto& out=n.node(); n.input(dns::H,in); n.invert(in,out); n.output(out);
            s.nodes={&in,&out};
        }
        s.after=[&](std::size_t) { CHECK(std::abs(v.node_information.an.voltage.real()-5.0)<1e-12); };
        configured_trace(engine,n,3,1,3,s);
        if(with_digital) for(auto const& row:s.rows) CHECK((row==std::vector<dns>{dns::H,dns::L}));
    }
}

void configured_callback_failure_accounting(api& engine)
{
    CHECK(engine.configured_trace);
    network n; auto& in=n.node(); n.input(dns::H,in);
    double actual{}; std::size_t completed{},captured{},ticks{};
    auto reject=[](void*,double,std::size_t) -> int { return 1; };
    CHECK(engine.configured_trace(&n.circuit,0.25,0.75,3,1,3,reject,nullptr,
        &actual,&completed,&captured,&ticks)==4);
    CHECK(actual==0.25 && completed==1 && captured==1 && ticks==3);
}

int main(int argc,char** argv)
{
    if(argc!=2) { std::cerr<<"usage: test_transient_digital /absolute/path/libphyengine.so\n"; return 2; }
    try
    {
        api engine(argv[1]);
        using test=std::pair<char const*,void (*)(api&)>;
        std::vector<test> tests{{"capability",capability},{"fresh_not_each_frame",fresh_not_each_frame},
            {"input_changes_not_stale",input_changes_not_stale},{"dff_edges_and_hold",dff_edges_and_hold},
            {"unsampled_steps_exactly_once",unsampled_steps_exactly_once},
            {"bounded_no_callback_exactly_once",bounded_no_callback_exactly_once},
            {"four_state_not_forced_zero",four_state_not_forced_zero},{"pure_analog_allowed",pure_analog_allowed},
            {"disconnected_analog_digital_allowed",disconnected_analog_digital_allowed},
            {"hybrid_digital_drive_is_solved",hybrid_digital_drive_is_solved},
            {"configured_capability",configured_capability},
            {"configured_three_ticks_unsampled",configured_three_ticks_unsampled},
            {"configured_bounded_three_ticks",configured_bounded_three_ticks},
            {"configured_one_equals_legacy",configured_one_equals_legacy},
            {"configured_invalid_no_execution",configured_invalid_no_execution},
            {"configured_overflow_guards",configured_overflow_guards},
            {"configured_upper_bound_and_four_state",configured_upper_bound_and_four_state},
            {"configured_hybrid_drive_is_solved",configured_hybrid_drive_is_solved},
            {"configured_analog_unchanged",configured_analog_unchanged},
            {"configured_callback_failure_accounting",configured_callback_failure_accounting}};
        unsigned failures{};
        for(auto const& [name,run]:tests)
        {
            try { run(engine); std::cout<<"PASS "<<name<<'\n'; }
            catch(std::exception const& error) { ++failures; std::cout<<"FAIL "<<name<<": "<<error.what()<<'\n'; }
        }
        std::cout<<tests.size()-failures<<'/'<<tests.size()<<" passed\n";
        return failures ? 1 : 0;
    }
    catch(std::exception const& error) { std::cerr<<error.what()<<'\n'; return 2; }
}
