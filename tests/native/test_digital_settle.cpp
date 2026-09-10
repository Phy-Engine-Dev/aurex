#include <phy_engine/phy_engine.h>
#include <phy_engine/model/models/digital/verilog_module.h>
#include <phy_engine/model/models/digital/logical/tick_delay.h>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace phy_engine;
using dns=model::digital_node_statement_t;
#define CHECK(x) do {if(!(x)) throw std::runtime_error(std::string(__func__)+": " #x);}while(false)
struct network
{
    circult c{};
    auto& node(){return netlist::create_node(c.nl);}
    auto* input(model::node_t& n,dns state) {auto [m,p]=netlist::add_model(c.nl,model::INPUT{.outputA=state}); netlist::add_to_node(c.nl,*m,0,n);return m;}
    void invert(model::node_t& i,model::node_t& o){auto [m,p]=netlist::add_model(c.nl,model::NOT{});netlist::add_to_node(c.nl,*m,0,i);netlist::add_to_node(c.nl,*m,1,o);}
    void dff(model::node_t& d,model::node_t& clk,model::node_t& q){auto [m,p]=netlist::add_model(c.nl,model::DFF{});netlist::add_to_node(c.nl,*m,0,d);netlist::add_to_node(c.nl,*m,1,clk);netlist::add_to_node(c.nl,*m,2,q);}
    void and_gate(model::node_t& a,model::node_t& b,model::node_t& o){auto [m,p]=netlist::add_model(c.nl,model::AND{});netlist::add_to_node(c.nl,*m,0,a);netlist::add_to_node(c.nl,*m,1,b);netlist::add_to_node(c.nl,*m,2,o);}
    void prepare(){c.set_analyze_type(analyze_type::TR);c.prepare();}
};
int main(int argc,char** argv)
{
    try
    {
        network good;auto& a=good.node();auto& b=good.node();good.input(a,dns::H);good.invert(a,b);good.prepare();
        CHECK(good.c.digital_clk().settled);CHECK(b.node_information.dn.state==dns::L);
        network conflict;auto& d=conflict.node();auto& clk=conflict.node();auto& q=conflict.node();auto& out=conflict.node();
        conflict.input(d,dns::H);conflict.input(clk,dns::L);conflict.and_gate(d,d,q);conflict.dff(d,clk,q);conflict.invert(d,out);conflict.prepare();
        auto const& rejected=conflict.c.digital_clk();CHECK(!rejected.settled);CHECK(rejected.reason==digital::settle_reason::multiple_drivers);CHECK(rejected.multiple_driver_nodes==1);CHECK(rejected.processed_events==0);
        network floating;auto& fd=floating.node();auto& fc=floating.node();auto& fq=floating.node();floating.input(fd,dns::H);floating.dff(fd,fc,fq);floating.prepare();CHECK(floating.c.digital_clk().settled);CHECK(floating.c.digital_settle.undriven_nodes==1);
        network oscillation;auto& loop=oscillation.node();auto& in=oscillation.node();auto& independent=oscillation.node();oscillation.invert(loop,loop);oscillation.input(in,dns::H);oscillation.invert(in,independent);oscillation.prepare();loop.node_information.dn.state=dns::L;oscillation.c.digital_event_budget=64;
        CHECK(!oscillation.c.digital_clk().settled);CHECK(oscillation.c.digital_settle.reason==digital::settle_reason::event_budget);CHECK(oscillation.c.digital_settle.pending_nodes>0);CHECK(oscillation.c.digital_settle.processed_events==64);CHECK(independent.node_information.dn.state==dns::L);CHECK(!oscillation.c.digital_settle.hot.empty());
        network clocked;auto& ci=clocked.node();auto& cd=clocked.node();auto& q1=clocked.node();auto& q2=clocked.node();auto* clock=clocked.input(ci,dns::L);clocked.input(cd,dns::H);clocked.dff(cd,ci,q1);clocked.dff(cd,ci,q2);clocked.prepare();CHECK(clocked.c.digital_clk().settled);CHECK(q1.node_information.dn.state==dns::L);clock->ptr->set_attribute(0,{.digital=dns::H,.type=model::variant_type::digital});CHECK(clocked.c.digital_clk().settled);CHECK(q1.node_information.dn.state==dns::H && q2.node_information.dn.state==dns::H);
        network pulse;auto& pc=pulse.node();auto& inverted=pulse.node();auto& po=pulse.node();auto* source=pulse.input(pc,dns::L);pulse.invert(pc,inverted);pulse.and_gate(pc,inverted,po);pulse.prepare();CHECK(pulse.c.digital_clk().settled);CHECK(po.node_information.dn.state==dns::L);source->ptr->set_attribute(0,{.digital=dns::H,.type=model::variant_type::digital});CHECK(pulse.c.digital_clk().settled);CHECK(po.node_information.dn.state==dns::L);
        auto module=model::make_verilog_module(u8"module top(input a, output y); assign y = ~a; endmodule",u8"top");
        CHECK(module.design && module.top_instance.mod);CHECK(module.digital_pin_role(0)==0);CHECK(module.digital_pin_role(1)==1);CHECK(module.digital_pin_role(2)==-1);
        network dynamic_good;auto& da=dynamic_good.node();auto& dy=dynamic_good.node();dynamic_good.input(da,dns::H);
        auto [vm,vm_pos]=netlist::add_model(dynamic_good.c.nl,model::VERILOG_MODULE{module});netlist::add_to_node(dynamic_good.c.nl,*vm,0,da);netlist::add_to_node(dynamic_good.c.nl,*vm,1,dy);
        auto [vo,vo_pos]=netlist::add_model(dynamic_good.c.nl,model::OUTPUT{});netlist::add_to_node(dynamic_good.c.nl,*vo,0,dy);
        CHECK(vm->ptr->get_digital_pin_role(0)==0 && vm->ptr->get_digital_pin_role(1)==1);dynamic_good.prepare();CHECK(dynamic_good.c.digital_clk().settled);CHECK(dynamic_good.c.digital_clk().settled);CHECK(dy.node_information.dn.state==dns::L);
        auto [other,other_pos]=netlist::add_model(dynamic_good.c.nl,model::VERILOG_MODULE{module});netlist::add_to_node(dynamic_good.c.nl,*other,0,da);netlist::add_to_node(dynamic_good.c.nl,*other,1,dy);dynamic_good.prepare();CHECK(!dynamic_good.c.digital_clk().settled);CHECK(dynamic_good.c.digital_settle.reason==digital::settle_reason::multiple_drivers);CHECK(dynamic_good.c.digital_settle.multiple_driver_nodes==1);
        auto inout=model::make_verilog_module(u8"module top(inout y); assign y = 1'b1; endmodule",u8"top");CHECK(inout.top_instance.mod);CHECK(inout.digital_pin_role(0)==1);
        network delay;auto [delay_model,delay_pos]=netlist::add_model(delay.c.nl,model::TICK_DELAY{});CHECK(delay_model->ptr->generate_pin_view().size==2);CHECK(delay_model->ptr->get_digital_pin_role(0)==0 && delay_model->ptr->get_digital_pin_role(1)==1);
        if(argc>1)
        {
            auto* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);CHECK(lib);
            auto tick=reinterpret_cast<int(*)(void*)>(dlsym(lib,"circuit_digital_clk"));CHECK(tick);CHECK(tick(&conflict.c)==11);CHECK(tick(&oscillation.c)==10);
            CHECK(tick(&dynamic_good.c)==11);
            using sample_fn=int(*)(void*,std::size_t*,std::size_t*,std::size_t,std::size_t,double*,double*,std::size_t*,double*,double*,std::size_t*,std::uint8_t*);
            auto sample=reinterpret_cast<sample_fn>(dlsym(lib,"circuit_sample_complex"));CHECK(sample);std::size_t position{};double value{};std::uint8_t digital{};
            CHECK(sample(&conflict.c,&position,&position,0,0,&value,&value,&position,&value,&value,&position,&digital)==11);
            CHECK(sample(&oscillation.c,&position,&position,0,0,&value,&value,&position,&value,&value,&position,&digital)==10);
            using bounded=int(*)(void*,double,double,std::size_t,std::size_t,double*,std::size_t*,std::size_t*);
            auto run=reinterpret_cast<bounded>(dlsym(lib,"circuit_run_transient_bounded_configured"));CHECK(run);double stop{};std::size_t steps{},ticks{};CHECK(run(&conflict.c,0.01,0.02,2,1,&stop,&steps,&ticks)==11);CHECK(steps==0 && ticks==0 && stop==0);
            auto status=reinterpret_cast<std::size_t(*)(void*,char*,std::size_t)>(dlsym(lib,"circuit_get_digital_settle_json"));CHECK(status);auto size=status(&conflict.c,nullptr,0);CHECK(size<65536);std::string data(size,'\0');CHECK(status(&conflict.c,data.data(),size)==size);CHECK(data.find("DIGITAL_MULTIPLE_DRIVERS")!=std::string::npos);CHECK(data.find("model_chunk")!=std::string::npos);
            std::cout<<"settle-json:"<<data.c_str()<<'\n';
            size=status(&oscillation.c,nullptr,0);data.assign(size,'\0');CHECK(status(&oscillation.c,data.data(),size)==size);
            std::cout<<"settle-json:"<<data.c_str()<<'\n';
            dlclose(lib);
        }
        std::cout<<"digital settle: inverter, multi-driver, undriven clock, fairness, shared-clock DFF, zero-delay pulse, dynamic Verilog ports, TICK_DELAY pin roles, failure ABI PASS\n";
    }
    catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
