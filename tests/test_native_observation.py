"""Native primitive physics and bounded observation tests, never a design fixture."""
from __future__ import annotations
import ctypes
import math
import os
from pathlib import Path
import unittest
from aurex.phy_engine.ffi import _Lib, PhyEngineError
from aurex.phy_engine.catalog import COMPONENTS

@unittest.skipUnless(os.environ.get("AUREX_PHY_ENGINE_BUILD"), "native build not configured")
class NativeObservationTests(unittest.TestCase):
    def setUp(self):
        self.lib=_Lib(str(Path(os.environ["AUREX_PHY_ENGINE_BUILD"]).resolve()/"libphyengine.so"))
        self.circuits=[]
        self.callback_type=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_double,ctypes.c_size_t)
        self.lib._dll.circuit_run_transient_trace.argtypes=[
            ctypes.c_void_p,ctypes.c_double,ctypes.c_double,ctypes.c_size_t,ctypes.c_size_t,
            self.callback_type,ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        self.lib._dll.circuit_run_transient_trace.restype=ctypes.c_int
        self.lib._dll.circuit_set_model_double_by_name.argtypes=[
            ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_char_p,ctypes.c_size_t,ctypes.c_double]
        self.lib._dll.circuit_set_model_double_by_name.restype=ctypes.c_int
    def tearDown(self):
        for circuit in self.circuits:circuit.close()
    def create(self,components):
        elements=[0];properties=[];nodes={};wires=[]
        for idx,component in enumerate(components):
            meta=COMPONENTS[component["type"]]
            elements.append(meta["code"])
            parameters={**meta["defaults"],**component.get("params",{})}
            properties.extend(parameters[prop] for prop in meta["props"])
            for pin,node in enumerate(component["nodes"]):nodes.setdefault(node,[]).append((idx+1,pin))
        for node,pins in nodes.items():
            first=pins[0]
            if node=="gnd":wires.extend([0,0,first[0],first[1]])
            elif len(pins)==1:wires.extend([first[0],first[1],*first])
            for other in pins[1:]:wires.extend([*first,*other])
        c=self.lib.create_circuit(elements=elements,wires=wires,properties=properties)
        self.circuits.append(c)
        return c
    def rc(self):
        return self.create([
            {"type":"vdc","nodes":["v","gnd"],"params":{"v":1}},
            {"type":"resistor","nodes":["v","c"],"params":{"r":1000}},
            {"type":"capacitor","nodes":["c","gnd"],"params":{"c":1e-4}}])
    def trace(self,c,dt,stop,every=50,cancel=None):
        values=[]
        def take(_user,time,step):
            values.append({"time":time,"step":step,"sample":c.sample_complex(max_pins=3)})
            return int(cancel is not None and len(values)>=cancel)
        callback=self.callback_type(take)
        actual=ctypes.c_double();steps=ctypes.c_size_t();samples=ctypes.c_size_t()
        status=self.lib._dll.circuit_run_transient_trace(
            c.ptr,dt,stop,10000,every,callback,None,ctypes.byref(actual),ctypes.byref(steps),ctypes.byref(samples))
        return status,actual.value,steps.value,samples.value,values
    def test_rc_trace_exact_endpoint_native_samples_and_scalar(self):
        c=self.rc()
        status,actual,steps,count,values=self.trace(c,5e-5,.5)
        self.assertEqual((status,actual,steps,count),(0,.5,10000,200))
        self.assertEqual(c.model_scalar(1,0),1000)
        prior=-1
        for index,value in enumerate(values):
            self.assertEqual(value["step"],(index+1)*50)
            self.assertEqual(value["time"],value["step"]*5e-5)
            sample=value["sample"]
            self.assertEqual(sample["voltage_ord"],[0,2,4,6])
            self.assertEqual(len(sample["current"]),1) # only V source is an MNA branch
            voltage=sample["voltage"][4]
            self.assertGreater(voltage,prior)
            self.assertAlmostEqual(voltage,1-math.exp(-value["time"]/.1),delta=.0003)
            prior=voltage
        self.assertEqual(values[-1]["sample"],c.sample_complex(max_pins=3))
    def test_trace_final_partial_interval_and_sample_budget(self):
        c=self.rc()
        status,actual,steps,count,values=self.trace(c,.0003,.001,every=3)
        self.assertEqual((status,actual,steps,count),(0,.001,4,2))
        self.assertEqual([point["time"] for point in values],[.0009,.001])
        self.assertEqual(self.trace(self.rc(),5e-5,.5,every=1)[0],2)
        self.assertEqual(self.trace(self.rc(),5e-5,.5,every=0)[0],2)

    def test_unsolvable_native_system_does_not_report_success(self):
        c=self.create([
            {"type":"vdc","nodes":["v","gnd"],"params":{"v":1}},
            {"type":"vdc","nodes":["v","gnd"],"params":{"v":2}}])
        c.set_analyze_type(1)
        with self.assertRaises(PhyEngineError):c.analyze()
        self.assertEqual(self.trace(c,.001,.01,every=1)[:4],(3,0,0,0))
    def test_callback_stop_reports_real_completed_time(self):
        status,actual,steps,count,_=self.trace(self.rc(),5e-5,.5,cancel=3)
        self.assertEqual((status,actual,steps,count),(4,.007500000000000001,150,3))
    def test_segmented_rc_retains_capacitor_history(self):
        split=self.rc();whole=self.rc()
        split.run_transient_bounded(.0001,.02)
        result=split.run_transient_bounded(.0001,.03)
        whole.run_transient_bounded(.0001,.05)
        self.assertEqual(result["actual_stop_s"],.05)
        self.assertAlmostEqual(split.sample_complex(max_pins=2)["voltage"][4],
            whole.sample_complex(max_pins=2)["voltage"][4],places=10)
    def bias(self,kind,vb,vc,ve,params=None):
        c=self.create([
            {"type":kind,"nodes":["b","c","e"],"params":params or {}},
            {"type":"vdc","nodes":["b","gnd"],"params":{"v":vb}},
            {"type":"vdc","nodes":["c","gnd"],"params":{"v":vc}},
            {"type":"vdc","nodes":["e","gnd"],"params":{"v":ve}}])
        c.set_analyze_type(1);c.analyze()
        return c
    def expected(self,kind,voltages,params=None):
        p={**COMPONENTS[kind]["defaults"],**(params or {})}
        sign=1 if kind=="npn" else -1
        thermal=1.380650524e-23*(p["temp_c"]+273.15)/1.6021765314e-19
        be=sign*(voltages[0]-voltages[2]);bc=sign*(voltages[0]-voltages[1])
        scale=p["is"]*p["beta"]*p["area"]
        f=scale*math.expm1(be/(p["n"]*thermal));r=scale*math.expm1(bc/thermal)
        ib=f/p["beta"]+r;ic=f-2*r
        return [sign*ib,sign*ic,sign*(-ib-ic)]
    def test_bjt_regions_kcl_source_oracle_and_pnp_mirror(self):
        for region,voltages in [
            ("forward",(0.6,3,0)),("saturation",(.7,.1,0)),
            ("cutoff",(0,.7,.2)),("reverse",(.6,0,3))]:
            pairs=[]
            for kind,sign in [("npn",1),("pnp",-1)]:
                v=[sign*x for x in voltages]
                c=self.bias(kind,*v)
                currents=[c.model_scalar(0,16+i) for i in range(3)]
                expected=self.expected(kind,v)
                sampled=c.sample_complex(max_pins=3)
                for i in range(3):
                    self.assertAlmostEqual(currents[i],expected[i],delta=max(1e-12,abs(expected[i])*1e-6),msg=region)
                    self.assertAlmostEqual(currents[i],-sampled["current"][i],delta=max(1e-12,abs(expected[i])*1e-6),msg=region)
                self.assertAlmostEqual(sum(currents),0,delta=1e-14)
                pairs.append(currents)
            for a,b in zip(*pairs):self.assertAlmostEqual(a,-b,places=10)
    def test_bjt_saturation_resistor_loaded_stays_inside_supply(self):
        c=self.create([
            {"type":"vdc","nodes":["v","gnd"],"params":{"v":5}},
            {"type":"resistor","nodes":["v","b"],"params":{"r":10000}},
            {"type":"resistor","nodes":["v","c"],"params":{"r":1000}},
            {"type":"npn","nodes":["b","c","gnd"]}])
        c.set_analyze_type(1);c.analyze()
        sample=c.sample_complex(max_pins=3)
        vb,vc,ve=sample["voltage"][6:9]
        self.assertTrue(0.0<vc<.3,(vb,vc,ve))
        self.assertTrue(.4<vb<.9,(vb,vc,ve))
        self.assertAlmostEqual(c.model_scalar(3,17),(5-vc)/1000,delta=1e-9)
        self.assertAlmostEqual(c.model_scalar(3,16),(5-vb)/10000,delta=1e-9)
    def test_bjt_jacobian_matches_forced_pin_bias_differences(self):
        base=[.61,.2,0]
        for dimension in range(3):
            h=1e-6;plus=base.copy();minus=base.copy()
            plus[dimension]+=h;minus[dimension]-=h
            a=self.bias("npn",*plus);b=self.bias("npn",*minus)
            actual=[(a.model_scalar(0,16+i)-b.model_scalar(0,16+i))/(2*h) for i in range(3)]
            oracle=[(x-y)/(2*h) for x,y in zip(self.expected("npn",plus),self.expected("npn",minus))]
            for x,y in zip(actual,oracle):self.assertAlmostEqual(x,y,delta=max(1e-9,abs(y)*1e-6))
    def test_diode_forward_primitive_dc(self):
        c=self.create([
            {"type":"vdc","nodes":["v","gnd"],"params":{"v":1}},
            {"type":"resistor","nodes":["v","d"],"params":{"r":1000}},
            {"type":"diode","nodes":["d","gnd"]}])
        c.set_analyze_type(1);c.analyze()
        s=c.sample_complex(max_pins=2)
        self.assertTrue(.4<s["voltage"][4]<.9,s)
        self.assertAlmostEqual(-s["current"][0],(1-s["voltage"][4])/1000,places=9)

    def test_bjt_reverse_gain_emission_configuration(self):
        c=self.bias("npn",.6,0,3)
        c.set_model_scalar(0,"BetaR",7)
        c.set_model_scalar(0,"Nr",1.2)
        self.assertEqual(c.model_scalar(0,5),7)
        self.assertEqual(c.model_scalar(0,6),1.2)
        c.analyze()
        ib,ic,ie=[c.model_scalar(0,16+i) for i in range(3)]
        self.assertAlmostEqual(ie/ib,7,delta=1e-5)
        self.assertAlmostEqual(ib+ic+ie,0,delta=1e-15)
        thermal=1.380650524e-23*300.15/1.6021765314e-19
        expected=1e-12*math.expm1(.6/(1.2*thermal))
        self.assertAlmostEqual(ie,expected,delta=1e-10)

    def test_named_scalar_rejects_invalid_and_readonly_without_false_success(self):
        c=self.bias("npn",.6,3,0)
        for name,value in [(b"BetaR",0),(b"BetaR",-1),(b"BetaR",math.nan),
                           (b"Nr",math.inf),(b"I_C",1)]:
            status=self.lib._dll.circuit_set_model_double_by_name(c.ptr,c.vec_pos[0],c.chunk_pos[0],name,len(name),value)
            self.assertEqual(status,4,(name,value))
        self.assertEqual(c.model_scalar(0,5),1)
        self.assertEqual(c.model_scalar(0,6),1)

    def test_bjt_small_signal_ac_uses_dc_jacobian(self):
        # Each independent pin excitation tests a whole column of the actual
        # stamped AC matrix against finite differences of the DC equations.
        bias=[.61,.2,0];amplitude=1e-4;h=1e-6
        for dimension in range(3):
            components=[{"type":"npn","nodes":["b","c","e"]}]
            branches=[];branch_count=0
            for index,pin in enumerate(["b","c","e"]):
                branches.append(branch_count);branch_count+=1
                node=pin+"bias" if index==dimension else pin
                components.append({"type":"vdc","nodes":[node,"gnd"],"params":{"v":bias[index]}})
                if index==dimension:
                    components.append({"type":"vac","nodes":[pin,node],"params":{"vp":amplitude,"freq_hz":1000}})
                    branch_count+=1
            c=self.create(components)
            c.set_analyze_type(2);c.set_ac_omega(2*math.pi*1000);c.analyze()
            sample=c.sample_complex(max_pins=3)
            plus=bias.copy();minus=bias.copy();plus[dimension]+=h;minus[dimension]-=h
            derivative=[(a-b)/(2*h) for a,b in zip(self.expected("npn",plus),self.expected("npn",minus))]
            for row in range(3):
                self.assertAlmostEqual(-sample["current"][branches[row]],derivative[row]*amplitude,
                    delta=max(1e-11,abs(derivative[row]*amplitude)*1e-6))

    def test_bjt_transient_quasi_static_kcl_at_every_actual_sample(self):
        c=self.create([
            {"type":"npn","nodes":["b","c","gnd"]},
            {"type":"vdc","nodes":["bias","gnd"],"params":{"v":.6}},
            {"type":"vac","nodes":["b","bias"],"params":{"vp":.03,"freq_hz":1000}},
            {"type":"vdc","nodes":["c","gnd"],"params":{"v":3}}])
        status,actual,steps,count,values=self.trace(c,1e-6,.002,every=10)
        self.assertEqual((status,actual,steps,count),(0,.002,2000,200))
        currents=[]
        for value in values:
            s=value["sample"];vb,vc,ve=s["voltage"][:3]
            expected=self.expected("npn",[vb,vc,ve])
            self.assertAlmostEqual(-s["current"][2],expected[1],delta=max(1e-10,abs(expected[1])*1e-6))
            currents.append(-s["current"][2])
        self.assertGreater(max(currents),2*min(currents))

if __name__=="__main__":unittest.main()
