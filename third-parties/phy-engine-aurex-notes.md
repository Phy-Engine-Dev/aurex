# Aurex native simulation additions

The patch in this directory targets Phy-Engine commit
`cdff6c17c3f7fb28154a8f3b776efad5cb0c8920`. It includes native observation,
camera-aware circuit visualization and generic primitive improvements. It does
not include task-specific circuit designs or prebuilt functional blocks.

## Spatial inspection and context

The default large-scene view uses a fixed 1280 x 960 canvas and preserves
native component positions. Full-scene fitting includes spatial outliers;
when those make the main layout tiny, inspection also returns a clearly
labelled primary-layout viewport plus the excluded references and coordinates.
That viewport is not a replacement for the full scene or pin-level evidence.
Focused inspection exposes pins and nearby connections without an unbounded
per-component sidebar. Physical occlusion remains possible.

Python tool results contain a compact type index and selected details. The
complete netlist and camera projection remain immutable artifacts, with
read_context document references in the agent journal. Published covers still
use the server-controlled full-scene camera, never the cropped primary view.

Local and clipped views reserve a lower-right global locator. It projects all
original component centers, including spatial outliers, and samples at most
400 center-to-center wires. Yellow marks components actually visible in the
main view. The locator uses three batched SVG paths, no second simulation or
rasterization, and no pin/value labels. It is an orientation aid, not evidence
of electrical connections. Full per-component locator coordinates and IDs are
retained as SVG metadata; the camera artifact records its scope and highlighted
IDs. Dense markers can overlap.
Publication covers remain fixed-angle, full-scene, and do not use this locator.

Original PLSAV Label values are carried separately from stable Identifier values
through native import, state rendering, editing and export. Missing labels are
not fabricated; auxiliary notQ gates do not inherit a public port label.
Focus resolution prefers exact Identifier, then short ref, then a unique Label.
An ambiguous Label reports candidate IDs instead of silently selecting a component.

Current portable patch SHA256:
`f4d43b8ff9ddff440b47797aa931c053328667ca1cf05e058ccb0aba0aca864a`.

The four public circuit tools default to strict boolean `with_image=false`.
The native renderer accepts a final options object `{"with_image":false}` to
produce data without SVG/PNG or Cairo work. Explicit true preserves camera and
locator rendering. Publication covers always force full-scene image generation.

`interface_only=true` reads actual digital input/output identifiers, labels,
positions, nodes and recorded logic state, with at most 64 ports per page.
It does not solve, expand internal gates, or render, and rejects simultaneous
`with_image=true`. Query selection is paginated after exact focus resolution
and deduplication; neighbor entries are separately marked. Digital transient
samples are read with `circuit_read_trace(component_ids=[...])`, preserving
actual timestamps and L/H/X/Z. Analog traces select `nodes` instead; the two
selectors cannot be combined. `circuit_read_stimulus` still reads only the
separately recorded stimulus sequence, not the TR samples.

Pure native digital simulation defaults to 4096 components, configurable with
`phy_engine.digital_component_limit` (1..16384). Both the tool and isolated
worker classify the actual native ABI model types; model-supplied flags cannot
bypass the unchanged 512-component analog/mixed-circuit protection. Rendering
native state documents supports 16384 elements without relaxing solver limits.
Large measurements remain complete in the state artifact; use
`circuit_read_stimulus` to page actual digital samples by component and step.
Successful propagation is not proof of the intended circuit's functionality.

## BJT model and compatibility

`BJT_NPN` and `BJT_PNP` retain C ABI codes 50/51 and the original five-property
layout: `Is,N,BetaF,Temp,Area`. They now use a quasi-static dual-junction
Ebers–Moll model instead of a forward-only current source. Positive terminal
current means current entering B, C or E. The three currents sum to zero.

For normalized NPN-polarity junction voltages, let `F=IsT*expm1(Vbe/(N*Ut))`
and `R=IsT*expm1(Vbc/(Nr*Ut))`. Then `Ib=F/BetaF+R/BetaR`,
`Ic=F-(1+1/BetaR)*R`, and `Ie=-(Ib+Ic)`. PNP uses sign-reversed voltages and
currents. The complete Jacobian is stamped into the native MNA solve for DC
and transient; AC uses the operating-point Jacobian.

Important compatibility detail: legacy PE `Is` meant the BE **base-current**
scale, unlike SPICE transport IS. Therefore `IsT=Is*BetaF*Area` preserves the
old forward-active scale. `BetaR=1` and `Nr=1` are independent defaults;
`Nr` does not inherit `N`. They are adjustable using the existing named scalar
setter. Attribute indices 0–4 remain unchanged, 5 is BetaR and 6 is Nr.
Read-only attributes 16/17/18 return actual constitutive terminal currents at
the solved voltages for DC/transient; 19/20 return Vbe/Vbc. These are not new
MNA branch indices and should not be read as AC phasors.

Numerical junction-voltage limiting aids Newton convergence. A smooth tangent
continuation above exponential argument 60 avoids overflow during extreme
trial steps. This is not a semiconductor breakdown model. The model does not
include Early effect, avalanche, parasitic resistances, charge storage,
junction capacitance, high-current beta rolloff or calibrated temperature
scaling of Is/Beta. Temperature changes thermal voltage only. Thus this is a
general low-frequency primitive, not a calibrated commercial transistor.

References: [ngspice BJT documentation](https://ngspice.sourceforge.io/docs/ngspice-manual.pdf)
and [upstream BJT equations](https://github.com/ngspice/ngspice/blob/master/src/spicelib/devices/bjt/bjtload.c).
The implementation is independently written, using the simplified transport
equations, not vendored ngspice source.

## PhysicsLab interoperability

NPN/PNP export to the real `Transistor` model, with PL B/C/E pins 0/1/2,
Properties `PNP=0/1` and `放大系数=BetaF`. This mapping is supported by the
[PhysicsLab SDK v2.0.6](https://github.com/SekaiArendelle/physicslab/blob/v2.0.6/physicsLab/circuit/elements/artificialCircuit.py).
Other PE physical parameters stay in the Aurex metadata sidecar; they have no
verified corresponding PL property. This preserves topology and authoring
parameters without claiming pointwise equivalence to the original app's
simulator. Diode/MOS remain native-only until their complete export semantics
are verified; missing mappings are errors, not silent omissions.

## Generic transient trace

`circuit_run_transient_bounded` and `circuit_run_transient_trace` use native
prepare/stamp/solve and actual `tr_duration`. Integer-indexed times avoid the
legacy accumulated-time overshoot: 0.5 s with 50 µs steps is exactly 10000
successful solves, with a shortened last step when necessary. This driver is
TR, not a hidden operating-point solve. Same-handle continuation preserves
capacitor history; trace prepares once and samples after completed solves.

The trace callback receives `(user, actual_time, completed_steps)`, allowing
`circuit_sample_complex` and model scalar getters to copy the real state.
It samples every requested number of steps and always includes the final
point, at most 201 points. There is no invented time-zero observation.
Callback nonzero stops immediately. Return codes are 0 success, 1 null
argument, 2 invalid parameter or budget, 3 native solve failure, 4 callback
stop, 5 unsupported analog/digital mixed node. Outputs report actual completed time, steps and sample count even when
the callback stops; a failed solve's partial internal state must be discarded.

The Aurex observation extension retains digital-propagation ABI version 1 and
the original trace/bounded entry points with one update per step. The additive
`circuit_transient_digital_propagation_configured_version()` reports version 1
for `circuit_run_transient_trace_configured` and
`circuit_run_transient_bounded_configured`. These accept
`digital_steps_per_tr_step` in 1..64, default 1. After every successful native
TR solve they execute exactly that many complete `digital_clk()` calls, then
sample if requested. Unsampled steps and no-callback bounded runs also execute
the configured count. Zero and out-of-range values are rejected before any
execution; integer-overflow checks protect the total count.

This is a count per completed solver step, not a digital time interval.
`tr_step` controls physical time. N advances tick-based state such as
TICK_DELAY and is not a convergence tolerance or an accuracy guarantee. The
1..64 limit guards per-step execution, not the overall agent task's duration.
The configured C ABI returns `actual_digital_steps`, the actual completed
digital-update calls in that invocation, alongside time and solver/sample
counts; it includes a completed step whose callback then stops. This total
does not prove Boolean convergence or circuit functionality.

Python removes the old implicit trailing tick on this path; explicit
`digital_clock_ticks` remain additional post-TR operations and cannot rewrite
the recorded trace. Old libraries and old state files lack verified per-step
provenance: their values are retained unchanged with a warning. Absence of the
configured capability must not be reported as successful configured execution.

Pure analog, pure digital, and disconnected analog/digital subgraphs are
supported. Nodes shared by analog and digital pins are rejected before any
step or callback because this driver does not implement same-step coupled
MNA/digital feedback. Configuring more digital ticks does not supply a coupled
mixed solver; it intentionally advances digital tick state without another
analog solve. This repair neither adds physical gate delays nor certifies an
original asynchronous CPU.

Existing PE capacitors use a trapezoidal Norton companion. New handles begin
with zero node voltages and zero capacitor history, not a solved DC operating
point. The first step has a half-step initialization effect for an ideal
voltage step. For a simple RC step, with `a=dt/(2RC)`, the discrete sequence is
`v_n/V=1-(1/(1+a))*((1-a)/(1+a))^(n-1)`.
Do not falsely describe this as backward Euler or as exact continuous-time
samples. Smaller steps reduce discretization/startup error. The current
tool does not expose arbitrary capacitor initial conditions.

## Reproducible verification

From the Aurex repo, after applying `phy-engine-aurex.patch` within the engine:

```sh
cmake -S third-parties/Phy-Engine/src -B .aurex/cache/phy-engine-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-21 -DCMAKE_C_COMPILER=clang-21 -DPHY_ENGINE_USE_LEVELDB=OFF
cmake --build .aurex/cache/phy-engine-build --target phyengine circuit_view verilog2plsav --parallel 1
PYTHONPATH=src AUREX_PHY_ENGINE_BUILD=.aurex/cache/phy-engine-build .venv/bin/python -m unittest discover -s tests -p test_native_observation.py -v
PYTHONPATH=src AUREX_PHY_ENGINE_BUILD=.aurex/cache/phy-engine-build .venv/bin/python -m unittest discover -s tests -p test_digital_tr_trace.py -v
```

The generic tests cover active/saturated/cutoff/reverse BJT bias, PNP symmetry,
KCL against independent voltage-source MNA currents, loaded saturation,
Jacobian sensitivity, AC linearization, configurable reverse parameters,
actual transient sampling, a diode forward-bias primitive, RC convergence,
same-handle continuation, endpoint accuracy and cancellation/sample budgets.
They contain no application-specific oscillator design.

The digital TR suite also compiles `tests/native/test_transient_digital.cpp`
against the selected native source (`AUREX_PHY_ENGINE_SOURCE` overrides the
default submodule directory). Its 20 native cases check fresh logic, input
changes, DFF edges and hold behavior, the old one-tick default, no-callback
propagation, X/Z preservation, mixed-node rejection and disconnected subgraphs.
Configured cases check N=3 tick-delay evolution and native totals, unsampled
steps, unchanged physical timestamps, N=1 equivalence to the legacy wrapper,
N=64 acceptance, 0/65/huge-value rejection without execution, overflow-safe
parameter rejection, configured mixed-node rejection and callback-stop counts.
These are generic primitive regressions, not a CPU instruction-correctness test.

The frozen count-enabled driver has 193 lines and git blob
`bb3014c6fd47697e07186f02f545cbe63397fb55`; its native C++ run passed 20/20,
and the preceding unchanged 10-case executable also passed against the new
library. Independent pure-analog regression passed 13/13 with 11 complete
canonical result groups byte-identical to the earlier library. Additional
configured N=3 RC/BJT runs preserved all analog records and timestamps, while
reporting 30000 updates for 10000 RC solves and 6000 for 2000 BJT solves.
