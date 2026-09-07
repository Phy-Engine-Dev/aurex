# PhysicsLab → Aurex → Phy-Engine compatibility contract

This contract covers the serialized electrical behavior that Aurex can recover
from a public PhysicsLab `.sav`. It does not claim bit-for-bit equivalence with
the closed PhysicsLab solver. Every approximation is attached to the generated
native primitive as `pl_source` provenance; archived `Statistics` are never
presented as a fresh solve.

## Audited source inventory

- Reference SDK: `SekaiArendelle/physicslab` commit
  `fa95b96910dd0fd4e09cf27e24cefaf9b91798ad`.
- Fixture: `tests/data/physicslab-all-circuit-elements-fa95b96.sav`.
- Fixture SHA-256:
  `d086f2a43c9cd0fcf1197d80dbed0c6e7f41c89c6de605c4c041e506cf58389d`.
- Inventory: 87 distinct public `ModelID` values, 91 saved instances. Ground is
  normalized to the native reference; the other 90 instances expand to native
  PE primitives without mutating the source file.

The covered catalog is: 555 Timer, 8bit Display, 8bit Input, Accelerometer,
Air Switch, Analog Joystick, And Gate, Attitude Sensor, Basic Capacitor, Basic
Diode, Basic Inductor, Battery Source, Buzzer, Color Light-Emitting Diode,
Comparator, Counter, Current Source, D Flipflop, DPDT Switch, Dual
Light-Emitting Diode, Electric Bell, Electric Fan, Electricity Meter, Full
Adder, Full Subtractor, Fuse Component, Galvanometer, Gravity Sensor, Ground
Component, Gyroscope, Half Adder, Half Subtractor, Imp Gate, Incandescent Lamp,
JK Flipflop, Light-Emitting Diode, Linear Accelerometer, Logic Input, Logic
Output, Magnetic Field Sensor, Microammeter, Multimeter, Multiplier, Musical
Box, Mutual Inductor, N-MOSFET, Nand Gate, Nimp Gate, No Gate, Nor Gate,
Operational Amplifier, Or Gate, P-MOSFET, Photodiode, Photoresistor, Proximity
Sensor, Pulse Source, Push Switch, Random Generator, Real-T Flipflop,
Rectifier, Relay Component, Resistance Box, Resistance Law, Resistor, SPDT
Switch, Sawtooth Source, Schmitt Trigger, Simple Ammeter, Simple Instrument,
Simple Switch, Simple Voltmeter, Sinewave Source, Slide Rheostat, Solenoid,
Spark Gap, Square Source, Student Source, T Flipflop, Tapped Transformer, Tesla
Coil, Transformer, Transistor, Triangle Source, Xnor Gate, Xor Gate and Yes
Gate.

## End-to-end guarantees

`circuit_analyze(path=<sav>)` performs this exact chain:

1. The native C++ reader parses all saved components, terminal numbers,
   topology, properties, positions, rotations and the camera.
2. Audited importers create native PE primitives and explicit helper devices.
   Series/shunt internal resistance uses the topology appropriate to the
   original device.
3. PE performs DC/OP/AC/TR or coupled digital propagation. Returned voltage,
   current, digital and model-state fields are sampled from that live native
   circuit.
4. The full spec, source scene, measurements and provenance are persisted in
   an immutable PE-state artifact. Pagination/summary never reruns the solver
   or substitutes archived PhysicsLab statistics.

The all-elements regression checks every original parent ID, every raw
property dictionary, every generated primitive ID, finite DC state, a
three-step TR trace, two digital propagations per physical step and byte-for-
byte preservation of the input `.sav`.

## Ratings, failure and interaction behavior

- Positive saved maximum current, terminal-voltage and instantaneous-power
  ratings are implemented by `rated_protection`. A limit violation records the
  trip current/voltage/power and irreversibly opens the audited terminal in the
  same nonlinear solve. Zero or absent ratings disable only that criterion.
- A saved `IsBroken=true` isolates every distinct external node of every
  imported electrical device. Pins that were tied in the save remain tied
  behind one open guard. The all-elements broken regression exercises all 90
  non-ground instances.
- The largest finite float32 `最大功率` value is PhysicsLab's unlimited-power
  serialization sentinel. It is preserved on export but does not create a
  finite native trip guard.
- Logic-gate `最大电流` is enforced when that output feeds an analog MNA load.
  A pure digital net has no current domain, so the same field remains metadata.
- Buttons, SPST/SPDT/DPDT switches, rheostats, source voltage and digital inputs
  can be changed on exact TR grid points. Changes are applied immediately
  before the named physical solve and are recorded in the trace.

The regression suite includes safe and tripped paths for current, voltage and
power; lamp overvoltage/power; fuse overcurrent; mixed gate-output overcurrent;
and pre-broken multi-terminal devices.

## Explicit engineering boundaries

These are bounded models rather than silent omissions:

- Environmental sensors retain output impedance and emit an explicit zero-
  stimulus source when the save contains no environment sample. No archived
  reading is invented as current input.
- LED light, buzzer/speaker sound, motor airflow, thermal accumulation,
  electrode erosion and Tesla-coil RF radiation are not serialized electrical
  states. Their electrical terminals and available lumped parameters are
  solved; those unidentifiable domains are not fabricated.
- Non-unity two-winding coupling is solved with explicit coupled inductors and
  saved winding losses. A tapped transformer save has no complete three-
  winding inductance matrix, so its non-unity-coupling case remains an explicit
  ideal-ratio approximation.
- Random Generator saves do not contain hidden PRNG state. Aurex assigns a
  deterministic, provenance-recorded nonzero surrogate seed so reset and
  transitions can be tested repeatably without claiming the original sequence.
- The behavioral 555, clamped op amp, relay, motor and spark gap implement
  documented terminal-level behavior; undocumented semiconductor internals,
  contact bounce and thermal histories are not inferred.
- Verilog high-impedance/Z semantics are rejected in strict PhysicsLab export.
  This prevents optimization from silently replacing tri-state behavior with
  ordinary Boolean gates.

## Reproduction

From the Aurex repository root:

```sh
export PYTHONPATH="$PWD/src"
export AUREX_PHY_ENGINE_BUILD="$PWD/.aurex/cache/phy-engine-build"
.venv/bin/python -m unittest -v tests.test_physicslab_pe_coverage
.venv/bin/python -m unittest discover -v tests
```

The first command is the catalog/damage/dynamic-device end-to-end gate. The
second is the complete Aurex offline suite. Network-dependent discovery and
optional private large-scene fixtures are marked skips and are not counted as
electrical compatibility evidence.
