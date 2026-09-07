"""Official ModelID/pin-schema coverage; this is not a numerical-model certificate."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

# PhysicsLab SDK fa95b969, AST-audited circuit/elements catalog (87 ModelIDs).
EXPECTED_PINS = {
    "555 Timer": 8,
    "8bit Input": 8,
    "Accelerometer": 3,
    "Air Switch": 2,
    "Analog Joystick": 6,
    "And Gate": 3,
    "Attitude Sensor": 3,
    "Basic Capacitor": 2,
    "Basic Diode": 2,
    "Basic Inductor": 2,
    "Battery Source": 2,
    "Buzzer": 2,
    "Color Light-Emitting Diode": 4,
    "Comparator": 3,
    "Counter": 6,
    "Current Source": 2,
    "D Flipflop": 4,
    "DPDT Switch": 6,
    "Dual Light-Emitting Diode": 2,
    "Eight Bit Display": 8,
    "Electric Bell": 2,
    "Electric Fan": 2,
    "Electricity Meter": 4,
    "Full Adder": 5,
    "Full Subtractor": 5,
    "Fuse Component": 2,
    "Galvanometer": 3,
    "Gravity Sensor": 3,
    "Ground Component": 1,
    "Gyroscope": 3,
    "Half Adder": 4,
    "Half Subtractor": 4,
    "Imp Gate": 3,
    "Incandescent Lamp": 2,
    "JK Flipflop": 5,
    "Light-Emitting Diode": 2,
    "Linear Accelerometer": 3,
    "Logic Input": 1,
    "Logic Output": 1,
    "Magnetic Field Sensor": 3,
    "Microammeter": 3,
    "Multimeter": 2,
    "Multiplier": 8,
    "Musical Box": 2,
    "Mutual Inductor": 4,
    "N-MOSFET": 3,
    "Nand Gate": 3,
    "Nimp Gate": 3,
    "No Gate": 2,
    "Nor Gate": 3,
    "Operational Amplifier": 3,
    "Or Gate": 3,
    "P-MOSFET": 3,
    "Photodiode": 2,
    "Photoresistor": 2,
    "Proximity Sensor": 1,
    "Pulse Source": 2,
    "Push Switch": 2,
    "Random Generator": 6,
    "Real-T Flipflop": 4,
    "Rectifier": 4,
    "Relay Component": 5,
    "Resistance Box": 2,
    "Resistance Law": 8,
    "Resistor": 2,
    "SPDT Switch": 3,
    "Sawtooth Source": 2,
    "Schmitt Trigger": 2,
    "Simple Ammeter": 3,
    "Simple Instrument": 2,
    "Simple Switch": 2,
    "Simple Voltmeter": 3,
    "Sinewave Source": 2,
    "Slide Rheostat": 4,
    "Solenoid": 4,
    "Spark Gap": 2,
    "Square Source": 2,
    "Student Source": 4,
    "T Flipflop": 4,
    "Tapped Transformer": 5,
    "Tesla Coil": 2,
    "Transformer": 4,
    "Transistor": 3,
    "Triangle Source": 2,
    "Xnor Gate": 3,
    "Xor Gate": 3,
    "Yes Gate": 2
}

@unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'), 'native renderer required')
class ElectricalSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.binary = Path(os.environ['AUREX_PHY_ENGINE_BUILD']) / 'circuit_view'

    def source(self, types):
        elements = [{'ModelID': model, 'Identifier': 'component-' + str(i),
                     'Position': f'{i * .1},0,0', 'Rotation': '0,0,180',
                     'Properties': {}, 'Statistics': {}, 'IsBroken': False}
                    for i, model in enumerate(types)]
        return {'Experiment': {'Type': 0, 'StatusSave': json.dumps({'Elements': elements, 'Wires': []})}}

    def render(self, source):
        path = self.directory / 'source.sav'
        path.write_text(json.dumps(source), encoding='utf-8')
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        output = self.directory / 'netlist.json'
        image = self.directory / 'unrequested.svg'
        process = subprocess.run([str(self.binary), 'render', str(path), str(image), str(output),
                                  '0', '24', '-', 'auto', 'isometric', '[]', '', '{}',
                                  '{"with_image":false}'], capture_output=True, text=True, timeout=30)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertFalse(image.exists())
        return process, json.loads(output.read_text()) if process.returncode == 0 else None

    def test_all_87_official_schemas_include_unwired_terminals_without_sidecar(self):
        source = self.source(EXPECTED_PINS)
        process, result = self.render(source)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(result['components']), 87)
        self.assertFalse(any('Unknown pin schema' in warning for warning in result['warnings']))
        nodes = []
        for component in result['components']:
            self.assertTrue(component['pin_count_known'])
            expected = EXPECTED_PINS[component['type']]
            self.assertEqual([p['pin'] for p in component['pins']], list(range(expected)))
            self.assertNotIn('Aurex', component['raw_element'])
            nodes.extend(p['node'] for p in component['pins'])
        # Only the explicit ground terminal is grounded; schema does not wire devices.
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(result['wires'], [])

    def test_legacy_display_alias_and_unknown_schema_are_distinct(self):
        process, result = self.render(self.source(['8bit Display', 'Eight Bit Display', 'Future Unknown Device']))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([len(c['pins']) for c in result['components']], [8, 8, 0])
        self.assertFalse(result['components'][-1]['pin_count_known'])
        self.assertTrue(any('Unknown pin schema' in warning for warning in result['warnings']))

    def test_invalid_known_terminal_is_rejected_not_silently_added(self):
        source = self.source(['DPDT Switch', 'Ground Component'])
        status = json.loads(source['Experiment']['StatusSave'])
        status['Wires'] = [{'Source': 'component-0', 'SourcePin': 6,
                            'Target': 'component-1', 'TargetPin': 0, 'ColorName': '蓝色导线'}]
        source['Experiment']['StatusSave'] = json.dumps(status)
        process, _ = self.render(source)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('pin', process.stderr)

    def test_raw_broken_flag_and_other_saved_fields_are_preserved(self):
        source = self.source(['Resistor'])
        status = json.loads(source['Experiment']['StatusSave'])
        element = status['Elements'][0]
        element.update({'IsBroken': True, 'Label': 'R original', 'custom_future_field': {'original': 123}})
        source['Experiment']['StatusSave'] = json.dumps(status)
        process, result = self.render(source)
        self.assertEqual(process.returncode, 0, process.stderr)
        component = result['components'][0]
        self.assertIs(component['is_broken'], True)
        self.assertEqual(component['raw_element']['custom_future_field'], {'original': 123})

    def test_malformed_saved_position_and_rotation_are_rejected(self):
        for key in ('Position', 'Rotation'):
            with self.subTest(key=key):
                source = self.source(['Resistor'])
                status = json.loads(source['Experiment']['StatusSave'])
                status['Elements'][0][key] = 'not coordinates'
                source['Experiment']['StatusSave'] = json.dumps(status)
                process, _ = self.render(source)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('xyz', process.stderr)

if __name__ == '__main__':
    unittest.main()

