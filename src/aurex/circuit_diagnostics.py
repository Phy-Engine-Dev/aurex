"""Deterministic, bounded connectivity evidence; never infers functional PASS.

Directions are native catalog pin semantics. Analog attachments remain a boundary:
absence of a digital driver on an analog net is not evidence of an undriven net.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math

from .phy_engine.catalog import COMPONENTS

_OUTPUTS = {kind: tuple(meta['digital_output_pins']) for kind, meta in COMPONENTS.items()
            if 'digital_output_pins' in meta}
# Native output pins are the ordinary assertion points.  PhysicsLab Logic
# Output / 8bit Display devices are intentional probes, so their input pins are
# also valid observation points even though they never drive the circuit.
_OBSERVABLE_DIGITAL_PINS = {
    kind: (tuple(range(len(meta.get('pin_labels', ()))))
           if kind in {'digital_output', 'digital_output8'} else _OUTPUTS[kind])
    for kind, meta in COMPONENTS.items()
    if kind in _OUTPUTS and kind not in {'digital_input', 'digital_input8'}
}
_SEQUENTIAL = {'digital_dff', 'digital_tff', 'digital_t_bar_ff',
               'digital_jkff', 'digital_counter4', 'digital_random4'}
_UNDRIVEN_INPUT_DEFAULTS = {
    ('digital_counter4', 'enable'): {
        'unconnected_state': 'Z', 'effective_value': 'HIGH', 'meaning': 'ENABLED'},
    ('digital_random4', 'reset_n'): {
        'unconnected_state': 'Z', 'effective_value': 'HIGH', 'meaning': 'NOT_RESET'},
}


def is_blocking_finding(row):
    """Return whether a static finding invalidates a functional run.

    A floating clock cannot advance a sequential model.  Two unconditional
    outputs on one node are invalid.  Tri-state buses, however, require the
    solver's actual drive-resolution evidence and must not be rejected merely
    because more than one conditional output is connected.
    """
    return bool(
        isinstance(row, dict) and
        (row.get('kind') == 'undriven_clock' or
         (row.get('kind') == 'multiple_drivers' and
          row.get('drive_policy') == 'multiple_active_output_pins')))


def run_contract_targets(contract):
    """Exact observation roots declared by a validated run contract."""
    targets = []
    for assertion in contract.get('expected', []):
        target = assertion.get('node') or assertion.get('component')
        if isinstance(target, str) and target not in targets:
            targets.append(target)
    return targets
_SOURCE_FIELDS = ('source_ref', 'model_id', 'parent_identifier', 'decomposition_role',
                  'is_helper', 'primitive_pin_mapping')


def _original_pin(row, index):
    source = row.get('pl_source', {})
    mapping = source.get('primitive_pin_mapping')
    if isinstance(mapping, list) and index < len(mapping):
        return mapping[index]
    mapping = source.get('pin_mapping')
    if isinstance(mapping, dict):
        return mapping.get(str(index))
    return None  # Older IR without mapping is unknown, never silently identity.


def _node_name(node):
    # Same aliases as the native command builder. Canonicalize a local graph
    # copy so the source artifact/hash and saved spelling remain unchanged.
    return 'gnd' if not node or node.casefold() in ('gnd', 'ground', '0') else node


def compact_execution_evidence(value, *, depth=0):
    """Preserve status/count/timing scalars while bounding diagnostic examples."""
    if isinstance(value, dict):
        # Scalar execution facts outrank arrays (pending/hot nodes, conflicts).
        result = {key: item for key, item in value.items()
                  if not isinstance(item, (dict, list)) and
                  (not isinstance(item, str) or len(item) <= 256)}
        if depth < 3:
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    result[key] = compact_execution_evidence(item, depth=depth + 1)
        return result
    if isinstance(value, list):
        rows = [compact_execution_evidence(item, depth=depth + 1) for item in value[:4]]
        return {'total': len(value), 'shown': len(rows), 'omitted': len(value) - len(rows), 'rows': rows}
    return value[:256] if isinstance(value, str) else value


def _page(rows, offset, limit):
    shown = rows[offset:offset + limit]
    end = offset + len(shown)
    return {'total': len(rows), 'offset': offset, 'shown': len(shown),
            'omitted': len(rows) - len(shown), 'has_more': end < len(rows),
            'next_offset': end if end < len(rows) else None, 'rows': shown}


def _bounded_refs(rows, limit=6):
    return {'total': len(rows), 'shown': min(len(rows), limit),
            'omitted': max(0, len(rows) - limit), 'pins': rows[:limit]}


def _cycles(edges):
    """Iterative Kosaraju; large flat synthesized networks cannot overflow Python."""
    vertices = set(edges)
    vertices.update(node for values in edges.values() for node in values)
    visited, order = set(), []
    for start in sorted(vertices):
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((child, False) for child in sorted(edges.get(node, ()), reverse=True)
                             if child not in visited)
    reverse = defaultdict(set)
    for node, children in edges.items():
        for child in children:
            reverse[child].add(node)
    visited, groups = set(), []
    for start in reversed(order):
        if start in visited:
            continue
        group, stack = [], [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for child in reverse[node]:
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        if len(group) > 1 or start in edges.get(start, ()):
            groups.append(sorted(group))
    return sorted(groups)


class CircuitGraph:
    def __init__(self, spec):
        self.spec = spec
        self.components = {row['id']: {**row, 'nodes': [_node_name(node) for node in row['nodes']]}
                           for row in spec['components']}
        self.drivers, self.loads, self.analog = defaultdict(list), defaultdict(list), defaultdict(list)
        self.edges = defaultdict(set)
        self.unknown_types = Counter()
        for row in self.components.values():
            kind, cid = row['type'], row['id']
            labels = COMPONENTS.get(kind, {}).get('pin_labels', [])
            outputs = _OUTPUTS.get(kind)
            if outputs is None:
                if kind.startswith('digital_'):
                    self.unknown_types[kind] += 1
                for index, node in enumerate(row['nodes']):
                    self.analog[node].append({'component': cid, 'pin': index})
                continue
            for index, node in enumerate(row['nodes']):
                pin = {'component': cid, 'pin': index, 'label': labels[index]}
                source = row.get('pl_source', {})
                if source.get('source_ref'):
                    pin['source_ref'] = source['source_ref']
                if source:
                    pin['original_id'] = source.get('parent_identifier', cid)
                    pin['original_pin'] = _original_pin(row, index)
                (self.drivers if index in outputs else self.loads)[node].append(pin)
            if kind not in _SEQUENTIAL:
                inputs = [n for i, n in enumerate(row['nodes']) if i not in outputs]
                for index in outputs:
                    for node in inputs:
                        self.edges[node].add(row['nodes'][index])
        self.nodes = set(self.drivers) | set(self.loads) | set(self.analog)

    def targets(self, targets):
        nodes, missing = set(), []
        for target in targets:
            matches = [row for row in self.components.values()
                       if row['id'] == target or row.get('pl_source', {}).get('source_ref') == target]
            if _node_name(target) in self.nodes:
                nodes.add(_node_name(target))
            elif matches:
                for row in matches:
                    outputs = _OUTPUTS.get(row['type'], ())
                    nodes.update(row['nodes'][i] for i in outputs)
                    if not outputs:
                        nodes.update(row['nodes'])
            else:
                missing.append(target)
        return nodes, missing

    def slice(self, targets, depth=8):
        seeds, missing = self.targets(targets)
        pending = deque((node, 0) for node in sorted(seeds))
        nodes, components, boundaries, frontier = set(), set(), [], set()
        while pending:
            node, level = pending.popleft()
            if node in nodes:
                continue
            nodes.add(node)
            if self.analog.get(node):
                boundaries.append({'node': node, 'kind': 'analog_network',
                                   'attachments': _bounded_refs(self.analog[node]),
                                   'retained_in_full_run': ['feedback', 'supply', 'ground', 'load'],
                                   'not_an_isolated_simulatable_subcircuit': True})
            for pin in self.drivers.get(node, ()):
                cid = pin['component']
                components.add(cid)
                row = self.components[cid]
                labels = COMPONENTS[row['type']]['pin_labels']
                if row['type'] in _SEQUENTIAL:
                    controls = {labels[i]: n for i, n in enumerate(row['nodes'])
                                if labels[i] in ('clk', 'reset_n', 'enable')}
                    boundaries.append({'component': cid, 'kind': 'sequential',
                                       'controls': controls,
                                       'data_nodes': [n for i, n in enumerate(row['nodes'])
                                                      if i not in _OUTPUTS[row['type']]
                                                      and labels[i] not in controls]})
                    parents = controls.values()
                else:
                    parents = [n for i, n in enumerate(row['nodes']) if i not in _OUTPUTS[row['type']]]
                for parent in parents:
                    if parent in nodes:
                        continue
                    if level < depth:
                        pending.append((parent, level + 1))
                    else:
                        frontier.add(parent)
        return nodes, components, boundaries, frontier - nodes, missing

    def findings(self, relevant=None):
        findings = []
        for node in sorted(self.nodes):
            if relevant is not None and node not in relevant:
                continue
            drivers, loads, analog = self.drivers[node], self.loads[node], self.analog[node]
            clocks = [pin for pin in loads if pin['label'] == 'clk']
            if not drivers and loads and not analog and node != 'gnd':
                default_loads = []
                for pin in loads:
                    kind = self.components[pin['component']]['type']
                    default = _UNDRIVEN_INPUT_DEFAULTS.get((kind, pin['label']))
                    if default:
                        default_loads.append({**pin, 'model_default': default})
                if len(default_loads) == len(loads):
                    findings.append({'kind': 'undriven_with_model_default',
                                     'severity': 'info', 'node': node, 'status': 'observed',
                                     'evidence': {'drivers': _bounded_refs(drivers),
                                                  'loads': _bounded_refs(default_loads),
                                                  'all_loads_have_model_defaults': True},
                                     'functional_effect': 'model-defined defaults apply; not a structural blocker'})
                else:
                    findings.append({'kind': 'undriven_clock' if clocks else 'undriven_net',
                                     'node': node, 'status': 'observed',
                                     'evidence': {'drivers': _bounded_refs(drivers),
                                                  'loads': _bounded_refs(loads, 3)},
                                     'functional_effect': 'not established'})
            if len(drivers) > 1:
                conditional = any(self.components[pin['component']]['type'] == 'digital_tri'
                                  for pin in drivers)
                findings.append({'kind': 'multiple_drivers', 'node': node, 'status': 'observed',
                                 'drive_policy': 'requires_resolution' if conditional else 'multiple_active_output_pins',
                                 'evidence': {'drivers': _bounded_refs(drivers),
                                              'loads': _bounded_refs(loads, 3)},
                                 'functional_effect': 'requires solver drive-resolution/settle evidence'})
        for group in _cycles(self.edges):
            if relevant is not None and not relevant.intersection(group):
                continue
            findings.append({'kind': 'combinational_cycle', 'status': 'derived',
                             'node_count': len(group), 'nodes': group[:8],
                             'nodes_omitted': max(0, len(group) - 8),
                             'functional_effect': 'topological cycle; oscillation not established'})
        order = {'multiple_drivers': 0, 'undriven_clock': 1, 'combinational_cycle': 2,
                 'undriven_net': 3, 'undriven_with_model_default': 4}
        return sorted(findings, key=lambda row: (order[row['kind']], row.get('node', '')))


def circuit_diagnostics(spec, *, mode='preflight', targets=(), offset=0, limit=8, depth=8,
                        measurements=None, max_characters=7000):
    graph = CircuitGraph(spec)
    relevant, exact_nodes = None, set()
    result = {'schema': 'aurex.circuit-diagnostics.v1', 'mode': mode,
              'basis': 'native pin direction and exact connectivity; no functional pass inferred',
              'source_sha256': spec.get('import_scope', {}).get('original_source_sha256'),
              'native_spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
              'statistics': {'components': len(graph.components), 'nodes': len(graph.nodes),
                             'unknown_digital_types': dict(graph.unknown_types)}}
    if targets:
        exact_nodes, _ = graph.targets(targets)
        relevant, components, boundaries, frontier, missing = graph.slice(targets, depth)
        result['target_scope'] = {'targets': list(targets), 'unresolved_targets': missing,
                                  'node_count': len(relevant), 'component_count': len(components),
                                  'max_combinational_depth': depth, 'frontier_count': len(frontier),
                                  'frontier': sorted(frontier)[:8],
                                  'sequential_data_not_expanded': True,
                                  'analog_network_not_expanded': True}
        if mode == 'slice':
            rows = []
            for cid in sorted(components):
                row = graph.components[cid]
                rows.append({'id': cid, 'type': row['type'], 'nodes': row['nodes'],
                             'pin_labels': COMPONENTS[row['type']]['pin_labels'],
                             'pin_directions': ['output' if index in _OUTPUTS[row['type']] else 'input'
                                                for index in range(len(row['nodes']))],
                             'sequential': row['type'] in _SEQUENTIAL,
                             'source': {k: row.get('pl_source', {})[k] for k in _SOURCE_FIELDS
                                        if k in row.get('pl_source', {})}})
            # Repeated importer prose is provenance, not per-gate evidence.
            # Identify the assumption set once without duplicating it in each
            # row or silently promoting approximations to exact equivalence.
            assumptions = sorted({str(item) for cid in components
                                  for item in graph.components[cid].get('pl_source', {}).get('assumptions', [])})
            if assumptions:
                result['import_assumptions'] = {'unique_count': len(assumptions),
                    'sha256': hashlib.sha256(json.dumps(assumptions, ensure_ascii=False).encode()).hexdigest(),
                    'code': 'IMPORTED_MODEL_ASSUMPTIONS_PRESENT', 'original_app_equivalence_proven': False}
            rows.sort(key=lambda row: (not bool(set(row['nodes']) & exact_nodes), row['id']))
            result['slice'] = _page(rows, offset, limit)
            result['boundaries'] = _page(boundaries, 0, min(limit, 4))
    global_findings = graph.findings()
    findings = graph.findings(relevant) if relevant is not None else global_findings
    findings.sort(key=lambda row: (row.get('node') not in exact_nodes,
                                   {'multiple_drivers': 0, 'undriven_clock': 1, 'combinational_cycle': 2,
                                    'undriven_net': 3, 'undriven_with_model_default': 4}[row['kind']],
                                   row.get('node', '')))
    result['finding_counts'] = dict(Counter(row['kind'] for row in findings))
    result['blocking_finding_counts'] = dict(Counter(
        ('multiple_active_output_pins' if row.get('kind') == 'multiple_drivers'
         else row['kind']) for row in findings if is_blocking_finding(row)))
    if relevant is not None:
        result['target_relevance_evaluated'] = True
        result['observation_targets'] = list(targets)
        result['global_finding_counts'] = dict(Counter(
            row['kind'] for row in global_findings))
        result['finding_scope'] = (
            'blocking_finding_counts covers only the backward observation cone; '
            'global_finding_counts is informational and does not block this contract')
        if mode == 'slice' and result['blocking_finding_counts']:
            result['verdict'] = 'INCONCLUSIVE'
            result['failure_class'] = 'invalid_netlist_or_drive_contract'
            result['reason'] = 'A blocking drive/clock finding intersects the requested observation cone.'
            result['execution'] = {'started': False}
    else:
        result['target_relevance_evaluated'] = False
        result['finding_scope'] = (
            'whole-design preflight only; no finding is yet proven relevant to a user-requested observation')
        if result['blocking_finding_counts']:
            result['next_action'] = {
                'tool': 'circuit_diagnose', 'mode': 'slice',
                'targets': 'one or more user-requested Logic Output/meter refs or native IDs; never the finding node',
                'simulation_before_target_slice': False,
            }
    result['findings'] = _page(findings, offset, limit)
    if mode == 'diagnose':
        evidence = measurements if isinstance(measurements, dict) else {}
        result['execution'] = {key: compact_execution_evidence(evidence[key]) for key in
                               ('execution', 'settle', 'coverage', 'digital_settle', 'failure') if key in evidence}
        transient = evidence.get('transient', {})
        result['timeline'] = {key: compact_execution_evidence(transient[key]) for key in
                              ('actual_stop_s', 'requested_stop_s', 'completed_steps',
                               'sample_count', 'sample_every', 'digital_propagation') if key in transient}
        result['conclusion'] = 'INCONCLUSIVE'
        result['reason'] = 'Structural findings are evidence, not a circuit specification or functional verification.'
    # The producer bounds the same payload for both Web and model; no implicit
    # giant raw netlist is attached. Paging and projection omission are distinct.
    result['presentation'] = {'max_characters': max_characters, 'projection_omitted': 0}
    def size():
        return len(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    while size() > max_characters:
        pages = [result[k] for k in ('findings', 'slice', 'boundaries')
                 if isinstance(result.get(k), dict) and result[k].get('rows')]
        if not pages:
            break
        # Preserve exact requested-node faults ahead of slice decoration and
        # unrelated upstream findings. A target query must answer the target.
        page = next((result[key] for key in ('boundaries', 'slice')
                     if isinstance(result.get(key), dict) and result[key].get('rows')), None)
        if page is None:
            page = result['findings']
        page['rows'].pop()
        page['shown'] -= 1
        page['omitted'] += 1
        # Source has_more refers to the originally requested source page.
        # next_offset must nevertheless allow recovery of every omitted row.
        page['projection_omitted'] = page.get('projection_omitted', 0) + 1
        page['next_offset'] = page['offset'] + page['shown']
        result['presentation']['projection_omitted'] += 1
    return result


def normalize_run_contract(spec, contract):
    """Validate declared expectations and expand clocks/buses to explicit frames.

    Inputs in a bus are explicitly MSB first. Clock units are stimulus frames,
    not propagation iterations or physical gate delay.
    """
    if not isinstance(contract, dict):
        raise ValueError('contract must be an object')
    source = contract.get('expected_source')
    if not isinstance(source, str) or not 1 <= len(source) <= 512:
        raise ValueError('contract.expected_source must name the independent specification/reference (1..512 characters)')
    expected = contract.get('expected')
    if not isinstance(expected, list) or not 1 <= len(expected) <= 32:
        raise ValueError('contract.expected must contain 1..32 independently specified assertions')
    components = {row['id']: row for row in spec['components']}
    nodes = {node for row in spec['components'] for node in row['nodes']}
    for row in expected:
        if not isinstance(row, dict):
            raise ValueError('expected assertions must be objects')
        if set(row) - {'component', 'pin', 'node', 'equals', 'tolerance', 'frame', 'time_s'}:
            raise ValueError('Unknown expected assertion fields; use component/pin or node, equals, tolerance, frame/time_s')
        if ('frame' in row) == ('time_s' in row):
            raise ValueError('Each assertion must select exactly one frame or exact recorded time_s')
        if 'frame' in row and (type(row['frame']) is not int or not 0 <= row['frame'] < 128):
            raise ValueError('frame is a zero-based stimulus frame in [0,127]')
        if 'time_s' in row and (type(row['time_s']) not in (int, float) or not math.isfinite(row['time_s']) or row['time_s'] < 0):
            raise ValueError('time_s must be finite and nonnegative')
        if 'node' in row:
            if row['node'] not in nodes or 'time_s' not in row:
                raise ValueError('Analog assertion requires a known node and exact time_s')
            for field in ('equals', 'tolerance'):
                if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]):
                    raise ValueError('Analog assertions require finite equals and tolerance')
            if row['tolerance'] < 0:
                raise ValueError('tolerance must be nonnegative')
        else:
            cid, pin = row.get('component'), row.get('pin')
            if cid not in components:
                raise ValueError('Digital assertion requires an exact native digital component ID')
            if components[cid]['type'] not in _OBSERVABLE_DIGITAL_PINS:
                raise ValueError('Digital functional assertions must observe a native output pin or Logic Output/Display probe pin, not an input/stimulus pin')
            if type(pin) is not int or not 0 <= pin < len(components[cid]['nodes']):
                raise ValueError('Digital assertion pin must be a native pin index')
            if pin not in _OBSERVABLE_DIGITAL_PINS[components[cid]['type']]:
                raise ValueError('Digital functional assertions must observe a native output pin or Logic Output/Display probe pin, not an input/stimulus pin')
            if type(row.get('equals')) is not int or row['equals'] not in (0, 1):
                raise ValueError('Expected functional digital value must be 0 or 1; X/Z never count as verification')
    clocks, buses = contract.get('clocks', []), contract.get('buses', [])
    if not isinstance(clocks, list) or len(clocks) > 4 or not isinstance(buses, list) or len(buses) > 8:
        raise ValueError('contract supports at most 4 clocks and 8 buses')
    table = contract.get('stimulus_table')
    if not clocks and not buses:
        return table
    if table is not None:
        raise ValueError('Use generated clocks/buses or explicit stimulus_table, not both')
    frames = contract.get('frame_count')
    if type(frames) is not int or not 1 <= frames <= 128:
        raise ValueError('Generated stimuli require frame_count in [1,128]')
    inputs, columns = [], []
    def add(cid, values):
        if cid not in components or components[cid]['type'] != 'digital_input' or cid in inputs:
            raise ValueError('Clock/bus columns must be distinct exact digital_input IDs')
        inputs.append(cid)
        columns.append(values)
    for clock in clocks:
        if not isinstance(clock, dict):
            raise ValueError('clock must be an object')
        period, high, phase = clock.get('period_frames'), clock.get('high_frames'), clock.get('phase_frames', 0)
        if (type(period) is not int or not 2 <= period <= 128 or type(high) is not int
                or not 1 <= high < period or type(phase) is not int or not 0 <= phase < period):
            raise ValueError('clock requires period_frames 2..128, high_frames 1..period-1, phase_frames 0..period-1')
        add(clock.get('input'), [int((index - phase) % period < high) for index in range(frames)])
    for bus in buses:
        if not isinstance(bus, dict):
            raise ValueError('bus must be an object')
        ids, values = bus.get('inputs_msb_first'), bus.get('values')
        if not isinstance(ids, list) or not 1 <= len(ids) <= 64 or not isinstance(values, list) or len(values) != frames:
            raise ValueError('bus requires 1..64 inputs_msb_first and exactly frame_count integer values')
        if any(type(value) is not int or not 0 <= value < 2 ** len(ids) for value in values):
            raise ValueError('bus values must fit the declared unsigned bus width')
        for index, cid in enumerate(ids):
            add(cid, [(value >> (len(ids) - index - 1)) & 1 for value in values])
    return {'inputs': inputs, 'vectors': [list(values) for values in zip(*columns)]}


def evaluate_run_contract(spec, contract, measurements, *, points=(), stimulus=()):
    """Evaluate only supplied expected values against actual recorded samples."""
    graph = CircuitGraph(spec)
    targets = run_contract_targets(contract)
    relevant = graph.slice(targets, depth=max(8, len(graph.nodes)))[0] if targets else None
    preflight = graph.findings(relevant)
    blockers = [row for row in preflight if is_blocking_finding(row)]
    transient = measurements.get('transient', {})
    settle = measurements.get('digital_settle') or transient.get('digital_propagation', {}).get('settle') or {}
    digital = any(row['type'].startswith('digital_') for row in spec['components'])
    healthy = (not blockers and measurements.get('execution_status') == 'completed'
               and (not digital or settle.get('settled') is True))
    if measurements.get('execution_status') == 'failed' or measurements.get('waveform_valid') is False:
        healthy = False
    requested, actual = transient.get('requested_stop_s'), transient.get('actual_stop_s')
    if isinstance(requested, (int, float)) and (not isinstance(actual, (int, float)) or actual + 1e-12 < requested):
        healthy = False
    results = []
    for index, assertion in enumerate(contract['expected']):
        value, reason = None, None
        if 'frame' in assertion:
            frame = assertion['frame']
            if frame < len(stimulus):
                row = stimulus[frame]
                pins = row.get('digital', {}).get(assertion['component'], [])
                if assertion['pin'] < len(pins):
                    value = pins[assertion['pin']]
                if digital and row.get('digital_settled') is not True:
                    reason = 'frame settle evidence absent'
            else:
                reason = 'stimulus frame not recorded'
        else:
            wanted = assertion['time_s']
            sample = next((point for point in points if isinstance(point.get('time_s'), (int, float))
                           and math.isclose(point['time_s'], wanted, rel_tol=1e-12, abs_tol=1e-15)), None)
            if sample is None:
                reason = 'exact time not sampled; no interpolation'
            elif 'node' in assertion:
                values = [voltage for row in sample.get('components', [])
                          for node, voltage in zip(row.get('nodes', []), row.get('voltage', []))
                          if node == assertion['node']]
                if values and all(math.isclose(values[0], item, rel_tol=1e-9, abs_tol=1e-12) for item in values):
                    value = values[0]
                elif values:
                    reason = 'contradictory saved node measurements'
            else:
                row = next((row for row in sample.get('components', []) if row.get('id') == assertion['component']), {})
                pins = row.get('digital', [])
                if assertion['pin'] < len(pins):
                    value = pins[assertion['pin']]
                if sample.get('digital_settled') is not True:
                    reason = 'sample settle evidence absent'
        if value is None:
            verdict, reason = 'INCONCLUSIVE', reason or 'measurement absent'
        elif reason:
            verdict = 'INCONCLUSIVE'
        elif 'node' not in assertion and value in (2, 3):
            verdict, reason = 'INCONCLUSIVE', 'observed X/Z'
        elif not healthy:
            verdict, reason = 'INCONCLUSIVE', 'simulation health gate not satisfied'
        else:
            verdict = 'PASS' if abs(value - assertion['equals']) <= assertion.get('tolerance', 0) else 'FAIL'
        result = {'index': index, 'expected': assertion, 'observed': value, 'verdict': verdict,
                  'status': 'observed', 'evidence': {'source': 'recorded solver sample'}}
        if reason:
            result['reason'] = reason
        results.append(result)
    counts = Counter(row['verdict'] for row in results)
    verdict = ('INCONCLUSIVE' if not healthy else 'FAIL' if counts['FAIL'] else
               'INCONCLUSIVE' if counts['INCONCLUSIVE'] else 'PASS')
    failure_class = ('invalid_netlist_or_drive_contract' if blockers else
                     'execution_not_verified' if not healthy else
                     'specification_assertion_failed' if verdict == 'FAIL' else
                     'insufficient_observation' if verdict == 'INCONCLUSIVE' else None)
    return {'verdict': verdict, 'failure_class': failure_class, 'expected_source': contract['expected_source'],
            'contract_sha256': hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'execution': {'health_gate_passed': healthy, 'settle': compact_execution_evidence(settle)},
            'coverage': {'assertions': len(results), 'pass': counts['PASS'], 'fail': counts['FAIL'],
                         'inconclusive': counts['INCONCLUSIVE'], 'recorded_frames': len(stimulus),
                         'recorded_time_samples': len(points),
                         'scope': 'Only declared assertions at recorded times/frames; not exhaustive design verification'},
            'assertions': results,
            'time_semantics': {'stimulus_frame_unit': 'logical stimulus frame; not physical time',
                               'digital_propagation_iterations_are_not_delays': True,
                               'sampling': 'after explicit stimulus frame settles; physical time_s only from TR recorded timestamps'},
            'preflight_blockers': _page(blockers, 0, 2)}
