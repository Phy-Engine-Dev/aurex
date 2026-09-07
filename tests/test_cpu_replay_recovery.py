from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig
from aurex.tools.circuits import _saved_pin_labels, circuit_analyze, circuit_inspect
from aurex.tools.registry import ToolRuntime, ToolError
from aurex.task_reply import FINAL_SYSTEM
from aurex.session_agent import SYSTEM


class ReplayRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.rt=ToolRuntime('replay-recovery','zh','',AurexConfig(),self.folder.name)

    def test_final_reviewer_allows_deliberate_rechecks_without_mechanical_loops(self):
        for expected in ('recorded_tool_evidence_index','允许原样重做','要求重复时说明','精确query/focus'):
            self.assertIn(expected,FINAL_SYSTEM)

    def test_agent_does_not_diagnose_isolated_x_as_cpu_failure(self):
        for expected in ('孤立或未连接引脚', 'X 只证明', '不是元件或 CPU 失效',
                         '连通性确认'):
            self.assertIn(expected, SYSTEM)

    def test_saved_digital_pin_semantics_come_from_the_import_mapping(self):
        self.assertEqual(_saved_pin_labels({'type': 'D Flipflop'}),
                         {0: 'q', 1: 'q_bar', 2: 'd', 3: 'clk'})
        self.assertEqual(_saved_pin_labels({'type': 'And Gate'}),
                         {0: 'a', 1: 'b', 2: 'out'})

    def test_wide_stimulus_table_is_constrained_to_sparse_agent_format(self):
        from aurex.tools.circuits import register_circuit_tools
        from aurex.tools.registry import ToolRegistry
        registry = ToolRegistry()
        register_circuit_tools(registry)
        analyze = next(tool for tool in registry.list() if tool.name == 'circuit_analyze')
        inputs = analyze.parameters['properties']['stimulus_table']['properties']['inputs']
        self.assertEqual(inputs['maxItems'], 15)
        self.assertIn('sparse stimulus', inputs['description'])

    def test_shape_error_reports_actual_count_and_safe_sparse_recovery(self):
        spec={'components':[{'id':'IN'+str(i),'type':'digital_input','nodes':['n'+str(i)],'params':{'state':0}} for i in range(45)]}
        with patch('aurex.tools.circuits.pe_simulate') as solve:
            with self.assertRaises(ToolError) as error:
                circuit_analyze(self.rt,{'spec':spec,'stimulus_table':{'inputs':['IN'+str(i) for i in range(45)],'vectors':[[0]*46]}})
        solve.assert_not_called()
        for expected in ('exactly 45','got 46','No simulation was executed','omit stimulus_table','not an implicit zero','first input'):
            self.assertIn(expected,str(error.exception))

    def test_isolated_stimulus_is_rejected_before_simulation(self):
        spec={'components':[
            {'id':'unused','type':'digital_input','nodes':['isolated'],'params':{'state':0}},
            {'id':'out','type':'digital_output','nodes':['other'],'params':{}}]}
        with patch('aurex.tools.circuits.pe_simulate') as solve:
            with self.assertRaises(ToolError) as error:
                circuit_analyze(self.rt,{'spec':spec,'stimulus_table':{'inputs':['unused'],'vectors':[[0],[1]]}})
        solve.assert_not_called()
        for expected in ('isolated digital_input','cannot affect another component','connected_to_other_components=true','No simulation was executed'):
            self.assertIn(expected,str(error.exception))

    @unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'),'native renderer required')
    def test_displayed_reference_can_be_queried_without_substring_matches(self):
        from aurex.tools.circuits import circuit_create
        build=Path(os.environ['AUREX_PHY_ENGINE_BUILD'])
        cfg=replace(self.rt.config,phy_engine=replace(self.rt.config.phy_engine,auto_build=False,
            cmake_build_dir=str(build),verilog2plsav_path=str(build/'verilog2plsav'),phyengine_lib_path=str(build/'libphyengine.so')))
        rt=replace(self.rt,config=cfg)
        out=circuit_create(rt,{'spec':{'components':[{'id':'p'+str(i),'type':'digital_input','nodes':['n'+str(i)],'params':{'state':0}} for i in range(12)]}})
        selected=circuit_inspect(rt,{'path':out['circuit_path'],'query':'C1','limit':1})
        ids=[c['id'] for c in selected['netlist']['components'] if c.get('selection_role')=='primary']
        self.assertEqual(ids,['p0'])

    @unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'),'native renderer required')
    def test_exact_node_query_returns_connected_components_without_netlist_scan(self):
        from aurex.tools.circuits import circuit_create
        build=Path(os.environ['AUREX_PHY_ENGINE_BUILD'])
        cfg=replace(self.rt.config,phy_engine=replace(self.rt.config.phy_engine,auto_build=False,
            cmake_build_dir=str(build),verilog2plsav_path=str(build/'verilog2plsav'),phyengine_lib_path=str(build/'libphyengine.so')))
        rt=replace(self.rt,config=cfg)
        out=circuit_create(rt,{'spec':{'components':[
            {'id':'in','type':'digital_input','nodes':['N0'],'params':{'state':0}},
            {'id':'gate','type':'digital_not','nodes':['N0','N1'],'params':{}},
            {'id':'out','type':'digital_output','nodes':['N1'],'params':{}}]}})
        interface=circuit_inspect(rt,{'path':out['circuit_path'],'interface_only':True})
        by_id={port['id']:port for port in interface['ports']}
        self.assertEqual(by_id['in']['node_connection_count'],2)
        self.assertTrue(by_id['in']['connected_to_other_components'])
        selected=circuit_inspect(rt,{'path':out['circuit_path'],'query':'N1','limit':8})
        self.assertEqual(selected['node_query']['match_count'],2)
        self.assertEqual(selected['node_query']['node'],'N1')
        primary={c['id'] for c in selected['netlist']['components'] if c.get('selection_role')=='primary'}
        # Native renderer node names are canonicalized during circuit creation;
        # N1 is the exact saved node joining input to the gate in that artifact.
        self.assertEqual(primary,{'in','gate'})
        self.assertIsNone(selected['node_query']['next_offset'])
        with self.assertRaisesRegex(ToolError,'No components connect'):
            circuit_inspect(rt,{'path':out['circuit_path'],'query':'N999','limit':8})

    @unittest.skipUnless(os.environ.get('AUREX_PHY_ENGINE_BUILD'),'native renderer required')
    def test_exact_node_pagination_never_skips_when_requested_limit_exceeds_renderer_page(self):
        from aurex.tools.circuits import circuit_create
        build=Path(os.environ['AUREX_PHY_ENGINE_BUILD'])
        cfg=replace(self.rt.config,phy_engine=replace(self.rt.config.phy_engine,auto_build=False,
            cmake_build_dir=str(build),verilog2plsav_path=str(build/'verilog2plsav'),phyengine_lib_path=str(build/'libphyengine.so')))
        rt=replace(self.rt,config=cfg)
        components=[{'id':'in','type':'digital_input','nodes':['shared'],'params':{'state':0}}]
        components += [{'id':'out'+str(i),'type':'digital_output','nodes':['shared'],'params':{}} for i in range(11)]
        out=circuit_create(rt,{'spec':{'components':components}})
        interface=circuit_inspect(rt,{'path':out['circuit_path'],'interface_only':True})
        shared=next(port['node'] for port in interface['ports'] if port['id']=='in')
        first=circuit_inspect(rt,{'path':out['circuit_path'],'query':shared,'limit':24})
        self.assertEqual(first['node_query']['match_count'],12)
        self.assertEqual(first['node_query']['requested_limit'],24)
        self.assertEqual(first['node_query']['limit'],8)
        self.assertEqual(first['node_query']['next_offset'],8)
        second=circuit_inspect(rt,{'path':out['circuit_path'],'query':shared,'offset':8,'limit':24})
        self.assertIsNone(second['node_query']['next_offset'])
        primary=lambda page:{c['id'] for c in page['netlist']['components'] if c.get('selection_role')=='primary'}
        self.assertEqual(len(primary(first)),8)
        self.assertEqual(len(primary(second)),4)
        self.assertEqual(primary(first)|primary(second),{'in',*[f'out{i}' for i in range(11)]})


if __name__=='__main__':unittest.main()
