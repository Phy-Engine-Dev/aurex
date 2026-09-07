import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from aurex.context_retrieval import read_context
from aurex.sessiondb import SessionDB


class LiteralFindTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.db=SessionDB(str(Path(self.temp.name)/'test.sqlite3'));self.sid=self.db.session('owned')
        self.raw='开头\n'+'x'*90000+' C152 节点N610 '+'y'*10000+' C152 第二处'
        self.did=self.db.document(self.sid,'Source',self.raw)

    def test_literal_find_avoids_prefix_scan_and_returns_exact_offsets(self):
        out=read_context(self.db,self.sid,{'document_id':self.did,'find':'C152','length':500})
        self.assertTrue(out['found']);self.assertEqual(out['match_offset'],self.raw.index('C152'))
        self.assertEqual(out['text'],self.raw[out['offset']:out['offset']+500])
        self.assertEqual(out['document_sha256'],hashlib.sha256(self.raw.encode()).hexdigest())
        next_one=read_context(self.db,self.sid,{'document_id':self.did,'find':'C152','offset':out['next_search_offset'],'length':500})
        self.assertEqual(next_one['match_offset'],self.raw.rindex('C152'))

    def test_no_match_or_regex_never_synthesizes_a_value(self):
        for needle in ('missing','C[0-9]+'):
            out=read_context(self.db,self.sid,{'document_id':self.did,'find':needle})
            self.assertFalse(out['found']);self.assertEqual(out['text'],'');self.assertIsNone(out['next_search_offset'])

    def test_exact_circuit_selector_redirects_to_bounded_connectivity_tool(self):
        did=self.db.document(self.sid,'circuit_inspect: full netlist_path','{"components":[],"nodes":[{"id":"N195"}]}')
        for needle in ('N195','"N195"','C152'):
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError,'Call circuit_inspect') as error:
                read_context(self.db,self.sid,{'document_id':did,'find':needle})
            self.assertIn(needle.strip('"'),str(error.exception))
            self.assertIn('No source value was inferred',str(error.exception))
        ordinary=self.db.document(self.sid,'Ordinary source','N195')
        self.assertTrue(read_context(self.db,self.sid,{'document_id':ordinary,'find':'N195'})['found'])

    def test_archived_renderer_netlist_rejects_json_select_and_plain_paging(self):
        did=self.db.document(self.sid,'circuit_inspect: full netlist_path',json.dumps({
            'components':[{'id':'a','type':'D Flipflop'}]}))
        attempts=(
            {'document_id':did,'json_pointer':'/components','select':{
                'where':{'type':'D Flipflop'},'fields':['id'],'limit':1}},
            {'document_id':did,'offset':0,'length':100},
        )
        for args in attempts:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError,'Call circuit_inspect'):
                read_context(self.db,self.sid,args)

    def test_subtree_find_is_scoped_and_hash_binds_original(self):
        original=json.dumps({'a':'C152 elsewhere','b':[{'name':'C152','node':'N610'}]},ensure_ascii=False)
        did=self.db.document(self.sid,'JSON',original)
        out=read_context(self.db,self.sid,{'document_id':did,'json_pointer':'/b','find':'C152','length':100})
        self.assertNotIn('elsewhere',out['text']);self.assertEqual(out['json_pointer'],'/b')
        self.assertEqual(out['document_sha256'],hashlib.sha256(original.encode()).hexdigest())

    def test_cross_session_invalid_search_and_boundaries_rejected(self):
        with self.assertRaises(ValueError):read_context(self.db,'foreign',{'document_id':self.did,'find':'C152'})
        for args in ({'find':''},{'find':None},{'find':True},{'find':'a'*257},{'find':'a','offset':-1},{'find':'a','length':20001}):
            with self.assertRaises(ValueError):read_context(self.db,self.sid,{'document_id':self.did,**args})

    def test_select_returns_complete_exact_records_without_scanning(self):
        original=json.dumps({'components':[
            {'id':'a','type':'D Flipflop','pins':[1,2,3],'private':'keep in source'},
            {'id':'b','type':'And Gate','pins':[4,5,6]},
            {'id':'c','type':'D Flipflop','pins':[7,8,9]}]},ensure_ascii=False)
        did=self.db.document(self.sid,'Netlist',original)
        out=read_context(self.db,self.sid,{'document_id':did,'json_pointer':'/components',
            'select':{'where':{'type':'D Flipflop'},'fields':['id','pins','missing'],'limit':1}})
        data=json.loads(out['text'])
        self.assertEqual(data['total_matches'],2);self.assertTrue(data['has_more_rows'])
        self.assertEqual(data['next_row_offset'],1);self.assertEqual(data['rows'][0]['source_index'],0)
        self.assertEqual(data['rows'][0]['value'],{'id':'a','pins':[1,2,3]})
        self.assertEqual(data['rows'][0]['missing_fields'],['missing'])
        page=read_context(self.db,self.sid,{'document_id':did,'json_pointer':'/components',
            'select':{'where':{'type':'D Flipflop'},'fields':['id'],'offset':1,'limit':1}})
        self.assertEqual(json.loads(page['text'])['rows'][0]['value'],{'id':'c'})

    def test_select_is_bounded_exact_and_array_only(self):
        did=self.db.document(self.sid,'JSON',json.dumps({'rows':[{'n':1},{'n':1.0}],'object':{}}))
        exact=read_context(self.db,self.sid,{'document_id':did,'json_pointer':'/rows',
            'select':{'where':{'n':1}}})
        self.assertEqual(json.loads(exact['text'])['total_matches'],1)
        invalid=(
            {'document_id':did,'select':{}},
            {'document_id':did,'json_pointer':'/rows','find':'n','select':{}},
            {'document_id':did,'json_pointer':'/object','select':{}},
            {'document_id':did,'json_pointer':'/rows','select':{'unknown':1}},
            {'document_id':did,'json_pointer':'/rows','select':{'limit':65}},
            {'document_id':did,'json_pointer':'/rows','select':{'fields':[]}},
        )
        for args in invalid:
            with self.assertRaises(ValueError):read_context(self.db,self.sid,args)


if __name__=='__main__':unittest.main()
