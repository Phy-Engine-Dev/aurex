"""Task stop and outcome journal regressions; no model or community calls."""
from __future__ import annotations
import http.client
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from unittest import mock

from aurex.config import AurexConfig, StorageConfig, TrackingConfig
from aurex.sessiondb import SessionDB, encode

class TaskJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=str(Path(self.temp.name)/"sessions.sqlite3")
        self.db=SessionDB(self.path)
        self.sid=self.db.session("A")
        self.rid=self.db.begin(self.sid,"test request","R1")
    def running(self):
        self.db.status(self.sid,"running");self.db.run_status(self.rid,"running")
    def test_cancel_persists_is_idempotent_and_is_not_completed(self):
        self.running()
        first=self.db.request_cancel(self.sid,self.rid)
        second=self.db.request_cancel(self.sid,self.rid)
        self.assertEqual(first["status"],"cancelling")
        self.assertTrue(second["cancel_requested"])
        self.assertTrue(SessionDB(self.path).cancel_requested(self.rid))
        self.assertEqual(self.db.get(self.sid)["status"],"cancelling")
        self.assertEqual(self.db.get(self.sid)["active_run_id"],self.rid)
        self.assertEqual(self.db.list()[0]["active_run_id"],self.rid)
        self.assertEqual([event["kind"] for event in self.db.events(self.sid)],["cancel_requested"])
    def test_cancel_rejects_wrong_session_and_terminal_run_is_unchanged(self):
        other=self.db.session("B")
        with self.assertRaises(ValueError):self.db.request_cancel(other,self.rid)
        self.assertFalse(self.db.cancel_requested(self.rid))
        self.db.finish_run(self.sid,self.rid,"completed")
        result=self.db.request_cancel(self.sid,self.rid)
        self.assertFalse(result["cancel_requested"])
        self.assertEqual(result["status"],"completed")
        self.assertEqual(self.db.get(self.sid)["status"],"completed")
    def test_cancel_wins_concurrent_final_transaction(self):
        self.running()
        self.db.request_cancel(self.sid,self.rid)
        self.assertEqual(self.db.finish_run(self.sid,self.rid,"completed"),"cancelled")
        self.assertEqual(self.db.get(self.sid)["status"],"cancelled")
        with self.db.connect() as db:
            self.assertEqual(db.execute("SELECT status FROM runs WHERE id=?",(self.rid,)).fetchone()[0],"cancelled")
    def test_queued_cancel_is_not_recovered_or_executed(self):
        other=self.db.session("B")
        keep=self.db.begin(other,"keep queued","R2")
        self.db.request_cancel(self.sid,self.rid)
        queued=SessionDB(self.path).recover()
        self.assertEqual([row["id"] for row in queued],[keep])
        self.assertEqual(self.db.get(self.sid)["status"],"cancelled")
        self.assertIsNone(self.db.get(self.sid)["active_run_id"])
    def test_running_cancel_repairs_only_unknown_tool_and_preserves_completed_one(self):
        self.running()
        self.db.message(self.sid,self.rid,{"role":"assistant","content":None,"tool_calls":[
            {"id":"done","type":"function","function":{"name":"read","arguments":"{}"}},
            {"id":"unknown","type":"function","function":{"name":"external","arguments":"{}"}}]})
        did,mid=self.db.tool_outcome(self.sid,self.rid,"done","read",encode({"ok":True,"data":{"value":5}}),True)
        self.db.request_cancel(self.sid,self.rid)
        self.assertEqual(SessionDB(self.path).recover(),[])
        messages=[row["message"] for row in self.db.messages(self.sid) if row["message"]["role"]=="tool"]
        self.assertEqual(len(messages),2)
        done=next(row for row in messages if row["tool_call_id"]=="done")
        self.assertIn(did,done["content"])
        unknown=next(row for row in messages if row["tool_call_id"]=="unknown")
        self.assertIn("side effects are unknown",unknown["content"])
        self.assertTrue(self.db.get_tool_outcome(self.sid,self.rid,"done")["ok"])
        self.assertEqual(self.db.get(self.sid)["status"],"cancelled")
    def test_raw_outcome_and_message_survive_crash_before_summary(self):
        raw=encode({"ok":True,"data":{"value":5,"text":"original "*2000}})
        did,mid=self.db.tool_outcome(self.sid,self.rid,"call","measure",raw,True)
        reopened=SessionDB(self.path)
        outcome=reopened.get_tool_outcome(self.sid,self.rid,"call")
        self.assertEqual(outcome["full_json"],raw)
        self.assertEqual(outcome["message_id"],mid)
        self.assertIn(did,reopened.messages(self.sid)[0]["message"]["content"])
        reopened.update_tool_message(self.sid,mid,"Measured5 V")
        self.assertEqual(reopened.read_document(self.sid,did,length=20000)["text"],raw)
        message=reopened.messages(self.sid)[0]["message"]
        self.assertEqual(message["tool_call_id"],"call")
        self.assertIn("Measured5 V",message["content"])
        self.assertIn(did,message["content"])
    def test_duplicate_outcome_is_immutable_and_concurrent_writes_are_single_record(self):
        rows=[];errors=[]
        def write(value):
            try:rows.append(SessionDB(self.path).tool_outcome(self.sid,self.rid,"call","read",encode({"ok":True,"value":value}),True))
            except Exception as error:errors.append(error)
        threads=[threading.Thread(target=write,args=(index,)) for index in range(5)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(5)
        self.assertFalse(errors)
        self.assertEqual(len(rows),5)
        self.assertEqual(len(set(rows)),1)
        saved=self.db.get_tool_outcome(self.sid,self.rid,"call")
        self.db.tool_outcome(self.sid,self.rid,"call","changed",encode({"ok":False,"error":"later"}),False)
        self.assertEqual(self.db.get_tool_outcome(self.sid,self.rid,"call"),saved)
        with self.db.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],1)
    def test_outcome_transaction_rolls_back_document_if_message_insert_fails(self):
        with self.db.connect() as db:
            db.execute("CREATE TRIGGER reject_tool BEFORE INSERT ON messages WHEN NEW.role='tool' BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.tool_outcome(self.sid,self.rid,"call","read",'{"ok":true}',True)
        with self.db.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0],0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM tool_outcomes").fetchone()[0],0)
    def test_outcome_ownership_and_begin_run_id_conflict(self):
        other=self.db.session("B")
        with self.assertRaises(ValueError):self.db.tool_outcome(other,self.rid,"call","read","{}",True)
        did,mid=self.db.tool_outcome(self.sid,self.rid,"call","read","{}",True)
        with self.assertRaises(ValueError):self.db.update_tool_message(other,mid,"wrong session")
        with self.assertRaises(ValueError):self.db.begin(other,"different request",self.rid)
        self.assertIsNone(self.db.get_tool_outcome(other,self.rid,"call"))
        user_mid=self.db.message(self.sid,self.rid,{"role":"user","content":"immutable user"})
        with self.assertRaises(ValueError):self.db.update_tool_message(self.sid,user_mid,"not tool")
    def test_reviewed_final_is_atomic_immutable_and_idempotent(self):
        rows=[];errors=[]
        def write():
            try:rows.append(SessionDB(self.path).final_answer(self.sid,self.rid,'review1','Verified answer'))
            except Exception as exc:errors.append(exc)
        threads=[threading.Thread(target=write) for _ in range(5)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(5)
        self.assertFalse(errors)
        self.assertEqual(len(rows),5)
        self.assertEqual(len({mid for mid,created in rows}),1)
        self.assertEqual(sum(created for mid,created in rows),1)
        message=self.db.messages(self.sid)[0]['message']
        self.assertEqual(message,{'role':'assistant','content':'Verified answer','_final_review_id':'review1'})
        with self.assertRaises(ValueError):self.db.final_answer(self.sid,self.rid,'review2','Verified answer')
        with self.assertRaises(ValueError):self.db.final_answer(self.sid,self.rid,'review1','Changed answer')
        self.assertEqual(self.db.get_task(self.rid)['status'],'queued','Final message must not finish the task')
    def test_reviewed_final_rejects_wrong_session_and_rolls_back_insert_failure(self):
        other=self.db.session('other')
        with self.assertRaises(ValueError):self.db.final_answer(other,self.rid,'review','Answer')
        with self.db.connect() as db:
            db.execute("CREATE TRIGGER reject_final BEFORE INSERT ON final_answers BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.db.final_answer(self.sid,self.rid,'review','Answer')
        self.assertEqual(self.db.messages(self.sid),[])
    def test_reviewed_final_recognizes_existing_marked_message(self):
        mid=self.db.message(self.sid,self.rid,{'role':'assistant','content':'Final','_final_review_id':'existing'})
        self.assertEqual(self.db.final_answer(self.sid,self.rid,'existing','Final'),(mid,False))
        self.assertEqual(len(self.db.messages(self.sid)),1)

class TaskAgentCancellationTests(unittest.TestCase):
    def setUp(self):
        from test_aurex_v3 import SessionAgentTests
        SessionAgentTests.setUp(self)
    def agent(self,outputs,registry=None):
        from test_aurex_v3 import SessionAgentTests
        return SessionAgentTests.agent(self,outputs,registry)
    def test_cancel_during_stream_executes_no_tool_and_releases_session(self):
        from test_aurex_v3 import reply
        from aurex.tools.registry import ToolRegistry, ToolSpec
        tools=ToolRegistry();executed=mock.Mock(return_value={"value":5})
        tools.register(ToolSpec("measure","Measure",{"type":"object"},executed))
        call={"id":"never","type":"function","function":{"name":"measure","arguments":"{}"}}
        agent,fake=self.agent([reply("still generating",calls=[call],finish="tool_calls"),reply("New request completed")],tools)
        original=fake.chat
        def cancelling_chat(messages,**options):
            agent.db.request_cancel("stream-session","stream-run")
            return original(messages,**options)
        fake.chat=cancelling_chat
        result=agent.handle(user_text="Measure",session_id="stream-session",run_id="stream-run")
        self.assertTrue(result["cancelled"])
        executed.assert_not_called()
        self.assertEqual(agent.db.get("stream-session")["status"],"cancelled")
        fake.chat=original
        resumed=agent.handle(user_text="A separate explicit request",session_id="stream-session",run_id="new-run")
        self.assertEqual(resumed["answer"],"New request completed")
        self.assertNotEqual(resumed['session_id'], 'stream-session')
        self.assertEqual(agent.db.get("stream-session")["status"],"cancelled")
        self.assertEqual(agent.db.get(resumed['session_id'])["status"],"completed")
    def test_cancel_after_tool_preserves_completed_result_and_skips_next_tool(self):
        from test_aurex_v3 import reply
        from aurex.tools.registry import ToolRegistry, ToolSpec
        tools=ToolRegistry();second=mock.Mock(return_value={"must_not_run":True})
        def completed(rt,args):
            agent.db.request_cancel("tool-session","tool-run")
            return {"measured":5,"units":"V"}
        tools.register(ToolSpec("measure","Measure",{"type":"object"},completed))
        tools.register(ToolSpec("next_tool","Next tool",{"type":"object"},second))
        calls=[{"id":cid,"type":"function","function":{"name":name,"arguments":"{}"}}
               for cid,name in [("finished","measure"),("not-started","next_tool")]]
        agent,fake=self.agent([reply("",calls=calls,finish="tool_calls")],tools)
        result=agent.handle(user_text="Measure and continue",session_id="tool-session",run_id="tool-run")
        self.assertTrue(result["cancelled"])
        second.assert_not_called()
        saved=agent.db.get_tool_outcome("tool-session","tool-run","finished")
        self.assertTrue(saved["ok"])
        self.assertEqual(json.loads(saved["full_json"])["data"],{"measured":5,"units":"V"})
        self.assertIsNone(agent.db.get_tool_outcome("tool-session","tool-run","not-started"))
        self.assertEqual(len(fake.requests),1)

class TaskControlHTTPTests(unittest.TestCase):
    def setUp(self):
        import aurex.web as web
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cfg=replace(AurexConfig(),storage=StorageConfig(cache_dir=self.temp.name),
            tracking=TrackingConfig(database_path=str(Path(self.temp.name)/"sessions.sqlite3")))
        self.db=SessionDB(self.cfg.tracking.database_path)
        self.sid=self.db.session("web")
        self.rid=self.db.begin(self.sid,"queued request","web-run")
        self.agent=mock.Mock()
        self.queue=None
        self.server=None;self.started=threading.Event();self.error=[]
        def server_factory(address,handler):
            self.server=ThreadingHTTPServer(("127.0.0.1",0),handler)
            self.started.set()
            return self.server
        def capture_queue(queue):self.queue=queue
        self.patches=[mock.patch.object(web,"ThreadingHTTPServer",side_effect=server_factory),
            mock.patch.object(web.PersistentTaskQueue,"start",autospec=True,side_effect=capture_queue),
            mock.patch.dict(os.environ,{self.cfg.tracking.token_env:"test-task-control"})]
        for patch in self.patches:patch.start()
        def launch():
            try:web.serve(cfg=self.cfg,config_path=str(Path(self.temp.name)/"config.json"),agent=self.agent)
            except Exception as error:self.error.append(error);self.started.set()
        self.thread=threading.Thread(target=launch)
        self.thread.start()
        self.assertTrue(self.started.wait(5))
        self.assertFalse(self.error)
        self.addCleanup(self.stop)
    def stop(self):
        if self.server:self.server.shutdown()
        self.thread.join(5)
        for patch in reversed(self.patches):patch.stop()
    def request(self,path,value,*,auth=True,origin=None):
        conn=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        headers={"Content-Type":"application/json"}
        if auth:headers["Authorization"]="Bearer test-task-control"
        if origin:headers["Origin"]=origin
        conn.request("POST",path,json.dumps(value),headers)
        response=conn.getresponse();result=json.loads(response.read());status=response.status;conn.close()
        return status,result
    def test_cancel_endpoint_then_queued_worker_does_not_call_agent(self):
        code,result=self.request("/api/sessions/web/cancel",{"run_id":self.rid})
        self.assertEqual(code,200);self.assertEqual(result["status"],"cancelled")
        self.assertFalse(self.queue.run_next())
        self.agent.handle.assert_not_called()
        self.assertEqual(self.db.get(self.sid)["status"],"cancelled")
        self.assertEqual(SessionDB(self.db.path).recover(),[])
    def test_cancel_ownership_auth_origin_and_schema_fail_closed(self):
        self.db.session("other")
        for path,body,options,expected in [
            ("/api/sessions/web/cancel",{"run_id":self.rid},{"auth":False},401),
            ("/api/sessions/other/cancel",{"run_id":self.rid},{},400),
            ("/api/sessions/web/cancel",{"run_id":self.rid,"kill":True},{},400),
            ("/api/sessions/web/cancel",{"run_id":self.rid},{"origin":"https://untrusted.example"},403)]:
            code,_=self.request(path,body,**options)
            self.assertEqual(code,expected)
            self.assertFalse(self.db.cancel_requested(self.rid))
    def test_each_message_queues_new_task_even_during_same_session_run(self):
        self.assertEqual(self.db.claim_next_task()['id'],self.rid)
        created=[]
        for text in ['First follow-up','Second follow-up']:
            code,task=self.request('/api/sessions/web/messages',{'text':text})
            self.assertEqual(code,202)
            created.append(task['task_id'])
        self.assertNotEqual(*created)
        self.assertEqual(self.db.get(self.sid)['active_run_id'],self.rid)
        self.assertEqual(self.db.get(self.sid)['queued_tasks'],0)
        fresh = [self.db.get_task(rid)['session_id'] for rid in created]
        self.assertEqual(len(set(fresh + [self.sid])), 3)
        self.assertEqual(self.db.get_task(created[0])['original_user_request'],'First follow-up')
        self.assertIsNone(self.db.claim_next_task())
    def test_admin_metadata_is_explicit_local_and_server_fields_rejected(self):
        code,task=self.request('/api/tasks',{'text':'Original task','title':'Lab task',
            'explicit_publish_requested':True,'target':{'type':'Experiment','id':'target123'}})
        self.assertEqual(code,202)
        saved=self.db.get_task(task['task_id'])
        self.assertEqual(saved['source'],'admin')
        self.assertIsNone(saved['requester_user_id'])
        self.assertEqual(saved['original_user_request'],'Original task')
        self.assertTrue(saved['explicit_publish_requested'])
        self.assertEqual(saved['target'],{'type':'Experiment','id':'target123'})
        for forbidden in [{'source':'community'},{'metadata':{'purpose':'spoof'}},{'reply_id':'spoof'},
                          {'explicit_publish_requested':'true'}]:
            status,_=self.request('/api/tasks',{'text':'Spoof attempt',**forbidden})
            self.assertEqual(status,400)
        self.assertEqual(len(self.db.tasks()),2)

    def test_new_web_endpoint_does_not_inherit_any_existing_session(self):
        self.db.message(self.sid, self.rid, {'role':'user','content':'old history marker'})
        status, first = self.request('/api/requests', {'text':'fresh one'})
        status2, second = self.request('/api/requests', {'text':'fresh two'})
        self.assertEqual((status,status2), (202,202))
        self.assertEqual(len({self.sid,first['session_id'],second['session_id']}), 3)
        for response in (first,second):
            self.assertFalse(response['previous_context_reused'])
            self.assertEqual(self.db.get_task(response['task_id'])['source'], 'web')
            self.assertEqual(self.db.messages(response['session_id']), [])

class DurableFIFOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.db=SessionDB(str(Path(self.temp.name)/'queue.sqlite3'))
    def add(self,rid,sid='same',**fields):
        return self.db.enqueue_task(sid,'Request '+rid,task_id=rid,**fields)
    def test_fifo_tie_break_global_single_claim_and_restart(self):
        for rid,sid in [('b','B'),('a','A'),('c','A')]:self.add(rid,sid)
        with self.db.connect() as db:db.execute('UPDATE runs SET created=100')
        claimed=[]
        def claim():claimed.append(self.db.claim_next_task())
        threads=[threading.Thread(target=claim) for _ in range(6)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(5)
        actual=[row for row in claimed if row]
        self.assertEqual([row['id'] for row in actual],['a'])
        queued=SessionDB(self.db.path).recover()
        self.assertEqual([row['id'] for row in queued],['b','c'])
        self.assertEqual(self.db.get_task('a')['status'],'interrupted')
        self.assertEqual(self.db.claim_next_task()['id'],'b')
        self.db.finish_run('B','b','completed')
        self.assertEqual(self.db.claim_next_task()['id'],'c')
    def test_metadata_roundtrip_immutable_and_trusted_community_identity(self):
        fields=dict(source='community',requester_user_id='actual-author',requester_nickname='Author',reply_id='comment1',
                    explicit_publish_requested=False,target={'type':'Experiment','id':'post1'},metadata={'dry_run':True})
        rid=self.add('community',**fields)
        saved=SessionDB(self.db.path).get_task(rid)
        for key,value in fields.items():self.assertEqual(saved[key],value)
        self.assertEqual(self.add('community',**fields),rid)
        self.assertEqual(self.db.begin('same','Request community',rid),rid)
        with self.assertRaises(ValueError):self.add('community',**{**fields,'requester_user_id':'different-author'})
        with self.assertRaises(ValueError):self.add('missing-author',source='community')
        self.assertEqual(len(self.db.tasks()),1)
    def test_cancel_old_queued_task_never_hides_running_sibling(self):
        self.add('first');self.add('second');self.add('third')
        self.assertEqual(self.db.claim_next_task()['id'],'first')
        result=self.db.request_cancel('same','second')
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(self.db.get('same')['status'],'running')
        self.assertEqual(self.db.get('same')['active_run_id'],'first')
        self.db.finish_run('same','first','completed')
        self.assertEqual(self.db.get('same')['status'],'queued')
        self.assertEqual(self.db.claim_next_task()['id'],'third')
        self.db.finish_run('same','second','cancelled')
        self.assertEqual(self.db.get('same')['active_run_id'],'third')
    def test_worker_serializes_all_sources_and_never_infers_completed(self):
        from aurex.web import PersistentTaskQueue
        order=[]
        def handle(**kw):
            order.append(kw['run_id'])
            if kw['run_id']!='unverified':self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'A model final is not a verified task state'}
        agent=mock.Mock();agent.handle.side_effect=handle
        queue=PersistentTaskQueue(self.db,agent)
        queue.enqueue('A','Web task',task_id='web',source='web')
        queue.enqueue('B','Community task',task_id='community',source='community',requester_user_id='author')
        queue.enqueue('A','Admin task',task_id='unverified',source='admin')
        queue.start();self.addCleanup(lambda:queue.close(wait=True,timeout=5))
        self.assertTrue(queue.wait_idle(timeout=5))
        self.assertEqual(order,['web','community','unverified'])
        self.assertEqual(self.db.get_task('unverified')['status'],'needs_attention')
    def test_second_worker_cannot_recover_live_owner(self):
        from aurex.web import PersistentTaskQueue
        entered=threading.Event();release=threading.Event()
        def handle(**kw):
            entered.set();release.wait(5)
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        agent=mock.Mock();agent.handle.side_effect=handle
        first=PersistentTaskQueue(self.db,agent);second=PersistentTaskQueue(SessionDB(self.db.path),agent)
        first.enqueue('one','Do work',task_id='running')
        first.start()
        self.addCleanup(lambda:(release.set(),first.close(wait=True,timeout=5)))
        self.assertTrue(entered.wait(5))
        with self.assertRaises(BlockingIOError):second.start()
        self.assertEqual(self.db.get_task('running')['status'],'running')
        self.assertFalse(first.wait_idle(timeout=0))
        release.set();self.assertTrue(first.wait_idle(timeout=5))
    def test_task_scoped_events_do_not_mix_requests(self):
        self.add('one');self.add('two')
        self.db.event('same','one','user',{'text':'first'})
        self.db.event('same','two','user',{'text':'second'})
        self.assertEqual([row['run_id'] for row in self.db.events('same',run_id='two')],['two'])
        with self.assertRaises(ValueError):self.db.events('wrong',run_id='two')

@unittest.skipUnless(shutil.which("node"),"Node is required for isolated UI tests")
class TaskControlUITests(unittest.TestCase):
    def run_ui(self,scenario):
        from test_aurex_v3 import TrackingUITests
        TrackingUITests.run_ui(self,scenario)
    def test_stop_button_uses_current_run_and_shows_pending_not_completed(self):
        self.run_ui(r"""
sid='A';let stopping=false;
fetchHook=(path,options)=>{if(path.endsWith('/cancel')){stopping=true;return response({status:'cancelling',cancel_requested:true})}return response(path==='/api/sessions'?[{id:'A',title:'Task',status:stopping?'cancelling':'running',active_run_id:'R1',updated:1}]:[])};
await list();assert(!$('stop').hidden&&!$('stop').disabled,'Active stop button missing');
assert(!$('send').disabled,'Active task must allow a separate request to join the queue');
await $('stop').onclick();
const call=fetchCalls.find(x=>x.path.endsWith('/cancel'));
assert(JSON.parse(call.options.body).run_id==='R1','Stop targeted another run');
assert($('status').textContent==='取消中'&&$('stop').disabled,'Stop request was presented as already stopped/completed');
""")
    def test_late_cancel_response_cannot_replace_new_session_controls(self):
        self.run_ui(r"""
sid='A';let finishStop;
fetchHook=(path,options)=>{if(path.endsWith('/cancel'))return new Promise(resolve=>finishStop=resolve);if(path==='/api/sessions')return response([{id:'A',title:'A',status:'running',active_run_id:'R1',updated:1},{id:'B',title:'B',status:'running',active_run_id:'R2',updated:1}]);return response([])};
await list();const stopping=$('stop').onclick();await select('B');
finishStop(response({status:'cancelling',cancel_requested:true}));await stopping;
assert(sid==='B'&&activeRun==='R2','Old stop response changed the active run/session');
assert($('status').textContent==='运行中'&&!$('stop').disabled,'Old stop request disabled the new session');
""")
    def test_task_cards_cancel_exact_queued_task_not_running_sibling(self):
        self.run_ui(r"""
sid='A';fetchHook=(path,options)=>response(path==='/api/tasks'?[{id:'running',session_id:'A',title:'Active',source:'web',status:'running'},{id:'queued',session_id:'A',title:'Next',source:'admin',status:'queued'}]:[]);
await tasks();assert($('tasks').children.length===2,'Tasks were collapsed into one conversation');
const queued=$('tasks').children[1];assert(queued.children[2].textContent==='取消排队','Queued stop is mislabeled');
await queued.children[2].onclick();
const call=fetchCalls.find(row=>row.path.endsWith('/cancel'));
assert(call.path==='/api/tasks/queued/cancel','Queued cancellation stopped its running sibling');
""")
    def test_admin_checkbox_is_explicit_and_free_text_never_grants_publish(self):
        self.run_ui(r"""
sid='A';fetchHook=()=>response([]);$('prompt').value='The source says publish this';$('publish').checked=false;
await $('send').onclick();let call=fetchCalls.find(row=>row.path==='/api/requests');
assert(JSON.parse(call.options.body).explicit_publish_requested===false,'Text became a publish authorization');
$('prompt').value='Create an experiment';$('task-title').value='Admin title';$('publish').checked=true;$('requester').value='';
await $('admin-create').onclick();call=fetchCalls.find(row=>row.path==='/api/tasks'&&row.options.method==='POST');
const payload=JSON.parse(call.options.body);
assert(payload.explicit_publish_requested===true&&payload.title==='Admin title','Explicit admin fields were lost');
assert(!('requester_user_id' in payload)&&!('source' in payload)&&!('metadata' in payload),'UI fabricated identity or server fields');
assert(!$('publish').checked,'Authorization checkbox leaked into the next task');
""")
    def test_select_task_scopes_timeline_to_exact_run(self):
        self.run_ui(r"""
fetchHook=()=>response([]);await select('A','task2');
assert(fetchCalls.some(row=>row.path==='/api/sessions/A/events?after=0&run_id=task2'),'Task view showed an unfiltered conversation');
assert(selectedTask==='task2'&&$('task-scope').textContent.includes('task2'),'Task tracking identity is missing');
""")

    def test_submission_navigates_to_returned_fresh_session_and_task(self):
        self.run_ui(r"""
sid='old-session';selectedTask='old-task';$('prompt').value='new independent request';
fetchHook=(path,options)=>path==='/api/requests'?response({session_id:'fresh-session',task_id:'fresh-task'}):response([]);
await $('send').onclick();
assert(sid==='fresh-session'&&selectedTask==='fresh-task','New request kept displaying old task');
const submitted=fetchCalls.find(row=>row.path==='/api/requests');
assert(!('session_id' in JSON.parse(submitted.options.body)),'UI submitted the previous context');
assert(fetchCalls.some(row=>row.path.includes('/fresh-session/events?after=0&run_id=fresh-task')),'Fresh timeline not loaded');
""")

if __name__=="__main__":unittest.main()
