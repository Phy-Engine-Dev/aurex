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
import time
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
        self.profile_user=object()
        self.queue=None
        self.server=None;self.started=threading.Event();self.error=[]
        def server_factory(address,handler):
            self.server=ThreadingHTTPServer(("127.0.0.1",0),handler)
            self.started.set()
            return self.server
        def capture_queue(queue):self.queue=queue
        self.patches=[mock.patch.object(web,"ThreadingHTTPServer",side_effect=server_factory),
            mock.patch.object(web.PersistentTaskQueue,"start",autospec=True,side_effect=capture_queue),
            mock.patch.object(web.plar,"get_user_by_id",side_effect=lambda _user,*,user_id:{
                'User':{'ID':user_id,'Nickname':'用户-'+user_id[:4],'Signature':'公开简介','Level':6},
                'Statistic':{'ExperimentCount':12,'CommentCount':34,'FollowerCount':5}}),
            mock.patch.dict(os.environ,{self.cfg.tracking.token_env:"test-task-control"})]
        for patch in self.patches:patch.start()
        def launch():
            try:web.serve(cfg=self.cfg,config_path=str(Path(self.temp.name)/"config.json"),agent=self.agent,user=self.profile_user,poll=False)
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
    def request(self,path,value,*,auth=True,origin=None,cookie=None):
        conn=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        headers={"Content-Type":"application/json"}
        if auth:headers["Authorization"]="Bearer test-task-control"
        if origin:headers["Origin"]=origin
        if cookie:headers["Cookie"]=cookie
        conn.request("POST",path,json.dumps(value),headers)
        response=conn.getresponse();result=json.loads(response.read());status=response.status;conn.close()
        return status,result
    def get_request(self,path,*,auth=True,cookie=None):
        conn=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        headers={}
        if auth:headers["Authorization"]="Bearer test-task-control"
        if cookie:headers["Cookie"]=cookie
        conn.request("GET",path,headers=headers)
        response=conn.getresponse();result=json.loads(response.read());status=response.status;conn.close()
        return status,result
    def login_user(self,cookie=None,user_id='a'*24):
        conn=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        headers={"Content-Type":"application/json"}
        if cookie:headers["Cookie"]=cookie
        conn.request("POST","/api/login",json.dumps({"mode":"user","user_id":user_id}),headers)
        response=conn.getresponse();result=json.loads(response.read());status=response.status
        cookies=[value.split(';',1)[0] for key,value in response.getheaders() if key.lower()=='set-cookie']
        conn.close();self.assertEqual(status,200);self.assertEqual(result['role'],'user')
        return '; '.join(value for value in cookies if value.startswith(('aurex_user=','aurex_user_id=')))
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

    def test_user_mode_is_owned_anonymous_and_admin_mode_is_global(self):
        user_a=self.login_user(user_id='a'*24);self.assertEqual(self.login_user(user_a,user_id='a'*24),user_a)
        code,a=self.request('/api/requests',{'text':'private request A'},auth=False,cookie=user_a)
        self.assertEqual(code,202)
        user_b=self.login_user(user_id='b'*24)
        code,b=self.request('/api/requests',{'text':'private request B'},auth=False,cookie=user_b)
        self.assertEqual(code,202)

        code,identity=self.get_request('/api/me',auth=False,cookie=user_a)
        self.assertEqual((code,identity['role']),(200,'user'))
        self.assertEqual(identity['user_id'],'a'*24)
        self.assertEqual(identity['profile']['nickname'],'用户-aaaa')
        code,sessions=self.get_request('/api/sessions',auth=False,cookie=user_a)
        self.assertEqual([row['id'] for row in sessions],[a['session_id']])
        self.assertEqual(self.get_request('/api/sessions/'+b['session_id'],auth=False,cookie=user_a)[0],404)
        self.assertEqual(self.get_request('/api/tasks/'+b['task_id'],auth=False,cookie=user_a)[0],404)
        self.assertEqual(self.request('/api/tasks',{'text':'forbidden admin task'},auth=False,cookie=user_a)[0],403)
        self.assertEqual(self.request('/api/tasks/'+b['task_id']+'/cancel',{},auth=False,cookie=user_a)[0],404)

        code,own_tasks=self.get_request('/api/tasks',auth=False,cookie=user_a)
        self.assertEqual([row['id'] for row in own_tasks],[a['task_id']])
        code,global_queue=self.get_request('/api/tasks?active=1',auth=False,cookie=user_a)
        self.assertEqual(code,200)
        self.assertEqual(sum(row['mine'] for row in global_queue),1)
        self.assertNotIn('private request B',json.dumps(global_queue))
        self.assertTrue(any(row['title']=='其他用户任务' for row in global_queue))
        self.assertEqual(self.request('/api/tasks/'+a['task_id']+'/cancel',{},auth=False,cookie=user_a)[0],200)

        code,admin_sessions=self.get_request('/api/sessions')
        self.assertEqual(code,200)
        self.assertTrue({self.sid,a['session_id'],b['session_id']}.issubset({row['id'] for row in admin_sessions}))
        code,admin_queue=self.get_request('/api/tasks?active=1')
        self.assertEqual(code,200)
        self.assertIn('private request B',json.dumps(admin_queue))

    def test_public_user_id_is_profile_label_not_a_session_credential(self):
        self.assertEqual(self.request('/api/login',{'mode':'user','user_id':'short'},auth=False)[0],400)
        first=self.login_user(user_id='c'*24)
        code,created=self.request('/api/requests',{'text':'C private'},auth=False,cookie=first)
        self.assertEqual(code,202)
        switched=self.login_user(first,user_id='d'*24)
        self.assertEqual(self.get_request('/api/sessions/'+created['session_id'],auth=False,cookie=switched)[0],404)
        restored=self.login_user(switched,user_id='c'*24)
        self.assertEqual(self.get_request('/api/sessions/'+created['session_id'],auth=False,cookie=restored)[0],200)
        code,identity=self.get_request('/api/me',auth=False,cookie=restored)
        self.assertEqual(code,200)
        self.assertEqual(identity['profile']['id'],'c'*24)
        self.assertNotIn('token',json.dumps(identity).casefold())

    def test_subagent_trace_is_parent_scoped_bounded_and_hides_private_messages(self):
        child='1'*32
        self.db.create_subagent(self.sid,self.rid,child,'核对一个节点',{
            'private_handoff':'must not be returned by the Web API'})
        self.db.subagent_message(child,{'role':'assistant','content':'PRIVATE_REASONING'})
        self.db.subagent_event(child,'model_start',{'step':1,'thinking':True})
        document_id,message_id=self.db.subagent_tool_outcome(
            child,child+':1:0:measure','circuit_inspect',
            encode({'ok':True,'data':{'voltage':4.98}}),True)
        self.db.update_subagent_tool_message(child,message_id,'voltage=4.98')
        report={'child_id':child,'status':'completed','conclusion':'实测约 4.98 V',
                'key_evidence':[document_id],'limitations':[],'next_action':'交还主 Agent'}
        self.db.finish_subagent(child,'completed',report)

        status,rows=self.get_request('/api/sessions/web/tasks/web-run/subagents')
        self.assertEqual(status,200)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['objective'],'核对一个节点')
        self.assertEqual(rows[0]['report']['conclusion'],'实测约 4.98 V')
        self.assertNotIn('context',rows[0])
        status,trace=self.get_request('/api/sessions/web/tasks/web-run/subagents/'+child)
        self.assertEqual(status,200)
        self.assertEqual(trace['subagent']['id'],child)
        self.assertEqual(trace['events'][0]['kind'],'model_start')
        self.assertEqual(trace['tool_outcomes'][0]['document_id'],document_id)
        self.assertNotIn('messages',trace)
        self.assertNotIn('PRIVATE_REASONING',json.dumps(trace))
        self.assertNotIn('private_handoff',json.dumps(trace))

    def test_subagent_api_rejects_cross_session_parent_and_child_combinations(self):
        child='2'*32
        self.db.create_subagent(self.sid,self.rid,child,'owned by web',{})
        other=self.db.session('other')
        other_run=self.db.begin(other,'other task','other-run')
        other_child='3'*32
        self.db.create_subagent(other,other_run,other_child,'owned by other',{})
        for path in (
            '/api/sessions/other/tasks/web-run/subagents',
            '/api/sessions/web/tasks/other-run/subagents',
            '/api/sessions/web/tasks/web-run/subagents/'+other_child,
            '/api/tasks/web-run/subagents',
        ):
            with self.subTest(path=path):
                status,result=self.get_request(path)
                self.assertEqual(status,404)
                self.assertIn('error',result)

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

    def test_parallel_claim_is_atomic_bounded_and_serial_per_session(self):
        for rid,sid in [('a','same'),('b','same'),('c','other'),('d','third')]:
            self.add(rid,sid)
        claimed=[];errors=[]
        def claim():
            try:claimed.append(SessionDB(self.db.path).claim_next_task(2))
            except Exception as exc:errors.append(exc)
        threads=[threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(5)
        self.assertFalse(errors)
        actual=[row for row in claimed if row]
        self.assertEqual(len(actual),2)
        self.assertEqual({row['id'] for row in actual},{'a','c'})
        self.assertEqual(len({row['session_id'] for row in actual}),2)
        self.assertIsNone(self.db.claim_next_task(2))
        self.db.finish_run('same','a','completed')
        self.assertEqual(self.db.claim_next_task(2)['id'],'b')
        with self.assertRaises(ValueError):self.db.claim_next_task(0)
        with self.assertRaises(ValueError):self.db.claim_next_task(True)
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
    def test_web_session_owner_filters_and_cannot_be_rebound(self):
        owned=self.db.session('owned',owner_id='owner-a')
        self.db.enqueue_task(owned,'private',task_id='private',owner_id='owner-a')
        self.db.enqueue_task('operator','admin',task_id='admin')
        self.assertEqual([row['id'] for row in self.db.list(owner_id='owner-a')],['owned'])
        self.assertEqual([row['id'] for row in self.db.tasks(owner_id='owner-a')],['private'])
        self.assertIsNone(self.db.get('operator',owner_id='owner-a'))
        with self.assertRaisesRegex(ValueError,'different Web user'):
            self.db.enqueue_task(owned,'intrusion',task_id='intrusion',owner_id='owner-b')
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

    def test_parallel_workers_run_independent_sessions_and_bound_capacity(self):
        from aurex.web import PersistentTaskQueue
        entered=[];lock=threading.Lock();two_running=threading.Event();release=threading.Event()
        def handle(**kw):
            with lock:
                entered.append(kw['run_id'])
                if len(entered)==2:two_running.set()
            release.wait(5)
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        agent=mock.Mock();agent.handle.side_effect=handle
        queue=PersistentTaskQueue(self.db,agent,max_parallel_tasks=2)
        for rid,sid in [('one','A'),('two','B'),('three','C')]:
            queue.enqueue(sid,'Task '+rid,task_id=rid)
        queue.start()
        self.addCleanup(lambda:(release.set(),queue.close(wait=True,timeout=5)))
        self.assertTrue(two_running.wait(5))
        with lock:self.assertEqual(set(entered),{'one','two'})
        self.assertEqual(len([row for row in self.db.tasks() if row['status']=='running']),2)
        release.set();self.assertTrue(queue.wait_idle(timeout=5))
        self.assertEqual(set(entered[:2]),{'one','two'})
        self.assertEqual(entered[2:],['three'])

    def test_parallel_workers_never_overlap_sibling_requests_in_one_session(self):
        from aurex.web import PersistentTaskQueue
        first_entered=threading.Event();release=threading.Event();order=[]
        def handle(**kw):
            order.append(kw['run_id'])
            if kw['run_id']=='first':
                first_entered.set();release.wait(5)
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        agent=mock.Mock();agent.handle.side_effect=handle
        queue=PersistentTaskQueue(self.db,agent,max_parallel_tasks=2)
        self.add('first','same');self.add('second','same')
        queue.start()
        self.addCleanup(lambda:(release.set(),queue.close(wait=True,timeout=5)))
        self.assertTrue(first_entered.wait(5))
        self.assertEqual(order,['first'])
        self.assertEqual(self.db.get_task('second')['status'],'queued')
        release.set();self.assertTrue(queue.wait_idle(timeout=5))
        self.assertEqual(order,['first','second'])

    def test_parallel_busy_count_wait_idle_and_close_cover_every_worker(self):
        from aurex.web import PersistentTaskQueue
        entered=threading.Event();lock=threading.Lock();active=set()
        releases={'one':threading.Event(),'two':threading.Event()}
        def handle(**kw):
            with lock:
                active.add(kw['run_id'])
                if len(active)==2:entered.set()
            releases[kw['run_id']].wait(5)
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        agent=mock.Mock();agent.handle.side_effect=handle
        queue=PersistentTaskQueue(self.db,agent,max_parallel_tasks=2)
        queue.enqueue('A','one',task_id='one');queue.enqueue('B','two',task_id='two')
        queue.start()
        self.addCleanup(lambda:(releases['one'].set(),releases['two'].set(),queue.close(wait=True,timeout=5)))
        self.assertTrue(entered.wait(5));self.assertTrue(queue.busy.is_set())
        self.assertFalse(queue.wait_idle(timeout=0))
        releases['one'].set()
        deadline=time.monotonic()+5
        while self.db.get_task('one')['status']!='completed' and time.monotonic()<deadline:
            time.sleep(.01)
        self.assertTrue(queue.busy.is_set(),'one worker clearing busy hid its active sibling')
        self.assertFalse(queue.wait_idle(timeout=0))
        releases['two'].set();self.assertTrue(queue.wait_idle(timeout=5))
        self.assertFalse(queue.busy.is_set())
        queue.close(wait=True,timeout=5)
        self.assertTrue(all(not worker.is_alive() for worker in queue.threads))
        self.assertIsNone(queue.lock_file)

    def test_queue_capacity_rejects_zero_and_boolean(self):
        from aurex.web import PersistentTaskQueue
        for value in (0, -1, True, 65, '2'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                PersistentTaskQueue(self.db,mock.Mock(),max_parallel_tasks=value)

    def test_shutdown_and_claim_are_linearized_and_leave_later_work_queued(self):
        from aurex.web import PersistentTaskQueue
        claim_entered=threading.Event();release_claim=threading.Event()
        handled=[]
        original_claim=self.db.claim_next_task
        def blocked_claim(capacity=1):
            claim_entered.set()
            release_claim.wait(5)
            return original_claim(capacity)
        self.db.claim_next_task=blocked_claim
        def handle(**kw):
            handled.append(kw['run_id'])
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        queue=PersistentTaskQueue(self.db,mock.Mock(handle=handle))
        queue.enqueue('A','first',task_id='first')
        queue.enqueue('B','second',task_id='second')
        queue.start()
        self.addCleanup(lambda:(release_claim.set(),queue.close(wait=True,timeout=5)))
        self.assertTrue(claim_entered.wait(5))
        close_entered=threading.Event();closed=threading.Event()
        def close_queue():
            close_entered.set()
            queue.close(wait=False)
            closed.set()
        closer=threading.Thread(target=close_queue)
        closer.start()
        # The first claim already owns the scheduling decision, so shutdown
        # must wait for that short transaction rather than racing through it.
        self.assertTrue(close_entered.wait(5))
        self.assertFalse(closed.wait(.05))
        release_claim.set()
        self.assertTrue(closed.wait(5));closer.join(5)
        queue.close(wait=True,timeout=5)
        self.assertEqual(handled,['first'])
        self.assertEqual(self.db.get_task('first')['status'],'completed')
        self.assertEqual(self.db.get_task('second')['status'],'queued')

    def test_partial_worker_start_failure_keeps_process_lock_until_started_worker_exits(self):
        from aurex.web import PersistentTaskQueue
        entered=threading.Event();release=threading.Event();errors=[]
        def handle(**kw):
            entered.set();release.wait(5)
            self.db.finish_run(kw['session_id'],kw['run_id'],'completed')
            return {'answer':'done'}
        queue=PersistentTaskQueue(self.db,mock.Mock(handle=handle),max_parallel_tasks=2)
        queue.enqueue('A','work',task_id='work')
        original_start=threading.Thread.start
        def fail_second(worker):
            if worker.name=='aurex-task-2':
                if not entered.wait(5):
                    raise AssertionError('first worker did not start')
                raise RuntimeError('synthetic thread start failure')
            return original_start(worker)
        def launch():
            try:queue.start()
            except Exception as exc:errors.append(exc)
        with mock.patch.object(threading.Thread,'start',fail_second):
            starter=threading.Thread(target=launch,name='test-starter')
            starter.start()
            self.assertTrue(entered.wait(5))
            self.assertIsNotNone(queue.lock_file)
            second=PersistentTaskQueue(SessionDB(self.db.path),mock.Mock())
            with self.assertRaises(BlockingIOError):second.start()
            release.set();starter.join(5)
        self.assertEqual(len(errors),1)
        self.assertIn('synthetic thread start failure',str(errors[0]))
        self.assertTrue(all(not worker.is_alive() for worker in queue.threads))
        self.assertIsNone(queue.lock_file)

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
sid='A';fetchHook=(path,options)=>response(path.startsWith('/api/tasks?')?[{id:'running',session_id:'A',title:'Active',source:'web',status:'running',queue_position:1},{id:'queued',session_id:'A',title:'Next',source:'admin',status:'queued',queue_position:2}]:[]);
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
assert(!('explicit_publish_requested' in JSON.parse(call.options.body)),'Normal user payload included a publication authorization field');
$('prompt').value='Create an experiment';currentRole='admin';$('task-title').value='Admin title';$('publish').checked=true;$('requester').value='';
await $('admin-create').onclick();call=fetchCalls.find(row=>row.path==='/api/tasks'&&row.options.method==='POST');
const payload=JSON.parse(call.options.body);
assert(payload.explicit_publish_requested===true&&payload.title==='Admin title','Explicit admin fields were lost');
assert(!('requester_user_id' in payload)&&!('source' in payload)&&!('metadata' in payload),'UI fabricated identity or server fields');
assert(!$('publish').checked,'Authorization checkbox leaked into the next task');
""")
    def test_role_controls_and_private_queue_card_are_not_fake_permissions(self):
        self.run_ui(r"""
applyIdentity({role:'user'});assert($('admin-settings').hidden,'User can see administrator controls');
assert($('role-badge').textContent==='普通用户','User role is not visible');
fetchHook=path=>response(path.startsWith('/api/tasks?')?[{id:'private-1',session_id:'',title:'其他用户任务',source:'private',status:'queued',mine:false,queue_position:1}]:[]);
await tasks();const card=$('tasks').children[0];assert(card.className.includes('private'),'Redacted global task is not marked private');
assert(card.children.length===2&&card.children[0].disabled,'User can open or cancel another user task');
applyIdentity({role:'admin'});assert(!$('admin-settings').hidden&&$('role-badge').textContent==='管理员','Administrator controls were not enabled');
document.body={dataset:{}};openMobilePanel('queue');assert(document.body.dataset.mobileView==='queue','Mobile queue did not open');closeMobilePanel();assert(document.body.dataset.mobileView==='chat','Mobile panel did not return to conversation');
""")

    def test_login_restores_exact_task_deep_link_and_public_profile(self):
        self.run_ui(r"""
location.search='?session=A&task=R';sid='A';fetchHook=path=>response(path==='/api/sessions'?[{id:'A',title:'Task',status:'running',active_run_id:'R',updated:1}]:[]);
await finishLogin({role:'user',user_id:'aaaaaaaaaaaaaaaaaaaaaaaa',profile:{id:'aaaaaaaaaaaaaaaaaaaaaaaa',nickname:'测试用户',signature:'公开简介',stats:{experiments:12,comments:34,followers:5}}});
assert(selectedTask==='R','Login lost the task scope from a deep link');
assert(fetchCalls.some(row=>row.path.includes('/api/sessions/A/events?after=0&run_id=R')),'Login loaded the whole conversation instead of the requested task');
assert($('profile-name').textContent==='测试用户'&&$('profile-stats').textContent.includes('实验 12'),'Public PhysicsLab profile was not shown');
""")

    def test_queue_and_sessions_are_peer_panels_with_resizable_mobile_layout(self):
        html=(Path(__file__).resolve().parents[1]/'src/aurex/tracking.html').read_text()
        peers=['<main id="workspace">','<aside id="sessions-pane"','<section id="conversation-pane"',
               '<div id="queue-resizer"','<aside id="queue-pane"']
        positions=[html.index(fragment) for fragment in peers]
        self.assertEqual(positions,sorted(positions))
        self.assertIn('grid-template-columns:var(--sessions-width) minmax(0,1fr) 7px var(--queue-width)',html)
        self.assertIn("storageSet('aurex.queueWidth'",html)
        self.assertIn('height:100dvh',html)
        self.assertIn('body[data-mobile-view="queue"] #queue-pane',html)
        self.assertIn('.side-pane{max-height:none}#sessions{display:block}.session{min-width:0;max-width:none}',html)
        self.assertIn('#login{z-index:100}',html)
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

    def test_subagents_render_as_one_foldable_parent_scoped_trace_each(self):
        self.run_ui(r"""
fetchCalls.length=0;
const child='11111111111111111111111111111111';
fetchHook=(path)=>{
 if(path==='/api/sessions')return response([{id:'S',title:'Parent',status:'running',active_run_id:'R',updated:1}]);
 if(path==='/api/sessions/S/events?after=0&run_id=R')return response([]);
 if(path==='/api/sessions/S/tasks/R/subagents')return response([{id:child,objective:'核对输出节点',status:'completed'}]);
 if(path==='/api/sessions/S/tasks/R/subagents/'+child)return response({
  subagent:{id:child,depth:1,objective:'核对输出节点',status:'completed',report:{status:'completed',conclusion:'输出为高电平',key_evidence:['doc1'],limitations:[],next_action:'交还主 Agent'}},
  events:[{id:1,kind:'tool_end',created:1,data:{name:'circuit_inspect',ok:true}}],
  tool_outcomes:[{name:'circuit_inspect',ok:true,call_id:child+':1:0:x',document_id:'doc1',created:1}],
  messages:[{data:{reasoning:'PRIVATE_CHILD_REASONING'}}]
 });
 return response([]);
};
await select('S','R');
assert(subagentGroups.size===1,'Child did not get one foldable panel');
const panel=subagentGroups.get(child);
assert(panel.box.className.includes('subagent'),'Child trace is not a dedicated foldable region');
assert(panel.heading.textContent.includes('已完成')&&panel.heading.textContent.includes('核对输出节点'),'Objective/status missing');
assert(panel.eventBody.textContent.includes('circuit_inspect'),'Child events missing');
assert(panel.toolBody.textContent.includes('doc1'),'Child tool evidence missing');
assert(panel.reportBody.textContent.includes('输出为高电平'),'Compact report missing');
assert(!JSON.stringify($('timeline')).includes('PRIVATE_CHILD_REASONING'),'Private child reasoning leaked into the UI');
await subagents();
assert(subagentGroups.size===1,'Polling duplicated the child panel');
assert(fetchCalls.some(row=>row.path==='/api/sessions/S/tasks/R/subagents'),'UI did not use a session-scoped child endpoint');
assert(!fetchCalls.some(row=>row.path==='/api/tasks/R/subagents'),'UI used the cross-session task-only endpoint');
""")

if __name__=="__main__":unittest.main()
