import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.agent import AurexAgent, AurexAgentError
from aurex.config import AurexConfig
from aurex.ollama import OllamaChatResponse
from aurex.tools.registry import ToolRegistry, ToolSpec, ToolRuntime
from aurex.agent import Plan, PlanStep


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("no more fake responses")
        return self.responses.pop(0)


def _mk_agent(*, planner_resps, executor_resps, registry: ToolRegistry):
    cfg = AurexConfig()
    import logging

    logger = logging.getLogger("test_aurex")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    agent = AurexAgent(cfg=cfg, config_path=os.path.join(ROOT, "dummy.json"), tools=registry, logger=logger)
    agent.planner_client = _FakeClient(planner_resps)  # type: ignore[assignment]
    agent.executor_client = _FakeClient(executor_resps)  # type: ignore[assignment]
    return agent


class TestAurexAgent(unittest.TestCase):
    def test_handle_no_steps(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"T1","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="好的", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        out = agent.handle(user_text="介绍一下你自己")
        self.assertEqual(out["answer"], "好的")

    def test_greeting_goes_through_ai(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"TQ","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="你好！", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        out = agent.handle(user_text="CONTEXT_JSON:\n{}\n\n你好")
        self.assertEqual(out["answer"], "你好！")

    def test_language_decided_by_planner(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"TL","user_lang":"en","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="OK", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        out = agent.handle(user_text="what is the 中 means in chinese")
        self.assertEqual(out["answer"], "OK")

    def test_planner_empty_response_fallback(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[OllamaChatResponse(content="", tool_calls=[], raw={}), OllamaChatResponse(content="", tool_calls=[], raw={})],
            executor_resps=[
                # executor(json-plan) attempt (invalid -> fallback)
                OllamaChatResponse(content="not json", tool_calls=[], raw={}),
                # final fallback answer
                OllamaChatResponse(content="fallback ok", tool_calls=[], raw={}),
            ],
            registry=reg,
        )
        out = agent.handle(user_text="please help")
        self.assertEqual(out["answer"], "fallback ok")

    def test_nl_plan_compiled_by_executor(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content="", tool_calls=[], raw={}),
                OllamaChatResponse(content="user_lang=zh\ngoal=打个招呼\n无需工具，直接回复。", tool_calls=[], raw={}),
                OllamaChatResponse(content="你好！", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(content='{"task_id":"TN","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={})
            ],
            registry=reg,
        )
        out = agent.handle(user_text="你好")
        self.assertEqual(out["answer"], "你好！")

    def test_one_tool_step(self):
        reg = ToolRegistry()

        def echo_tool(_rt, args):
            return {"echo": args}

        reg.register(
            ToolSpec(
                name="echo",
                description="",
                parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                handler=echo_tool,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"T2","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"echo","hint":"set x"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="done", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "echo", "arguments": {"x": 1}}}],
                    raw={},
                )
            ],
            registry=reg,
        )
        out = agent.handle(user_text="run")
        self.assertEqual(out["answer"], "done")
        self.assertEqual(len(out["tool_results"]), 1)
        self.assertTrue(out["tool_results"][0].ok)
        self.assertEqual(out["tool_results"][0].data["echo"]["x"], 1)

    def test_executor_wrong_tool_then_retry_ok(self):
        reg = ToolRegistry()

        reg.register(
            ToolSpec(
                name="echo",
                description="",
                parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                handler=lambda _rt, args: {"x": args.get("x")},
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"T3","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"echo","hint":"x=1"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
                    raw={},
                ),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "echo", "arguments": {"x": 7}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )
        out = agent.handle(user_text="run")
        self.assertEqual(out["answer"], "ok")
        self.assertEqual(out["tool_results"][0].data["x"], 7)

    def test_local_context_tool_args_filled_from_context_json(self):
        reg = ToolRegistry()

        def _local_ctx(_rt: ToolRuntime, args: dict) -> dict:
            return {
                "target_key": args.get("target_key"),
                "target_type": args.get("target_type"),
                "target_id": args.get("target_id"),
            }

        reg.register(
            ToolSpec(
                name="local_get_target_context",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=_local_ctx,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TLC","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"local_get_target_context","hint":"use context"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {"function": {"name": "local_get_target_context", "arguments": {"take": 5, "target_id": "u1"}}}
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )

        out = agent.handle(user_text='CONTEXT_JSON:\n{"target":{"type":"User","id":"u1"}}\n\n分析这个留言板')
        self.assertEqual(out["answer"], "ok")
        self.assertEqual(len(out["tool_results"]), 1)
        self.assertTrue(out["tool_results"][0].ok)
        self.assertEqual(out["tool_results"][0].data["target_key"], "User:u1")
        self.assertEqual(out["tool_results"][0].data["target_type"], "User")
        self.assertEqual(out["tool_results"][0].data["target_id"], "u1")

    def test_execute_retries_same_step_on_tool_error(self):
        reg = ToolRegistry()

        def needs_id(_rt: ToolRuntime, args: dict) -> dict:
            vid = str(args.get("id") or "")
            if len(vid) != 24:
                raise Exception("invalid id")  # will be wrapped as error
            return {"ok": True, "id": vid}

        reg.register(
            ToolSpec(
                name="needs_id",
                description="",
                parameters={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                handler=needs_id,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TR","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"needs_id","hint":"use id from results"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "needs_id", "arguments": {"id": "0123456789abcdef"}}}],
                    raw={},
                ),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "needs_id", "arguments": {"id": "0123456789abcdef01234567"}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )
        out = agent.handle(user_text="run")
        self.assertEqual(out["answer"], "ok")
        self.assertTrue(out["tool_results"][-1].ok)

    def test_execute_injects_featured_tag_for_query_experiments(self):
        reg = ToolRegistry()

        def _qe(_rt: ToolRuntime, args: dict) -> dict:
            return {"args": dict(args)}

        reg.register(
            ToolSpec(
                name="plar_query_experiments",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"category": {"type": "string"}, "take": {"type": "integer"}, "tags": {"type": ["array", "null"]}},
                    "required": ["category"],
                },
                handler=_qe,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TQF","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"plar_query_experiments","hint":"latest featured"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_query_experiments", "arguments": {"category": "Experiment", "take": 1}}}],
                    raw={},
                )
            ],
            registry=reg,
        )

        out = agent.handle(user_text="最新实验区精选的标题是什么")
        self.assertEqual(out["answer"], "ok")
        self.assertEqual(len(out["tool_results"]), 1)
        self.assertTrue(out["tool_results"][0].ok)
        got_args = out["tool_results"][0].data["args"]
        self.assertIn("tags", got_args)
        self.assertIn("精选", got_args["tags"])

    def test_execute_injects_user_id_for_my_latest_query_from_context_json(self):
        reg = ToolRegistry()

        def _qe(_rt: ToolRuntime, args: dict) -> dict:
            return {"args": dict(args)}

        reg.register(
            ToolSpec(
                name="plar_query_experiments",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"category": {"type": "string"}, "take": {"type": "integer"}, "user_id": {"type": ["string", "null"]}},
                    "required": ["category"],
                },
                handler=_qe,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        author_id = "a" * 24
        target_id = "b" * 24
        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TML","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"plar_query_experiments","hint":"latest my experiment"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {"function": {"name": "plar_query_experiments", "arguments": {"category": "Experiment", "take": 1}}}
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )

        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"User","id":"'
            + target_id
            + '"},"comment":{"id":"c1","author_id":"'
            + author_id
            + '","author_nickname":"goodenough"}}\n\n我的最新实验是什么'
        )
        out = agent.handle(user_text=user_text)
        self.assertEqual(out["answer"], "ok")
        got_args = out["tool_results"][0].data["args"]
        self.assertEqual(got_args.get("user_id"), author_id)

    def test_execute_injects_user_id_from_previous_plar_get_user_result(self):
        reg = ToolRegistry()

        def _get_user(_rt: ToolRuntime, args: dict) -> dict:
            return {"id": "c" * 24, "nickname": args.get("name")}

        def _qe(_rt: ToolRuntime, args: dict) -> dict:
            return {"args": dict(args)}

        reg.register(
            ToolSpec(
                name="plar_get_user",
                description="",
                parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                handler=_get_user,
            )
        )
        reg.register(
            ToolSpec(
                name="plar_query_experiments",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"category": {"type": "string"}, "take": {"type": "integer"}, "user_id": {"type": ["string", "null"]}},
                    "required": ["category"],
                },
                handler=_qe,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TUL","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"plar_get_user","hint":"get user goodenough"},{"id":"s2","tool":"plar_query_experiments","hint":"latest by that user"}]}',
                    tool_calls=[],
                    raw={},
                ),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_get_user", "arguments": {"name": "goodenough"}}}],
                    raw={},
                ),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_query_experiments", "arguments": {"category": "Experiment", "take": 1}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )

        out = agent.handle(user_text="goodenough 的最新实验是什么")
        self.assertEqual(out["answer"], "ok")
        got_args = out["tool_results"][1].data["args"]
        self.assertEqual(got_args.get("user_id"), "c" * 24)

    def test_execute_this_user_latest_featured_discussion_uses_target_id_and_exchange_tag(self):
        reg = ToolRegistry()

        def _qe(_rt: ToolRuntime, args: dict) -> dict:
            return {"args": dict(args)}

        reg.register(
            ToolSpec(
                name="plar_query_experiments",
                description="",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "take": {"type": "integer"},
                        "user_id": {"type": ["string", "null"]},
                        "tags": {"type": ["array", "null"]},
                    },
                    "required": ["category"],
                },
                handler=_qe,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        author_id = "a" * 24
        target_id = "b" * 24
        agent = _mk_agent(
            planner_resps=[
                # Planner forgets tools -> _plan_from_obj patch should force QueryExperiments.
                OllamaChatResponse(content='{"task_id":"TTU","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                # Executor also forgets user_id/tags and uses wrong category; execute() should correct/inject.
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {"function": {"name": "plar_query_experiments", "arguments": {"category": "Experiment", "take": 1}}}
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )

        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"User","id":"'
            + target_id
            + '"},"comment":{"id":"c1","author_id":"'
            + author_id
            + '","author_nickname":"MapMaths"}}\n\n该用户最新的物理类讨论区精选作品是什么'
        )
        out = agent.handle(user_text=user_text)
        self.assertEqual(out["answer"], "ok")
        got_args = out["tool_results"][0].data["args"]
        self.assertEqual(got_args.get("user_id"), target_id)
        self.assertEqual(got_args.get("category"), "Discussion")
        self.assertIn("tags", got_args)
        self.assertIn("精选", got_args["tags"])
        self.assertIn("交流", got_args["tags"])

    def test_execute_this_user_latest_featured_discussion_on_experiment_target_uses_target_owner_author_id(self):
        reg = ToolRegistry()

        owner_id = "d" * 24

        def _ctx(_rt: ToolRuntime, _args: dict) -> dict:
            return {"author": {"id": owner_id, "nickname": "Owner"}}

        def _qe(_rt: ToolRuntime, args: dict) -> dict:
            return {"args": dict(args)}

        reg.register(
            ToolSpec(
                name="plar_get_experiment_context",
                description="",
                parameters={"type": "object", "properties": {"summary_id": {"type": "string"}, "category": {"type": "string"}}},
                handler=_ctx,
            )
        )
        reg.register(
            ToolSpec(
                name="plar_query_experiments",
                description="",
                parameters={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "take": {"type": "integer"},
                        "user_id": {"type": ["string", "null"]},
                        "tags": {"type": ["array", "null"]},
                    },
                    "required": ["category"],
                },
                handler=_qe,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        author_id = "a" * 24
        target_id = "b" * 24  # experiment summary_id
        agent = _mk_agent(
            planner_resps=[
                # Planner forgets tools -> _plan_from_obj patch should force context->QueryExperiments.
                OllamaChatResponse(content='{"task_id":"TTX","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_get_experiment_context", "arguments": {"summary_id": target_id, "category": "Experiment"}}}],
                    raw={},
                ),
                # Executor forgets user_id/tags and uses wrong category; execute() should correct/inject.
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_query_experiments", "arguments": {"category": "Experiment", "take": 1}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )

        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"Experiment","id":"'
            + target_id
            + '"},"comment":{"id":"c1","author_id":"'
            + author_id
            + '","author_nickname":"MapMaths"}}\n\n该用户最新的物理类讨论区精选作品是什么'
        )
        out = agent.handle(user_text=user_text)
        self.assertEqual(out["answer"], "ok")
        got_args = out["tool_results"][1].data["args"]
        self.assertEqual(got_args.get("user_id"), owner_id)
        self.assertEqual(got_args.get("category"), "Discussion")
        self.assertIn("tags", got_args)
        self.assertIn("精选", got_args["tags"])
        self.assertIn("交流", got_args["tags"])

    def test_writer_markdown_is_formatted_to_plain_text(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"TMD","user_lang":"en","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="# Title\n\n- item1\n- item2", tool_calls=[], raw={}),
                OllamaChatResponse(content="Title\n- item1\n- item2", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        out = agent.handle(user_text="format it")
        self.assertEqual(out["answer"], "Title\n- item1\n- item2")

    def test_fallback_guard_refuses_ungrounded_community_data_answer(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[OllamaChatResponse(content="", tool_calls=[], raw={}), OllamaChatResponse(content="", tool_calls=[], raw={})],
            executor_resps=[
                # executor(json-plan) attempt (invalid -> fallback)
                OllamaChatResponse(content="not json", tool_calls=[], raw={}),
                # final fallback answer (ungrounded)
                OllamaChatResponse(
                    content="根据提供的信息，最后一个发帖的人是揉碎星月。",
                    tool_calls=[],
                    raw={},
                ),
            ],
            registry=reg,
        )

        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"User","id":"'
            + ("b" * 24)
            + '"},"comment":{"id":"c1","author_id":"'
            + ("a" * 24)
            + '","author_nickname":"揉碎星月"}}\n\n紫兰斋发布的第一个实验的评论区的第一个人是谁'
        )
        out = agent.handle(user_text=user_text)
        self.assertIn("无法", out["answer"])
        self.assertIn("请提供", out["answer"])

    def test_my_id_query_returns_comment_author_id_not_target_id(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        author_id = "a" * 24
        target_id = "b" * 24
        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"TID","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                # writer outputs the WRONG id (target id) -> should be corrected deterministically.
                OllamaChatResponse(content=f"你的用户ID是：{target_id}", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"User","id":"'
            + target_id
            + '"},"comment":{"id":"c1","author_id":"'
            + author_id
            + '","author_nickname":"揉碎星月"}}\n\n我的id是什么'
        )
        out = agent.handle(user_text=user_text)
        self.assertEqual(out["answer"], f"你的用户ID是：{author_id}")

    def test_follow_query_uses_check_tool_and_returns_deterministic_conclusion(self):
        reg = ToolRegistry()

        def _check(_rt: ToolRuntime, args: dict) -> dict:
            return {
                "follower": {"id": "0" * 24, "nickname": args.get("follower_name")},
                "followee": {"id": "1" * 24, "nickname": args.get("followee_name")},
                "is_following": True,
                "matched": {"id": "1" * 24, "nickname": args.get("followee_name")},
                "checked": {"pages_scanned": 1, "items_scanned": 1, "incomplete": False},
            }

        reg.register(
            ToolSpec(
                name="plar_check_following",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"follower_name": {"type": "string"}, "followee_name": {"type": "string"}},
                    "required": ["follower_name", "followee_name"],
                },
                handler=_check,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                # Planner forgets tools -> plan patched should force plar_check_following.
                OllamaChatResponse(content='{"task_id":"TF","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                # Writer tries to be vague; deterministic conclusion should override.
                OllamaChatResponse(content="无法确认。", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {"function": {"name": "plar_check_following", "arguments": {"follower_name": "goodenough", "followee_name": "MapMaths"}}}
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )
        out = agent.handle(user_text="用户goodenough有没有关注用户MapMaths")
        self.assertIn("结论：", out["answer"])
        self.assertIn("goodenough", out["answer"])
        self.assertIn("MapMaths", out["answer"])
        self.assertIn("关注了", out["answer"])

    def test_oldest_comment_query_uses_scan_result_deterministically(self):
        reg = ToolRegistry()

        def _oldest(_rt: ToolRuntime, _args: dict) -> dict:
            return {
                "target": {"type": "User", "id": "0" * 24},
                "pages_scanned": 2,
                "comments_scanned": 3,
                "incomplete": False,
                "oldest_comment": {"id": "c1", "ts_ms": 1, "author_nickname": "ydhfgdus", "author_id": "u1", "text": "难绷"},
                "comments": [{"id": "c1", "ts_ms": 1, "author_nickname": "ydhfgdus", "author_id": "u1", "text": "难绷"}],
            }

        reg.register(
            ToolSpec(
                name="plar_get_oldest_comment",
                description="",
                parameters={"type": "object", "properties": {"target_type": {"type": "string"}, "target_id": {"type": "string"}}},
                handler=_oldest,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TOC","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"plar_get_oldest_comment","hint":"scan"}]}',
                    tool_calls=[],
                    raw={},
                ),
                # writer output is ignored by deterministic post-processing.
                OllamaChatResponse(content="随便写点", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {"function": {"name": "plar_get_oldest_comment", "arguments": {"target_type": "User", "target_id": "0" * 24}}}
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )

        out = agent.handle(user_text="用户MapMaths的留言板里最早的评论是谁发布的？内容是什么")
        self.assertTrue(out["answer"].startswith("最早的一条评论"))
        self.assertIn("作者：ydhfgdus", out["answer"])
        self.assertIn("内容：难绷", out["answer"])

    def test_planner_patch_adds_oldest_comment_scan_steps(self):
        reg = ToolRegistry()

        def _get_user(_rt: ToolRuntime, args: dict) -> dict:
            return {"id": "c" * 24, "nickname": args.get("name")}

        def _oldest(_rt: ToolRuntime, _args: dict) -> dict:
            return {
                "target": {"type": "User", "id": "c" * 24},
                "pages_scanned": 1,
                "comments_scanned": 1,
                "incomplete": False,
                "oldest_comment": {"id": "c1", "ts_ms": 1, "author_nickname": "A", "author_id": "u", "text": "hi"},
                "comments": [{"id": "c1", "ts_ms": 1, "author_nickname": "A", "author_id": "u", "text": "hi"}],
            }

        reg.register(
            ToolSpec(
                name="plar_get_user",
                description="",
                parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                handler=_get_user,
            )
        )
        reg.register(
            ToolSpec(
                name="plar_get_oldest_comment",
                description="",
                parameters={"type": "object", "properties": {"target_type": {"type": "string"}, "target_id": {"type": "string"}}},
                handler=_oldest,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                # Planner forgets tools -> _plan_from_obj patch should add get_user + oldest_comment scan.
                OllamaChatResponse(content='{"task_id":"TPA","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="writer", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "plar_get_user", "arguments": {"name": "MapMaths"}}}], raw={}),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_get_oldest_comment", "arguments": {"target_type": "User", "target_id": "c" * 24}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )

        out = agent.handle(user_text="用户MapMaths的留言板里最早的评论是谁发布的？内容是什么")
        self.assertEqual(len(out["tool_results"]), 2)
        self.assertIn("作者：A", out["answer"])
        self.assertIn("内容：hi", out["answer"])

    def test_end_final_hex_error_is_sanitized_for_zh(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="dummy",
                description="",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda _rt, _args: {},
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"TEND","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"dummy","hint":"x"}]}',
                    tool_calls=[],
                    raw={},
                )
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[
                        {
                            "function": {
                                "name": "end",
                                "arguments": {
                                    "final": "The user ID provided is not a 24-hex string. Please provide a valid user ID to proceed."
                                },
                            }
                        }
                    ],
                    raw={},
                )
            ],
            registry=reg,
        )

        out = agent.handle(user_text="列出评论")
        self.assertIn("24 位十六进制 ID", out["answer"])
        self.assertNotIn("24-hex", out["answer"])

    def test_reply_prefix_is_stripped_for_llm_and_echo_is_rewritten(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )
        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(content='{"task_id":"TE","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                # writer echoes the user request (bug)
                OllamaChatResponse(content="概括实验RRR的评论区", tool_calls=[], raw={}),
                # anti-echo rewriter produces a real clarification
                OllamaChatResponse(content="我可以帮你概括评论区。请提供该实验/讨论的链接或ID。", tool_calls=[], raw={}),
            ],
            executor_resps=[],
            registry=reg,
        )
        user_text = "回复<user=bbbbbbbbbbbbbbbbbbbbbbbb>@aurex</user>: 概括实验RRR的评论区"
        out = agent.handle(user_text=user_text)
        self.assertIn("请提供", out["answer"])

    def test_executor_violates_plan_twice_raises(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="echo",
                description="",
                parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                handler=lambda _rt, args: {"x": args.get("x")},
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"T4","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"echo","hint":"x=1"}]}',
                    tool_calls=[],
                    raw={},
                )
            ],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "web_search", "arguments": {"query": "x"}}}],
                    raw={},
                ),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "web_search", "arguments": {"query": "y"}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )
        with self.assertRaises(AurexAgentError):
            agent.handle(user_text="run")

    def test_executor_end_short_circuits_writer(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="echo",
                description="",
                parameters={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                handler=lambda _rt, args: {"x": args.get("x")},
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        planner = _FakeClient(
            [
                OllamaChatResponse(
                    content='{"task_id":"T5","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"echo","hint":"x=1"}]}',
                    tool_calls=[],
                    raw={},
                )
            ]
        )
        executor = _FakeClient(
            [
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "end", "arguments": {"final": "need more info"}}}],
                    raw={},
                )
            ]
        )

        cfg = AurexConfig()
        agent = AurexAgent(cfg=cfg, config_path=os.path.join(ROOT, "dummy.json"), tools=reg)
        agent.planner_client = planner  # type: ignore[assignment]
        agent.executor_client = executor  # type: ignore[assignment]

        out = agent.handle(user_text="run")
        self.assertEqual(out["answer"], "need more info")
        self.assertEqual(len(planner.calls), 1)

    def test_plan_rejects_multiple_publish_steps(self):
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="plar_upload_sav",
                description="",
                parameters={"type": "object", "properties": {"sav_path": {"type": "string"}}, "required": ["sav_path"]},
                handler=lambda _rt, args: {"ok": True, "sav_path": args.get("sav_path")},
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                OllamaChatResponse(
                    content='{"task_id":"Tpub","user_lang":"zh","goal":"g","steps":[{"id":"s1","tool":"plar_upload_sav","hint":"a"},{"id":"s2","tool":"plar_upload_sav","hint":"b"}]}',
                    tool_calls=[],
                    raw={},
                ),
                # NL plan attempt (empty -> triggers fallback)
                OllamaChatResponse(content="", tool_calls=[], raw={}),
            ],
            executor_resps=[
                # executor(json-plan) attempt (invalid -> fallback)
                OllamaChatResponse(content="not json", tool_calls=[], raw={}),
                # final fallback answer
                OllamaChatResponse(content="fallback ok", tool_calls=[], raw={}),
            ],
            registry=reg,
        )
        out = agent.handle(user_text="publish twice")
        self.assertFalse(bool(getattr(out["plan"], "planner_ok", True)))
        self.assertEqual(out["answer"], "fallback ok")

    def test_execute_blocks_second_publish_call(self):
        reg = ToolRegistry()

        def _publish(_rt: ToolRuntime, args: dict) -> dict:
            return {"published": True, "sav_path": args.get("sav_path")}

        reg.register(
            ToolSpec(
                name="plar_upload_sav",
                description="",
                parameters={"type": "object", "properties": {"sav_path": {"type": "string"}}, "required": ["sav_path"]},
                handler=_publish,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[],
            executor_resps=[
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_upload_sav", "arguments": {"sav_path": "a.sav"}}}],
                    raw={},
                ),
                OllamaChatResponse(
                    content="",
                    tool_calls=[{"function": {"name": "plar_upload_sav", "arguments": {"sav_path": "b.sav"}}}],
                    raw={},
                ),
            ],
            registry=reg,
        )

        plan = Plan(
            task_id="Tpub2",
            user_lang="zh",
            goal="g",
            steps=[
                PlanStep(id="s1", tool="plar_upload_sav", hint="a"),
                PlanStep(id="s2", tool="plar_upload_sav", hint="b"),
            ],
        )
        tool_results, end_final = agent.execute(plan=plan, user_text="x", user=None, user_lang="zh")
        self.assertEqual(len(tool_results), 1)
        self.assertTrue(tool_results[0].ok)
        self.assertIsNotNone(end_final)

    def test_publish_workflow_is_forced_and_upload_intro_prefixed_with_user_tag(self):
        reg = ToolRegistry()

        def _gen_v(_rt: ToolRuntime, args: dict) -> dict:
            spec = str(args.get("spec") or "").strip()
            self.assertTrue(bool(spec))
            return {"verilog": "module top; endmodule", "top_module": str(args.get("top_module") or "top")}

        def _v2sav(_rt: ToolRuntime, args: dict) -> dict:
            self.assertEqual(str(args.get("verilog") or "").strip(), "module top; endmodule")
            self.assertTrue(bool(args.get("force_build")))
            return {"sav_path": "out.sav"}

        def _write(_rt: ToolRuntime, args: dict) -> dict:
            topic = str(args.get("topic") or "").strip()
            self.assertTrue(bool(topic))
            return {"title": "T", "introduction": "Intro", "tags": ["物理", "精选"]}

        upload_calls: list[dict] = []

        def _upload(_rt: ToolRuntime, args: dict) -> dict:
            upload_calls.append(dict(args))
            # Fail once to verify retry is allowed before success.
            if len(upload_calls) == 1:
                raise Exception("temporary upload failure")
            return {"ok": True, "id": "x"}

        reg.register(
            ToolSpec(
                name="llm_generate_verilog",
                description="",
                parameters={"type": "object", "properties": {"spec": {"type": "string"}, "top_module": {"type": "string"}}, "required": []},
                handler=_gen_v,
            )
        )
        reg.register(
            ToolSpec(
                name="verilog_to_sav",
                description="",
                parameters={"type": "object", "properties": {"verilog": {"type": "string"}}, "required": []},
                handler=_v2sav,
            )
        )
        reg.register(
            ToolSpec(
                name="llm_write_publish_text",
                description="",
                parameters={"type": "object", "properties": {"topic": {"type": "string"}}, "required": []},
                handler=_write,
            )
        )
        reg.register(
            ToolSpec(
                name="plar_upload_sav",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"sav_path": {"type": "string"}, "title": {"type": "string"}, "introduction": {"type": "string"}},
                    "required": ["sav_path", "title", "introduction"],
                },
                handler=_upload,
            )
        )
        reg.register(
            ToolSpec(
                name="end",
                description="",
                parameters={"type": "object", "properties": {"final": {"type": "string"}}, "required": ["final"]},
                handler=lambda _rt, args: {"final": args.get("final", "")},
            )
        )

        agent = _mk_agent(
            planner_resps=[
                # Planner forgets steps; _plan_from_obj should force the full publish workflow.
                OllamaChatResponse(content='{"task_id":"TPUB","user_lang":"zh","goal":"g","steps":[]}', tool_calls=[], raw={}),
                OllamaChatResponse(content="ok", tool_calls=[], raw={}),
            ],
            executor_resps=[
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "llm_generate_verilog", "arguments": {}}}], raw={}),
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "verilog_to_sav", "arguments": {}}}], raw={}),
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "llm_write_publish_text", "arguments": {}}}], raw={}),
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "plar_upload_sav", "arguments": {}}}], raw={}),
                OllamaChatResponse(content="", tool_calls=[{"function": {"name": "plar_upload_sav", "arguments": {}}}], raw={}),
            ],
            registry=reg,
        )

        author_id = "a" * 24
        target_id = "b" * 24
        user_text = (
            'CONTEXT_JSON:\n{"target":{"type":"User","id":"'
            + target_id
            + '"},"comment":{"id":"c1","author_id":"'
            + author_id
            + '","author_nickname":"MapMaths"}}\n\n请发布一个实验：与门'
        )
        out = agent.handle(user_text=user_text)
        self.assertEqual(out["answer"], "ok")

        # Upload should be retried once.
        self.assertEqual(len(upload_calls), 2)
        intro = str(upload_calls[-1].get("introduction") or "")
        self.assertTrue(intro.startswith(f"<user={author_id}>@MapMaths</user>\n\n"))
        self.assertEqual(upload_calls[-1].get("category"), "Experiment")


if __name__ == "__main__":
    unittest.main()
