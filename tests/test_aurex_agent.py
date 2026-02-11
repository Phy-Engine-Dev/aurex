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
            executor_resps=[OllamaChatResponse(content="fallback ok", tool_calls=[], raw={})],
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
            executor_resps=[OllamaChatResponse(content="fallback ok", tool_calls=[], raw={})],
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


if __name__ == "__main__":
    unittest.main()
