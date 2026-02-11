from __future__ import annotations

from typing import Any

from ..jsonutil import JsonExtractError, extract_first_json
from ..ollama import message
from .registry import ToolError, ToolRuntime


def _require_planner(runtime: ToolRuntime):
    if runtime.planner_client is None:
        raise ToolError("llm tool requires runtime.planner_client")
    return runtime.planner_client


def llm_generate_verilog(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, str]:
    spec = str(args.get("spec") or "").strip()
    if not spec:
        raise ToolError("llm_generate_verilog: spec is empty")
    top = str(args.get("top_module") or "top").strip() or "top"

    client = _require_planner(runtime)
    sys = (
        "你是硬件设计助手。\n"
        "任务：根据用户需求生成可编译的 Verilog-2001 源码。\n"
        "严格输出规则：\n"
        "- 只输出 Verilog 源码本体；不要解释；不要 Markdown code fence；不要前后缀文字。\n"
        "- 使用模块名：" + top + "。\n"
        "- 避免 SystemVerilog 语法（如 logic/always_ff/interface/package）。\n"
        "- 端口与参数要自洽；如需要时钟/复位，请显式端口。\n"
    )
    resp = client.chat(messages=[message("system", sys), message("user", spec)])
    code = (resp.content or "").strip()
    if "module" not in code or "endmodule" not in code:
        raise ToolError("llm_generate_verilog: model output does not look like Verilog")
    return {"top_module": top, "verilog": code}


def llm_write_publish_text(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    topic = str(args.get("topic") or "").strip()
    if not topic:
        raise ToolError("llm_write_publish_text: topic is empty")
    verilog = str(args.get("verilog") or "").strip()
    extra = str(args.get("extra") or "").strip()

    client = _require_planner(runtime)
    lang_code = (runtime.user_lang or "").strip().lower() or "en"

    sys = (
        "你是 Physics Lab AR 社区的实验发布编辑。\n"
        "只输出严格 JSON 对象：{title:introduction:tags?}。\n"
        "要求：\n"
        "- 不要输出任何非 JSON 文本。\n"
        "- title：短标题，<=40字（中文）或 <=80 chars（英文）。\n"
        "- introduction：发布简介/报告摘要，<=800字（中文）或 <=1200 chars（英文）。\n"
        "- tags：可选，字符串数组；仅在非常确定时给出。\n"
        "- 语言：必须与用户语言一致（user_lang={lang_code}）。若以 zh 开头则用中文，否则用对应语言。\n"
        "- 不要包含 @aurex。\n"
    )
    user = "需求：\n" + topic
    if verilog:
        user += "\n\nVerilog（仅供参考，勿逐行解释）：\n" + verilog[:6000]
    if extra:
        user += "\n\n补充信息：\n" + extra[:4000]

    resp = client.chat(messages=[message("system", sys), message("user", user)])
    try:
        obj = extract_first_json(resp.content)
    except JsonExtractError as e:
        raise ToolError(f"llm_write_publish_text: invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ToolError("llm_write_publish_text: JSON must be an object")

    title = obj.get("title")
    intro = obj.get("introduction")
    if not isinstance(title, str) or not title.strip():
        raise ToolError("llm_write_publish_text: missing title")
    if not isinstance(intro, str) or not intro.strip():
        raise ToolError("llm_write_publish_text: missing introduction")

    tags = obj.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            raise ToolError("llm_write_publish_text: tags must be an array of strings")

    return {"title": title.strip(), "introduction": intro.strip(), "tags": tags}


LLM_GENERATE_VERILOG_TOOL = {
    "name": "llm_generate_verilog",
    "description": "Generate Verilog-2001 code from a natural language spec (uses planner model).",
    "parameters": {
        "type": "object",
        "properties": {
            "spec": {"type": "string"},
            "top_module": {"type": "string", "default": "top"},
        },
        "required": ["spec"],
    },
}

LLM_WRITE_PUBLISH_TEXT_TOOL = {
    "name": "llm_write_publish_text",
    "description": "Write a publish-ready title/introduction (and optional tags) as strict JSON (uses planner model).",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "verilog": {"type": ["string", "null"]},
            "extra": {"type": ["string", "null"]},
        },
        "required": ["topic"],
    },
}
