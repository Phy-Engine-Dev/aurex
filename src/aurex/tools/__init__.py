from __future__ import annotations

from typing import Any

from .llm import (
    LLM_GENERATE_VERILOG_TOOL,
    LLM_WRITE_PUBLISH_TEXT_TOOL,
    llm_generate_verilog,
    llm_write_publish_text,
)
from .local_context import LOCAL_GET_TARGET_CONTEXT_TOOL, local_get_target_context
from .phy_engine import (
    PE_SIMULATE_TOOL,
    PHY_ENGINE_BUILD_TOOL,
    VERILOG_TO_SAV_TOOL,
    pe_simulate,
    phy_engine_build,
    verilog_to_sav,
)
from .plar_tools import (
    PLAR_SUMMARY_TOOL,
    PLAR_EXPERIMENT_FILE_TOOL,
    PLAR_CHECK_FOLLOWING_TOOL,
    PLAR_GET_COMMENTS_TOOL,
    PLAR_GET_OLDEST_COMMENT_TOOL,
    PLAR_GET_USER_TOOL,
    PLAR_LIST_TAGS_TOOL,
    PLAR_OLDEST_BY_USER_TOOL,
    PLAR_QUERY_TOOL,
    PLAR_RELATIONS_TOOL,
    PLAR_UPLOAD_SAV_TOOL,
    PLAR_PUBLISH_EXPERIMENT_TOOL,
    plar_check_following,
    plar_list_builtin_tags,
    plar_get_comments,
    plar_get_oldest_comment,
    plar_get_summary,
    plar_get_experiment_file,
    plar_get_relations,
    plar_get_user,
    plar_oldest_by_user,
    plar_query_experiments,
    plar_upload_sav,
    plar_publish_experiment,
)
from .registry import ToolRegistry, ToolSpec
from .web_search import WEB_FETCH_TOOL, web_fetch
from .circuits import register_circuit_tools
from .hdl import register_hdl_tools
from .hdl_workspace import register_hdl_workspace_tools
from .content import (
    PLAR_READ_BODY_TOOL,
    PLAR_READ_TITLE_TOOL,
    plar_read_body,
    plar_read_title,
)


def _end_tool(_runtime, args: dict[str, Any]) -> dict[str, Any]:
    final = args.get("final")
    if isinstance(final, str):
        return {"final": final}
    return {"final": ""}


def create_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def add(meta: dict[str, Any], handler):
        reg.register(
            ToolSpec(
                name=meta["name"],
                description=meta["description"],
                parameters=meta["parameters"],
                handler=handler,
            )
        )

    add(WEB_FETCH_TOOL, web_fetch)

    add(LOCAL_GET_TARGET_CONTEXT_TOOL, local_get_target_context)

    add(PLAR_QUERY_TOOL, plar_query_experiments)
    add(PLAR_GET_USER_TOOL, plar_get_user)
    add(PLAR_GET_COMMENTS_TOOL, plar_get_comments)
    add(PLAR_GET_OLDEST_COMMENT_TOOL, plar_get_oldest_comment)
    add(PLAR_LIST_TAGS_TOOL, plar_list_builtin_tags)
    add(PLAR_OLDEST_BY_USER_TOOL, plar_oldest_by_user)
    add(PLAR_RELATIONS_TOOL, plar_get_relations)
    add(PLAR_CHECK_FOLLOWING_TOOL, plar_check_following)
    add(PLAR_SUMMARY_TOOL, plar_get_summary)
    add(PLAR_EXPERIMENT_FILE_TOOL, plar_get_experiment_file)
    add(PLAR_UPLOAD_SAV_TOOL, plar_upload_sav)
    add(PLAR_PUBLISH_EXPERIMENT_TOOL, plar_publish_experiment)
    # Narrow prose readers.  These intentionally replace broad archive/content
    # browsing in the model-facing toolset; raw source remains durable in the
    # session database for operator diagnostics and circuit tools.
    add(PLAR_READ_TITLE_TOOL, plar_read_title)
    add(PLAR_READ_BODY_TOOL, plar_read_body)

    add(PHY_ENGINE_BUILD_TOOL, phy_engine_build)
    add(VERILOG_TO_SAV_TOOL, verilog_to_sav)
    add(PE_SIMULATE_TOOL, pe_simulate)
    register_circuit_tools(reg)
    register_hdl_tools(reg)
    register_hdl_workspace_tools(reg)

    add(LLM_GENERATE_VERILOG_TOOL, llm_generate_verilog)
    add(LLM_WRITE_PUBLISH_TEXT_TOOL, llm_write_publish_text)

    reg.register(
        ToolSpec(
            name="end",
            description="Finish tool loop early with a final message (string).",
            parameters={
                "type": "object",
                "properties": {"final": {"type": "string"}},
                "required": ["final"],
            },
            handler=_end_tool,
        )
    )

    return reg
