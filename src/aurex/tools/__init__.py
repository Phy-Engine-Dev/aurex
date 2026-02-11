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
    PLAR_CONTEXT_TOOL,
    PLAR_CHECK_FOLLOWING_TOOL,
    PLAR_GET_COMMENTS_TOOL,
    PLAR_GET_OLDEST_COMMENT_TOOL,
    PLAR_GET_USER_TOOL,
    PLAR_LIST_TAGS_TOOL,
    PLAR_OLDEST_BY_USER_TOOL,
    PLAR_QUERY_TOOL,
    PLAR_RELATIONS_TOOL,
    PLAR_STATUS_SAVE_TOOL,
    PLAR_UPLOAD_SAV_TOOL,
    plar_check_following,
    plar_list_builtin_tags,
    plar_get_comments,
    plar_get_oldest_comment,
    plar_get_experiment_context,
    plar_get_relations,
    plar_get_status_save,
    plar_get_user,
    plar_oldest_by_user,
    plar_query_experiments,
    plar_upload_sav,
)
from .registry import ToolRegistry, ToolSpec
from .web_search import WEB_SEARCH_TOOL, ddg_web_search


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

    add(WEB_SEARCH_TOOL, ddg_web_search)

    add(LOCAL_GET_TARGET_CONTEXT_TOOL, local_get_target_context)

    add(PLAR_QUERY_TOOL, plar_query_experiments)
    add(PLAR_GET_USER_TOOL, plar_get_user)
    add(PLAR_GET_COMMENTS_TOOL, plar_get_comments)
    add(PLAR_GET_OLDEST_COMMENT_TOOL, plar_get_oldest_comment)
    add(PLAR_LIST_TAGS_TOOL, plar_list_builtin_tags)
    add(PLAR_OLDEST_BY_USER_TOOL, plar_oldest_by_user)
    add(PLAR_RELATIONS_TOOL, plar_get_relations)
    add(PLAR_CHECK_FOLLOWING_TOOL, plar_check_following)
    add(PLAR_CONTEXT_TOOL, plar_get_experiment_context)
    add(PLAR_STATUS_SAVE_TOOL, plar_get_status_save)
    add(PLAR_UPLOAD_SAV_TOOL, plar_upload_sav)

    add(PHY_ENGINE_BUILD_TOOL, phy_engine_build)
    add(VERILOG_TO_SAV_TOOL, verilog_to_sav)
    add(PE_SIMULATE_TOOL, pe_simulate)

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
