from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[["ToolRuntime", dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolResult:
    task_id: str
    step_id: str
    ok: bool
    data: Any | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolRuntime:
    task_id: str
    user_lang: str
    config_path: str
    config: Any
    cache_dir: str
    user: Any | None = None
    planner_client: Any | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        name = (tool.name or "").strip()
        if not name:
            raise ToolError("tool.name is empty")
        if name in self._tools:
            raise ToolError(f"Duplicate tool: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> ToolSpec:
        tool = self._tools.get((name or "").strip())
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        return tool

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def ollama_schemas(self) -> list[dict[str, Any]]:
        from ..ollama import tool_schema

        out: list[dict[str, Any]] = []
        for t in self.list():
            out.append(tool_schema(name=t.name, description=t.description, parameters=t.parameters))
        return out

