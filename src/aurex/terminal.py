from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class WebAPIError(RuntimeError):
    pass


class AurexWebClient:
    """Small authenticated client shared by the terminal UI and its tests."""

    def __init__(self, base_url: str, token: str = "", *, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, path: str, data: dict[str, Any] | None = None) -> Any:
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode()
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.base_url + path, data=body, headers=headers,
                          method="POST" if body is not None else "GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw).get("error")
            except Exception:
                detail = raw.decode(errors="replace") or str(exc)
            raise WebAPIError(str(detail)) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WebAPIError(str(exc)) from exc
        try:
            return json.loads(raw)
        except Exception as exc:
            raise WebAPIError("Aurex Web returned invalid JSON") from exc

    def healthy(self) -> bool:
        try:
            value = self.request("/health")
            return bool(value.get("healthy") and value.get("service") == "aurex3")
        except WebAPIError:
            return False

    def sessions(self) -> list[dict[str, Any]]:
        return self.request("/api/sessions")

    def tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.request("/api/tasks?" + urlencode({"limit": limit}))

    def task(self, task_id: str) -> dict[str, Any]:
        return self.request("/api/tasks/" + quote(task_id, safe=""))

    def events(self, session_id: str, task_id: str) -> list[dict[str, Any]]:
        query = urlencode({"run_id": task_id})
        return self.request("/api/sessions/" + quote(session_id, safe="") + "/events?" + query)

    def submit(self, text: str, *, session_id: str | None = None, publish: bool = False) -> dict[str, Any]:
        data = {"text": text, "explicit_publish_requested": bool(publish)}
        if session_id:
            path = "/api/sessions/" + quote(session_id, safe="") + "/messages"
        else:
            path = "/api/tasks"
        return self.request(path, data)

    def cancel(self, task_id: str) -> dict[str, Any]:
        return self.request("/api/tasks/" + quote(task_id, safe="") + "/cancel", {})


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(value: Any) -> str:
    return _CONTROL.sub("", str(value or "")).replace("\r", " ").replace("\n", " ").strip()


def _event_text(event: dict[str, Any]) -> str:
    kind = _clean(event.get("kind") or "event")
    data = event.get("data")
    if isinstance(data, dict):
        for key in ("text", "content", "message", "summary", "error", "name", "status"):
            value = data.get(key)
            if isinstance(value, (str, int, float, bool)) and _clean(value):
                return f"{kind}: {_clean(value)}"
        value = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        value = _clean(data)
    return f"{kind}: {_clean(value)}"


@dataclass
class DashboardState:
    api: AurexWebClient
    sessions: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    task_id: str | None = None
    notice: str = "输入问题后按 Enter；/help 查看命令"

    def refresh(self) -> None:
        self.sessions = self.api.sessions()
        self.tasks = self.api.tasks()
        ids = {str(task.get("id")) for task in self.tasks}
        if self.task_id and self.task_id not in ids:
            try:
                task = self.api.task(self.task_id)
                self.tasks.insert(0, task)
            except WebAPIError:
                self.task_id = None
        if not self.task_id and self.tasks:
            preferred = next((task for task in self.tasks if task.get("status") in {"running", "cancelling"}), self.tasks[0])
            self.select_task(str(preferred["id"]), refresh_events=False)
        if self.task_id and self.session_id:
            self.events = self.api.events(self.session_id, self.task_id)

    def select_task(self, task_id: str, *, refresh_events: bool = True) -> None:
        task = next((row for row in self.tasks if str(row.get("id")) == task_id), None)
        if task is None:
            task = self.api.task(task_id)
        self.task_id = str(task["id"])
        self.session_id = str(task["session_id"])
        if refresh_events:
            self.events = self.api.events(self.session_id, self.task_id)
        self.notice = "已选择任务 " + self.task_id

    def select_session(self, session_id: str) -> None:
        if not any(str(row.get("id")) == session_id for row in self.sessions):
            raise WebAPIError("会话不存在")
        self.session_id = session_id
        matching = next((task for task in self.tasks if str(task.get("session_id")) == session_id), None)
        self.task_id = str(matching["id"]) if matching else None
        self.events = self.api.events(session_id, self.task_id) if self.task_id else []
        self.notice = "已选择会话 " + session_id

    def submit(self, text: str, *, publish: bool = False) -> None:
        result = self.api.submit(text, session_id=self.session_id, publish=publish)
        self.session_id = str(result["session_id"])
        self.task_id = str(result["task_id"])
        self.notice = ("已提交发布任务 " if publish else "已提交任务 ") + self.task_id
        self.refresh()

    def command(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        if not line.startswith("/"):
            self.submit(line)
            return False
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command in {"/quit", "/exit"}:
            return True
        if command == "/new":
            self.session_id = self.task_id = None
            self.events = []
            self.notice = "下一条消息将创建新会话"
        elif command == "/session" and argument:
            self.select_session(argument)
        elif command == "/task" and argument:
            self.select_task(argument)
        elif command == "/cancel":
            if not self.task_id:
                raise WebAPIError("当前没有选中的任务")
            result = self.api.cancel(self.task_id)
            self.notice = "取消请求：" + _clean(result.get("status"))
            self.refresh()
        elif command == "/refresh":
            self.refresh()
            self.notice = "已刷新"
        elif command == "/publish" and argument:
            self.submit(argument, publish=True)
        elif command == "/help":
            self.notice = "/new 新会话  /session ID  /task ID  /cancel  /publish 问题  /refresh  /quit"
        else:
            raise WebAPIError("未知或缺少参数的命令；输入 /help")
        return False

    def plain_text(self) -> str:
        running = sum(row.get("status") in {"running", "cancelling"} for row in self.tasks)
        queued = sum(row.get("status") == "queued" for row in self.tasks)
        lines = ["Aurex CLI — 已连接", f"服务: running={running} queued={queued} sessions={len(self.sessions)}", "", "队列"]
        for row in self.tasks[:8]:
            lines.append(f"  [{row.get('status','?')}] {row.get('id','')}  {_clean(row.get('title'))}")
        lines.extend(["", "会话"])
        for row in self.sessions[:8]:
            lines.append(f"  [{row.get('status','?')}] {row.get('id','')}  {_clean(row.get('title'))}")
        if self.task_id:
            lines.extend(["", "当前任务 " + self.task_id])
            lines.extend("  " + _event_text(event) for event in self.events[-12:])
        return "\n".join(lines)


class TerminalApp:
    """A small OpenCode-style, sectioned terminal view backed by Aurex Web."""

    def __init__(self, state: DashboardState):
        self.state = state
        self.input = ""
        self.task_cursor = 0
        self.session_cursor = 0
        self.focus = "input"
        self.last_refresh = 0.0

    @staticmethod
    def _put(screen, row: int, col: int, text: str, width: int, attr=0) -> None:
        if row < 0 or col < 0 or width <= 0:
            return
        try:
            screen.addnstr(row, col, _clean(text), width, attr)
        except Exception:
            pass

    def _section(self, screen, row: int, title: str, width: int, attr=0) -> int:
        self._put(screen, row, 0, f"── {title} " + "─" * max(0, width - len(title) - 4), width, attr)
        return row + 1

    def draw(self, screen) -> None:
        import curses
        height, width = screen.getmaxyx()
        screen.erase()
        if height < 12 or width < 48:
            self._put(screen, 0, 0, "终端至少需要 48×12；调整窗口后自动恢复。", width, curses.A_BOLD)
            screen.refresh()
            return
        running = sum(row.get("status") in {"running", "cancelling"} for row in self.state.tasks)
        queued = sum(row.get("status") == "queued" for row in self.state.tasks)
        self._put(screen, 0, 0, " Aurex 3 CLI ", width, curses.color_pair(1) | curses.A_BOLD)
        row = self._section(screen, 2, "服务", width, curses.color_pair(2))
        self._put(screen, row, 2, f"已连接  运行 {running}  排队 {queued}  会话 {len(self.state.sessions)}", width - 2)
        row += 2

        row = self._section(screen, row, "任务队列" + (" [Tab/↑↓/Enter]" if self.focus == "tasks" else ""), width, curses.color_pair(2))
        task_rows = min(4, max(1, len(self.state.tasks)))
        for index, task in enumerate(self.state.tasks[:task_rows]):
            marker = ">" if self.focus == "tasks" and index == self.task_cursor else " "
            selected = "*" if str(task.get("id")) == self.state.task_id else " "
            text = f"{marker}{selected} {str(task.get('status','?')):11} {str(task.get('id',''))[:12]}  {_clean(task.get('title'))}"
            self._put(screen, row + index, 1, text, width - 2, curses.A_REVERSE if marker == ">" else 0)
        row += task_rows + 1

        row = self._section(screen, row, "会话" + (" [Tab/↑↓/Enter]" if self.focus == "sessions" else ""), width, curses.color_pair(2))
        session_rows = min(3, max(1, len(self.state.sessions)))
        for index, session in enumerate(self.state.sessions[:session_rows]):
            marker = ">" if self.focus == "sessions" and index == self.session_cursor else " "
            selected = "*" if str(session.get("id")) == self.state.session_id else " "
            text = f"{marker}{selected} {str(session.get('status','?')):11} {str(session.get('id',''))[:12]}  {_clean(session.get('title'))}"
            self._put(screen, row + index, 1, text, width - 2, curses.A_REVERSE if marker == ">" else 0)
        row += session_rows + 1

        row = self._section(screen, row, "当前任务时间线", width, curses.color_pair(2))
        footer_rows = 4
        timeline_rows = max(1, height - row - footer_rows)
        for index, event in enumerate(self.state.events[-timeline_rows:]):
            self._put(screen, row + index, 2, _event_text(event), width - 3)

        input_row = height - 3
        self._section(screen, input_row - 1, "输入" + (" [Tab]" if self.focus == "input" else ""), width, curses.color_pair(2))
        prompt = "> " + self.input
        self._put(screen, input_row, 0, prompt, width - 1, curses.A_REVERSE if self.focus == "input" else 0)
        self._put(screen, height - 1, 0, self.state.notice, width - 1, curses.color_pair(3))
        if self.focus == "input":
            try:
                screen.move(input_row, min(width - 2, 2 + len(self.input)))
            except Exception:
                pass
        screen.refresh()

    def run(self, screen) -> None:
        import curses
        curses.curs_set(1)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        screen.timeout(250)
        self.state.refresh()
        while True:
            now = time.monotonic()
            if now - self.last_refresh >= 1.0:
                try:
                    self.state.refresh()
                except WebAPIError as exc:
                    self.state.notice = "刷新失败：" + str(exc)
                self.last_refresh = now
            self.draw(screen)
            try:
                key = screen.get_wch()
            except curses.error:
                continue
            if key == "\t":
                order = ["input", "tasks", "sessions"]
                self.focus = order[(order.index(self.focus) + 1) % len(order)]
            elif isinstance(key, int) and key in {curses.KEY_UP, curses.KEY_DOWN}:
                delta = -1 if key == curses.KEY_UP else 1
                if self.focus == "tasks" and self.state.tasks:
                    self.task_cursor = (self.task_cursor + delta) % min(4, len(self.state.tasks))
                elif self.focus == "sessions" and self.state.sessions:
                    self.session_cursor = (self.session_cursor + delta) % min(3, len(self.state.sessions))
            elif key in ("\n", "\r") or key == curses.KEY_ENTER:
                try:
                    if self.focus == "tasks" and self.state.tasks:
                        self.state.select_task(str(self.state.tasks[self.task_cursor]["id"]))
                    elif self.focus == "sessions" and self.state.sessions:
                        self.state.select_session(str(self.state.sessions[self.session_cursor]["id"]))
                    else:
                        line, self.input = self.input, ""
                        if self.state.command(line):
                            return
                except WebAPIError as exc:
                    self.state.notice = "操作失败：" + str(exc)
            elif self.focus == "input" and (key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE):
                self.input = self.input[:-1]
            elif self.focus == "input" and isinstance(key, str) and key.isprintable():
                self.input += key


__all__ = ["AurexWebClient", "DashboardState", "TerminalApp", "WebAPIError"]
