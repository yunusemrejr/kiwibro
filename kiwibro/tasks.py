"""Task and session coordination engine for Kiwibro long-running workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from .config import data_dir

log = logging.getLogger(__name__)

VALID_STATUSES = ("todo", "in_progress", "blocked", "done")
VALID_PRIORITIES = ("low", "medium", "high")


@dataclass
class TaskItem:
    id: str
    title: str
    conversation_id: str | None = None
    notes: str = ""
    priority: str = "medium"
    status: str = "todo"
    depends_on: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskItem:
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            conversation_id=data.get("conversation_id"),
            notes=data.get("notes", ""),
            priority=data.get("priority", "medium"),
            status=data.get("status", "todo"),
            depends_on=list(data.get("depends_on") or []),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


class TaskBoard:
    """Thread-safe and persisted task coordination engine."""

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskItem] = {}
        if persistence_path is not None:
            self._storage_path = Path(persistence_path).expanduser().resolve()
        else:
            try:
                self._storage_path = data_dir() / "tasks.json"
            except Exception:
                self._storage_path = Path.home() / ".kiwibro" / "tasks.json"
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self._storage_path.exists():
                return
            try:
                content = self._storage_path.read_text(encoding="utf-8").strip()
                if not content:
                    return
                raw = json.loads(content)
                if isinstance(raw, list):
                    for item in raw:
                        t = TaskItem.from_dict(item)
                        self._tasks[t.id] = t
                elif isinstance(raw, dict):
                    for item in raw.values():
                        t = TaskItem.from_dict(item)
                        self._tasks[t.id] = t
            except Exception as exc:
                log.warning("Failed to load tasks from %s: %s", self._storage_path, exc)

    def _save(self) -> None:
        with self._lock:
            try:
                self._storage_path.parent.mkdir(parents=True, exist_ok=True)
                data = [t.to_dict() for t in self._tasks.values()]
                tmp_path = self._storage_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp_path.replace(self._storage_path)
            except Exception as exc:
                log.warning("Failed to persist tasks to %s: %s", self._storage_path, exc)

    def list_tasks(
        self,
        conversation_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._tasks.values())
            if conversation_id is not None:
                items = [t for t in items if t.conversation_id == conversation_id]
            if status is not None:
                st = status.lower()
                items = [t for t in items if t.status == st]
            priority_rank = {"high": 0, "medium": 1, "low": 2}
            items.sort(key=lambda t: (priority_rank.get(t.priority, 1), t.created_at))
            return [t.to_dict() for t in items]

    def add_task(
        self,
        title: str,
        conversation_id: str | None = None,
        notes: str = "",
        priority: str = "medium",
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise ValueError("Task title cannot be empty")
        priority = (priority or "medium").lower()
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'. Must be one of: {VALID_PRIORITIES}")
        with self._lock:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            now = time.time()
            item = TaskItem(
                id=task_id,
                title=title,
                conversation_id=conversation_id,
                notes=notes or "",
                priority=priority,
                status="todo",
                depends_on=list(depends_on or []),
                created_at=now,
                updated_at=now,
            )
            self._tasks[task_id] = item
            self._save()
            return item.to_dict()

    def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        notes: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task with id '{task_id}' not found")
            item = self._tasks[task_id]
            if status is not None:
                st = status.lower()
                if st not in VALID_STATUSES:
                    raise ValueError(f"Invalid status '{status}'. Must be one of: {VALID_STATUSES}")
                item.status = st
            if priority is not None:
                pr = priority.lower()
                if pr not in VALID_PRIORITIES:
                    raise ValueError(f"Invalid priority '{priority}'. Must be one of: {VALID_PRIORITIES}")
                item.priority = pr
            if title is not None:
                t = title.strip()
                if not t:
                    raise ValueError("Task title cannot be empty")
                item.title = t
            if notes is not None:
                item.notes = notes
            item.updated_at = time.time()
            self._save()
            return item.to_dict()

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save()
                return True
            return False

    def clear_tasks(
        self,
        conversation_id: str | None = None,
        only_done: bool = False,
    ) -> int:
        with self._lock:
            to_remove: list[str] = []
            for tid, t in self._tasks.items():
                if conversation_id is not None and t.conversation_id != conversation_id:
                    continue
                if only_done and t.status != "done":
                    continue
                to_remove.append(tid)
            for tid in to_remove:
                del self._tasks[tid]
            if to_remove:
                self._save()
            return len(to_remove)

    def execute(
        self,
        action: str,
        args: dict[str, Any],
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        act = (action or "").lower().strip()
        conv_id = args.get("conversation_id", conversation_id)
        try:
            if act == "list":
                tasks = self.list_tasks(conversation_id=conv_id, status=args.get("status"))
                summary = f"Listed {len(tasks)} task(s)."
                return {"ok": True, "tasks": tasks, "summary": summary}
            elif act == "add":
                task = self.add_task(
                    title=args.get("title", ""),
                    conversation_id=conv_id,
                    notes=args.get("notes", ""),
                    priority=args.get("priority", "medium"),
                    depends_on=args.get("depends_on"),
                )
                tasks = self.list_tasks(conversation_id=conv_id)
                return {"ok": True, "tasks": tasks, "summary": f"Created task '{task['title']}' ({task['id']})."}
            elif act == "update":
                task = self.update_task(
                    task_id=args.get("task_id", ""),
                    status=args.get("status"),
                    title=args.get("title"),
                    notes=args.get("notes"),
                    priority=args.get("priority"),
                )
                tasks = self.list_tasks(conversation_id=conv_id)
                return {"ok": True, "tasks": tasks, "summary": f"Updated task '{task['id']}' to {task['status']}."}
            elif act == "delete":
                task_id = args.get("task_id", "")
                deleted = self.delete_task(task_id=task_id)
                tasks = self.list_tasks(conversation_id=conv_id)
                if deleted:
                    return {"ok": True, "tasks": tasks, "summary": f"Deleted task '{task_id}'."}
                return {"ok": False, "tasks": tasks, "summary": f"Task '{task_id}' not found."}
            elif act == "clear":
                count = self.clear_tasks(conversation_id=conv_id, only_done=bool(args.get("only_done", False)))
                tasks = self.list_tasks(conversation_id=conv_id)
                return {"ok": True, "tasks": tasks, "summary": f"Cleared {count} task(s)."}
            elif act == "summary":
                tasks = self.list_tasks(conversation_id=conv_id)
                counts = {"todo": 0, "in_progress": 0, "blocked": 0, "done": 0}
                for t in tasks:
                    st = t.get("status", "todo")
                    if st in counts:
                        counts[st] += 1
                lines = [
                    f"Task Summary ({len(tasks)} total: {counts['todo']} todo, "
                    f"{counts['in_progress']} in_progress, {counts['blocked']} blocked, {counts['done']} done):"
                ]
                for t in tasks:
                    lines.append(
                        f"- [{t['status'].upper()}] {t['title']} ({t['id']}, priority: {t['priority']})"
                    )
                return {"ok": True, "tasks": tasks, "summary": "\n".join(lines)}
            else:
                return {
                    "ok": False,
                    "tasks": [],
                    "summary": f"Unknown action '{action}'. Valid actions: ('list', 'add', 'update', 'delete', 'clear', 'summary').",
                }
        except Exception as exc:
            return {"ok": False, "tasks": [], "summary": f"Task execution failed: {exc}"}

    def evaluate_deliverable(
        self,
        path_or_text: str | Path,
        requirements: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify an artifact or code against quality checklists and syntax requirements."""
        import ast

        text: str
        p = Path(path_or_text) if isinstance(path_or_text, (str, Path)) else None
        if p and p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(path_or_text)

        lazy_patterns = [
            r"#\s*\.\.\.\s*(rest|existing|remaining|code)",
            r"//\s*\.\.\.\s*(rest|existing|remaining|code)",
            r"//\s*TODO:?\s*(keep|rest|existing)",
            r"#\s*TODO:?\s*(keep|rest|existing)",
            r"/\*\s*\.\.\.\s*(rest|existing|code)?\s*\*/",
        ]

        issues: list[str] = []
        gates: dict[str, bool] = {
            "evidence_collected": True,
            "code_verified": True,
            "no_placeholders": True,
            "requirements_met": True,
        }

        for pat in lazy_patterns:
            if re.search(pat, text, re.I):
                gates["no_placeholders"] = False
                issues.append(f"Contains lazy placeholder: '{pat}'")
                break

        if "def " in text or "class " in text or (p and p.suffix == ".py"):
            try:
                ast.parse(text)
            except SyntaxError as exc:
                gates["code_verified"] = False
                issues.append(f"Python syntax error at line {exc.lineno}: {exc.msg}")

        if requirements:
            lower_text = text.lower()
            unmet = [req for req in requirements if req.lower() not in lower_text]
            if unmet:
                gates["requirements_met"] = False
                issues.append(f"Unmet requirements: {', '.join(unmet)}")

        passed = all(gates.values())
        return {
            "passed": passed,
            "gates": gates,
            "issues": issues,
        }


TASK_BOARD = TaskBoard()


def _exec_task_board(args: dict[str, Any], ctx: Any) -> Any:
    from .extensions import tool_result

    action = args.get("action", "list")
    conv_id = args.get("conversation_id") or getattr(ctx, "conversation_id", None)
    res = TASK_BOARD.execute(action, args, conversation_id=conv_id)
    return tool_result(
        ok=bool(res.get("ok", True)),
        content=str(res.get("summary", "")),
        meta=res,
    )


def _task_board_summary(args: dict[str, Any]) -> str:
    act = args.get("action", "list")
    title = args.get("title") or args.get("task_id") or ""
    return f"task_board: {act} {title}".strip()


from .extensions import ExtensionTool, schema

TOOLS = (
    ExtensionTool(
        name="task_board",
        description="Coordinate and manage long-running tasks, milestones, and session todo items.",
        parameters=schema(
            properties={
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "update", "delete", "clear", "summary"],
                    "description": "Action to perform on the task board.",
                },
                "title": {
                    "type": "string",
                    "description": "Task title (required for 'add', optional for 'update').",
                },
                "task_id": {
                    "type": "string",
                    "description": "Unique task ID (required for 'update' and 'delete').",
                },
                "status": {
                    "type": "string",
                    "enum": ["todo", "in_progress", "blocked", "done"],
                    "description": "Filter for 'list' or new status for 'update'.",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Priority level ('low', 'medium', 'high').",
                },
                "notes": {
                    "type": "string",
                    "description": "Notes or context for the task.",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of prerequisite tasks.",
                },
                "only_done": {
                    "type": "boolean",
                    "description": "If true, only clears done tasks.",
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation/session ID.",
                },
            },
            required=["action"],
        ),
        risk="read",
        phase="tool",
        label="Task Board",
        summary=_task_board_summary,
        validate=lambda args: args,
        execute=_exec_task_board,
    ),
)

