from __future__ import annotations

import json
import json.decoder
import mimetypes
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTORESEARCH_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = DASHBOARD_ROOT / "static"
METASTUDIES_ROOT = REPO_ROOT / "metastudies"
CODEX_ROOT = Path.home() / ".codex"
SESSIONS_ROOT = CODEX_ROOT / "sessions"
SHELL_SNAPSHOTS_ROOT = CODEX_ROOT / "shell_snapshots"

SESSION_ID_PATTERN = re.compile(r"([0-9a-f]{8}-[0-9a-f-]{27,})", re.IGNORECASE)


@dataclass(slots=True)
class RunInvocation:
    index: int
    role: str
    target_path: str
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    status: str = "pending"
    action: str | None = None
    exit_code: int | None = None
    duration_seconds: int | None = None
    session_id: str | None = None
    resume_command: str | None = None
    info_lines: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": f"run-{self.index}",
            "index": self.index,
            "role": self.role,
            "target_path": self.target_path,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "status": self.status,
            "action": self.action,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "session_id": self.session_id,
            "resume_command": self.resume_command,
            "info_lines": self.info_lines or [],
            "level": classify_target(Path(self.target_path))[0],
            "target_name": Path(self.target_path).name,
        }


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def text_response(
    handler: BaseHTTPRequestHandler,
    payload: str,
    *,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    body = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def iso_to_display(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


def normalize_query_path(query_value: str | None, *, allow_missing: bool = False) -> Path:
    if not query_value:
        raise ValueError("missing path")
    path = Path(query_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not allow_missing and not path.exists():
        raise FileNotFoundError(str(path))
    return path


def ensure_under(path: Path, roots: list[Path]) -> Path:
    for root in roots:
        try:
            path.relative_to(root)
            return path
        except ValueError:
            continue
    raise PermissionError(f"path not allowed: {path}")


def query_flag(query: dict[str, list[str]], key: str, *, default: bool = False) -> bool:
    value = query.get(key, [None])[0]
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def read_last_lines(path: Path, lines: int = 200) -> str:
    if lines <= 0:
        return ""

    chunk_size = 64 * 1024
    newline_count = 0
    chunks: list[bytes] = []

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newline_count <= lines:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return "".join(text.splitlines(keepends=True)[-lines:])


def read_file_preview(path: Path, *, lines: int | None = None, max_bytes: int = 1_000_000) -> str:
    if lines is not None:
        return read_last_lines(path, lines=lines)
    raw = path.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def iter_jsonl_records(
    path: Path, *, ignore_incomplete_final_record: bool = False
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        while True:
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line))
            except json.decoder.JSONDecodeError:
                if (
                    ignore_incomplete_final_record
                    and not raw_line.endswith(b"\n")
                    and handle.peek(1) == b""
                ):
                    break
                raise
    return records


def parse_extra(extra: str) -> tuple[int | None, int | None, str | None]:
    exit_code: int | None = None
    duration_seconds: int | None = None
    session_id: str | None = None
    for field in extra.split("\t"):
        if field.startswith("exit:"):
            try:
                exit_code = int(field.split(":", 1)[1])
            except ValueError:
                exit_code = None
        elif field.endswith("s") and field[:-1].isdigit():
            duration_seconds = int(field[:-1])
        else:
            match = SESSION_ID_PATTERN.search(field)
            if match:
                session_id = match.group(1)
    return exit_code, duration_seconds, session_id


def parse_run_log(log_path: Path) -> list[RunInvocation]:
    invocations: list[RunInvocation] = []
    active: dict[tuple[str, str], list[int]] = {}

    if not log_path.exists():
        return invocations

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            timestamp = parts[0].strip()
            action = parts[1].strip()
            role = parts[2].strip()
            target_path = parts[3].strip()
            extra = "\t".join(parts[4:]).strip()
            key = (role, target_path)

            if action == "START":
                invocation = RunInvocation(
                    index=len(invocations),
                    role=role,
                    target_path=target_path,
                    start_timestamp=timestamp,
                    status="running",
                    action="START",
                    info_lines=[],
                )
                invocations.append(invocation)
                active.setdefault(key, []).append(invocation.index)
                continue

            if action == "INFO":
                session_id = None
                if extra:
                    match = SESSION_ID_PATTERN.search(extra)
                    if match:
                        session_id = match.group(1)
                candidate_indexes = active.get(key) or [
                    invocation.index
                    for invocation in reversed(invocations)
                    if invocation.role == role and invocation.target_path == target_path
                ]
                if candidate_indexes:
                    invocation = invocations[candidate_indexes[-1]]
                else:
                    invocation = RunInvocation(
                        index=len(invocations),
                        role=role,
                        target_path=target_path,
                        status="info",
                        action="INFO",
                        info_lines=[],
                    )
                    invocations.append(invocation)
                invocation.info_lines = invocation.info_lines or []
                invocation.info_lines.append(extra)
                if session_id and not invocation.session_id:
                    invocation.session_id = session_id
                    invocation.resume_command = extra
                continue

            if action in {"DONE", "FAIL"}:
                exit_code, duration_seconds, session_id = parse_extra(extra)
                active_indexes = active.get(key, [])
                if active_indexes:
                    invocation = invocations[active_indexes.pop(0)]
                    if not active_indexes:
                        active.pop(key, None)
                else:
                    invocation = RunInvocation(
                        index=len(invocations),
                        role=role,
                        target_path=target_path,
                        status="orphaned",
                        action=action,
                        info_lines=[],
                    )
                    invocations.append(invocation)
                invocation.end_timestamp = timestamp
                invocation.status = "succeeded" if action == "DONE" else "failed"
                invocation.action = action
                invocation.exit_code = exit_code
                invocation.duration_seconds = duration_seconds
                if session_id and not invocation.session_id:
                    invocation.session_id = session_id
                continue

    return invocations


def classify_target(path: Path) -> tuple[str, dict[str, str]]:
    parts = path.parts
    if "metastudies" not in parts:
        return "external", {}
    pivot = parts.index("metastudies")
    metadata: dict[str, str] = {}
    if pivot + 1 < len(parts):
        metadata["metastudy"] = parts[pivot + 1]
    if "studies" in parts[pivot + 1 :]:
        study_index = parts.index("studies", pivot + 1)
        if study_index + 1 < len(parts):
            metadata["study"] = parts[study_index + 1]
    if "substudies" in parts[pivot + 1 :]:
        substudy_index = parts.index("substudies", pivot + 1)
        if substudy_index + 1 < len(parts):
            metadata["substudy"] = parts[substudy_index + 1]

    if "substudy" in metadata:
        return "substudy", metadata
    if "study" in metadata:
        return "study", metadata
    return "metastudy", metadata


def build_agent_tree(metastudy_path: Path, invocations: list[RunInvocation]) -> dict[str, Any]:
    metastudy_name = metastudy_path.name
    root: dict[str, Any] = {
        "id": f"group:{metastudy_path}",
        "kind": "group",
        "group_type": "metastudy",
        "name": metastudy_name,
        "path": str(metastudy_path),
        "children": [],
    }
    groups: dict[str, dict[str, Any]] = {str(metastudy_path): root}

    def ensure_group(path: Path) -> dict[str, Any]:
        path_key = str(path)
        existing = groups.get(path_key)
        if existing is not None:
            return existing

        level, metadata = classify_target(path)
        if level == "substudy":
            parent_path = path.parent.parent
        elif level == "study":
            parent_path = metastudy_path
        else:
            parent_path = metastudy_path

        parent_group = ensure_group(parent_path)
        group = {
            "id": f"group:{path}",
            "kind": "group",
            "group_type": level,
            "name": path.name,
            "path": path_key,
            "metadata": metadata,
            "children": [],
        }
        parent_group["children"].append(group)
        groups[path_key] = group
        return group

    relevant = [
        invocation
        for invocation in invocations
        if Path(invocation.target_path) == metastudy_path
        or metastudy_path in Path(invocation.target_path).parents
    ]

    relevant.sort(
        key=lambda item: (
            item.start_timestamp or item.end_timestamp or "",
            item.index,
        )
    )

    for invocation in relevant:
        target_path = Path(invocation.target_path)
        group = ensure_group(target_path)
        group["children"].append(
            {
                "id": f"run:{invocation.index}",
                "kind": "run",
                "path": invocation.target_path,
                "role": invocation.role,
                "status": invocation.status,
                "start_timestamp": invocation.start_timestamp,
                "end_timestamp": invocation.end_timestamp,
                "start_display": iso_to_display(invocation.start_timestamp),
                "end_display": iso_to_display(invocation.end_timestamp),
                "duration_seconds": invocation.duration_seconds,
                "exit_code": invocation.exit_code,
                "session_id": invocation.session_id,
                "resume_command": invocation.resume_command,
                "target_name": target_path.name,
                "invocation": invocation.to_dict(),
            }
        )

    def sort_children(node: dict[str, Any]) -> None:
        groups_list = [child for child in node["children"] if child["kind"] == "group"]
        runs_list = [child for child in node["children"] if child["kind"] == "run"]
        groups_list.sort(key=lambda item: (item["group_type"], item["name"]))
        runs_list.sort(
            key=lambda item: (
                item["start_timestamp"] or item["end_timestamp"] or "",
                item["id"],
            )
        )
        node["children"] = groups_list + runs_list
        for child in groups_list:
            sort_children(child)

    sort_children(root)
    return root


def file_tree(path: Path, *, depth: int = 0, max_depth: int = 7) -> dict[str, Any]:
    stat = path.stat()
    node: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "type": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "modified_timestamp": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if path.is_file() or depth >= max_depth:
        return node

    children: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.name.startswith("."):
            continue
        children.append(file_tree(child, depth=depth + 1, max_depth=max_depth))
    node["children"] = children
    return node


@lru_cache(maxsize=1)
def discover_session_files() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not SESSIONS_ROOT.exists():
        return mapping
    for path in SESSIONS_ROOT.rglob("*.jsonl"):
        match = SESSION_ID_PATTERN.search(path.name)
        if match:
            mapping[match.group(1)] = str(path)
    return mapping


def refresh_session_cache() -> dict[str, str]:
    discover_session_files.cache_clear()
    return discover_session_files()


def session_file_for(session_id: str) -> Path | None:
    mapping = discover_session_files()
    path = mapping.get(session_id)
    if path is None:
        mapping = refresh_session_cache()
        path = mapping.get(session_id)
    return Path(path) if path else None


def shell_snapshots_for(session_id: str) -> list[str]:
    if not SHELL_SNAPSHOTS_ROOT.exists():
        return []
    snapshots = sorted(str(path) for path in SHELL_SNAPSHOTS_ROOT.glob(f"{session_id}*.sh"))
    return snapshots


def extract_message_text(content: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in content:
        if "text" in item and isinstance(item["text"], str):
            chunks.append(item["text"])
        else:
            chunks.append(json.dumps(item, indent=2))
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())


def parse_session(session_id: str) -> dict[str, Any]:
    session_path = session_file_for(session_id)
    if session_path is None or not session_path.exists():
        raise FileNotFoundError(f"session not found: {session_id}")

    items: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "path": str(session_path),
        "shell_snapshots": shell_snapshots_for(session_id),
    }

    for record in iter_jsonl_records(session_path, ignore_incomplete_final_record=True):
        record_type = record.get("type")
        payload = record.get("payload", {})

        if record_type == "session_meta":
            metadata["timestamp"] = payload.get("timestamp")
            metadata["cwd"] = payload.get("cwd")
            metadata["originator"] = payload.get("originator")
            metadata["model_provider"] = payload.get("model_provider")
            metadata["source"] = payload.get("source")
            metadata["cli_version"] = payload.get("cli_version")
            continue

        if record_type != "response_item":
            continue

        payload_type = payload.get("type")
        if payload_type == "message":
            text = extract_message_text(payload.get("content", []))
            if not text.strip():
                continue
            hidden_by_default = payload.get("role") in {"developer", "system"} or (
                payload.get("role") == "user" and len(text) > 4_000
            )
            items.append(
                {
                    "kind": "message",
                    "role": payload.get("role", "unknown"),
                    "phase": payload.get("phase"),
                    "text": text,
                    "hidden_by_default": hidden_by_default,
                }
            )
            continue

        if payload_type == "function_call":
            items.append(
                {
                    "kind": "tool_call",
                    "tool_name": payload.get("name", "unknown"),
                    "call_id": payload.get("call_id"),
                    "text": payload.get("arguments", ""),
                    "hidden_by_default": False,
                }
            )
            continue

        if payload_type == "function_call_output":
            items.append(
                {
                    "kind": "tool_output",
                    "call_id": payload.get("call_id"),
                    "text": payload.get("output", ""),
                    "hidden_by_default": False,
                }
            )
            continue

        if payload_type == "custom_tool_call":
            items.append(
                {
                    "kind": "tool_call",
                    "tool_name": payload.get("name", "custom_tool"),
                    "call_id": payload.get("call_id"),
                    "text": payload.get("input", ""),
                    "hidden_by_default": False,
                }
            )
            continue

        if payload_type == "custom_tool_call_output":
            items.append(
                {
                    "kind": "tool_output",
                    "call_id": payload.get("call_id"),
                    "text": payload.get("output", ""),
                    "hidden_by_default": False,
                }
            )

    return {
        "metadata": metadata,
        "items": items,
        "raw_text": read_file_preview(session_path, max_bytes=1_500_000),
    }


def metastudy_summary(metastudy_path: Path) -> dict[str, Any]:
    run_log = metastudy_path / "run.log"
    invocations = parse_run_log(run_log)
    statuses = {"running": 0, "succeeded": 0, "failed": 0}
    for invocation in invocations:
        statuses.setdefault(invocation.status, 0)
        statuses[invocation.status] += 1

    objective_path = metastudy_path / "objective.md"
    study_dirs = sorted(
        path for path in (metastudy_path / "studies").glob("*") if path.is_dir()
    ) if (metastudy_path / "studies").exists() else []
    return {
        "name": metastudy_path.name,
        "path": str(metastudy_path),
        "objective_exists": objective_path.exists(),
        "run_log_exists": run_log.exists(),
        "study_count": len(study_dirs),
        "status_counts": statuses,
        "latest_run_timestamp": max(
            (
                invocation.end_timestamp or invocation.start_timestamp
                for invocation in invocations
                if invocation.end_timestamp or invocation.start_timestamp
            ),
            default=None,
        ),
        "latest_run_display": iso_to_display(
            max(
                (
                    invocation.end_timestamp or invocation.start_timestamp
                    for invocation in invocations
                    if invocation.end_timestamp or invocation.start_timestamp
                ),
                default=None,
            )
        ),
    }


def list_metastudies() -> list[dict[str, Any]]:
    if not METASTUDIES_ROOT.exists():
        return []
    metastudies = [
        path
        for path in sorted(METASTUDIES_ROOT.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    summaries = [metastudy_summary(path) for path in metastudies]
    summaries.sort(
        key=lambda item: (
            item["latest_run_timestamp"] or "",
            item["name"],
        ),
        reverse=True,
    )
    return summaries


def important_files_for_target(target_path: Path) -> list[dict[str, str]]:
    candidates = [
        target_path / "objective.md",
        target_path / "plan.md",
        target_path / "state.md",
        target_path / "report.md",
        target_path / "review.md",
        target_path / "config.py",
        target_path / "training.log",
    ]
    files: list[dict[str, str]] = []
    for path in candidates:
        if path.exists() and path.is_file():
            files.append({"label": path.name, "path": str(path)})
    return files


def metastudy_payload(path: Path, *, include_file_tree: bool = True) -> dict[str, Any]:
    run_log = path / "run.log"
    invocations = parse_run_log(run_log)
    agent_tree = build_agent_tree(path, invocations)
    run_invocations = [invocation.to_dict() for invocation in invocations]
    for invocation in run_invocations:
        invocation["important_files"] = important_files_for_target(Path(invocation["target_path"]))
        if invocation["session_id"]:
            invocation["session_path"] = str(session_file_for(invocation["session_id"])) if session_file_for(invocation["session_id"]) else None
    payload = {
        "summary": metastudy_summary(path),
        "paths": {
            "metastudy": str(path),
            "run_log": str(run_log),
            "readme": str(AUTORESEARCH_ROOT / "README.md"),
        },
        "agent_tree": agent_tree,
        "invocations": run_invocations,
        "run_log_tail": read_file_preview(run_log, lines=200) if run_log.exists() else "",
    }
    if include_file_tree:
        payload["file_tree"] = file_tree(path)
    return payload


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "AutoresearchDashboard/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/":
                return self.serve_static("index.html")
            if path == "/favicon.ico":
                return self.serve_static("favicon.svg")
            if path.startswith("/static/"):
                return self.serve_static(path.removeprefix("/static/"))
            if path == "/api/health":
                return json_response(
                    self,
                    {
                        "ok": True,
                        "repo_root": str(REPO_ROOT),
                        "metastudies_root": str(METASTUDIES_ROOT),
                    },
                )
            if path == "/api/metastudies":
                return json_response(self, {"metastudies": list_metastudies()})
            if path == "/api/metastudy":
                target = normalize_query_path(query.get("path", [None])[0])
                ensure_under(target, [METASTUDIES_ROOT])
                include_file_tree = query_flag(query, "include_file_tree", default=True)
                return json_response(
                    self,
                    metastudy_payload(target, include_file_tree=include_file_tree),
                )
            if path == "/api/file":
                requested = normalize_query_path(query.get("path", [None])[0])
                ensure_under(requested, [METASTUDIES_ROOT, SESSIONS_ROOT, SHELL_SNAPSHOTS_ROOT, AUTORESEARCH_ROOT])
                lines_param = query.get("lines", [None])[0]
                lines = int(lines_param) if lines_param else None
                return json_response(
                    self,
                    {
                        "path": str(requested),
                        "content": read_file_preview(requested, lines=lines),
                    },
                )
            if path == "/api/session":
                session_id = query.get("id", [None])[0]
                if not session_id:
                    raise ValueError("missing id")
                return json_response(self, parse_session(session_id))
        except PermissionError as exc:
            return json_response(self, {"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as exc:
            return json_response(self, {"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            return json_response(
                self,
                {"error": f"internal error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        json_response(self, {"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[dashboard] {fmt % args}\n")

    def serve_static(self, relative_path: str) -> None:
        asset_path = (STATIC_ROOT / relative_path).resolve()
        ensure_under(asset_path, [STATIC_ROOT])
        if not asset_path.exists() or not asset_path.is_file():
            json_response(self, {"error": "static asset not found"}, status=HTTPStatus.NOT_FOUND)
            return
        mime_type, _ = mimetypes.guess_type(asset_path.name)
        content_type = mime_type or "application/octet-stream"
        body = asset_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"} else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = os.environ.get("AUTORESEARCH_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTORESEARCH_DASHBOARD_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Autoresearch dashboard serving at http://{host}:{port}", flush=True)
    print(f"Watching metastudies root: {METASTUDIES_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
