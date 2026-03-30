"""Shared utilities for autoresearch launcher scripts."""

from __future__ import annotations

import os
import pty
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent.parent          # scripts/ -> autoresearch/ -> repo root
AUTORESEARCH = Path(__file__).parent.parent        # scripts/ -> autoresearch/
# Override via AUTORESEARCH_MODEL env var if needed
MODEL = os.environ.get("AUTORESEARCH_MODEL", "gpt-5.4")
SESSION_ID_PATTERN = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(
    log_path: Path, action: str, role: str, target: Path, extra: str = ""
) -> None:
    """Append one tab-separated log line to log_path and echo to stdout."""
    parts = [_now(), f"{action:<5}", role, str(target)]
    if extra:
        parts.append(extra)
    line = "\t".join(parts) + "\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def _run_noninteractive_with_resume_logging(
    cmd: list[str], log_path: Path, role: str, target_path: Path
) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        return process.wait()

    session_id: str | None = None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if session_id is not None:
                continue
            match = SESSION_ID_PATTERN.search(line)
            if match is None:
                continue
            session_id = match.group(1)
            log_event(
                log_path,
                "INFO ",
                role,
                target_path,
                f"codex resume --include-non-interactive {session_id}",
            )
    finally:
        process.stdout.close()

    return process.wait()


def _run_interactive_with_resume_logging(
    cmd: list[str], log_path: Path, role: str, target_path: Path
) -> int:
    session_id: str | None = None
    text_buffer = ""

    def master_read(fd: int) -> bytes:
        nonlocal session_id, text_buffer
        data = os.read(fd, 1024)
        if not data or session_id is not None:
            return data

        text_buffer = (text_buffer + data.decode(errors="ignore"))[-4096:]
        match = SESSION_ID_PATTERN.search(text_buffer)
        if match is None:
            return data

        session_id = match.group(1)
        log_event(log_path, "INFO ", role, target_path, f"codex resume {session_id}")
        return data

    status = pty.spawn(cmd, master_read=master_read)
    return os.waitstatus_to_exitcode(status)


def launch(
    role: str,
    target_path: Path,
    log_path: Path,
    extra_context: str = "",
    *,
    interactive: bool = False,
) -> int:
    """
    Invoke a Codex agent for the given role.

    interactive=False (default): runs `codex exec` — non-interactive batch job.
    interactive=True: runs `codex` — interactive TUI session (user can converse).

    Builds the prompt as: <role>.md contents + context block (target path, log path,
    any extra context). Logs START before exec and DONE/FAIL(exit:N) after.
    Returns the process exit code.
    """
    instructions_file = AUTORESEARCH / "prompts" / f"{role}.md"
    if not instructions_file.exists():
        print(
            f"ERROR: instructions file not found: {instructions_file}", file=sys.stderr
        )
        return 1

    instructions = instructions_file.read_text()

    extra_context_parts: list[str] = []
    if extra_context:
        extra_context_parts.append(extra_context.rstrip())
    env_extra_context = os.environ.get("AUTORESEARCH_EXTRA_CONTEXT", "").strip()
    if env_extra_context:
        extra_context_parts.append(env_extra_context)

    context_block = (
        f"\n---\nTarget path: {target_path.resolve()}\nLog path: {log_path.resolve()}\n"
    )
    assigned_gpu = os.environ.get("AUTORESEARCH_ASSIGNED_GPU")
    if assigned_gpu:
        context_block += f"Assigned GPU: {assigned_gpu}\n"
    if extra_context_parts:
        context_block += "\n".join(extra_context_parts) + "\n"

    prompt = instructions + context_block

    cmd = ["codex"] if interactive else ["codex", "exec"]
    cmd += [
        prompt,
        "--dangerously-bypass-approvals-and-sandbox",
        "-m", MODEL,
        "-C", str(REPO),
    ]

    log_event(log_path, "START", role, target_path)
    t0 = time.monotonic()

    if interactive:
        rc = _run_interactive_with_resume_logging(cmd, log_path, role, target_path)
    else:
        rc = _run_noninteractive_with_resume_logging(cmd, log_path, role, target_path)

    elapsed = int(time.monotonic() - t0)
    action = "DONE " if rc == 0 else "FAIL "
    log_event(log_path, action, role, target_path, f"exit:{rc}\t{elapsed}s")
    return rc
