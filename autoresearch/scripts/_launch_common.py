"""Shared utilities for autoresearch launcher scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent.parent          # scripts/ -> autoresearch/ -> repo root
AUTORESEARCH = Path(__file__).parent.parent        # scripts/ -> autoresearch/
# Override via AUTORESEARCH_MODEL env var if needed
MODEL = os.environ.get("AUTORESEARCH_MODEL", "gpt-5.4")


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

    context_block = (
        f"\n---\nTarget path: {target_path.resolve()}\nLog path: {log_path.resolve()}\n"
    )
    if extra_context:
        context_block += f"{extra_context}\n"

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

    result = subprocess.run(cmd)

    elapsed = int(time.monotonic() - t0)
    rc = result.returncode
    action = "DONE " if rc == 0 else "FAIL "
    log_event(log_path, action, role, target_path, f"exit:{rc}\t{elapsed}s")
    return rc
