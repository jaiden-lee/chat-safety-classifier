"""LLM access for offline data generation and labeling audits. Never used in the serving path.

`LLMBackend` is the interface the generators call. `ClaudeCLIBackend` runs Claude Code in
headless mode (`claude -p`), which uses the logged-in Claude subscription instead of an API key.
An API-key backend can be added later behind the same interface.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(RuntimeError):
    """The LLM call failed after all retries."""


@dataclass(frozen=True)
class LLMResult:
    """One completion: the text plus the metadata worth keeping for provenance."""

    text: str
    model: str  # resolved model id, e.g. "claude-haiku-5-5"
    duration_s: float
    raw: dict = field(default_factory=dict, repr=False)  # full CLI response


class LLMBackend(Protocol):
    """Anything that turns a prompt into text."""

    def complete(self, prompt: str, *, model: str, system: str) -> LLMResult: ...


class ClaudeCLIBackend:
    """Calls `claude -p` as a subprocess, with tools, MCP servers, and session saving turned off.

    The prompt goes on stdin. `--system-prompt` replaces Claude Code's default coding-agent
    prompt, so calls are small and the model acts as a plain text generator. Failed calls
    are retried with exponential backoff. Usage-limit errors (subscription limits reset after
    hours) instead wait `limit_wait_s` and retry, up to `max_limit_waits` times, without using
    up the normal retries.
    """

    def __init__(
        self,
        executable: str = "claude",
        timeout_s: float = 900,
        max_retries: int = 4,
        backoff_s: float = 30,
        effort: str | None = "low",
        limit_wait_s: float = 1800,
        max_limit_waits: int = 12,
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise LLMError(f"{executable!r} not found on PATH; is Claude Code installed?")
        self.executable = resolved
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.effort = effort  # Claude Code --effort: low, medium, high, xhigh, max
        self.limit_wait_s = limit_wait_s
        self.max_limit_waits = max_limit_waits
        # Run outside the repo so no project CLAUDE.md or settings leak into generation.
        self.cwd = tempfile.gettempdir()

    def complete(self, prompt: str, *, model: str, system: str) -> LLMResult:
        cmd = [
            self.executable, "-p",
            "--model", model,
            "--output-format", "json",
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--system-prompt", system,
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        last_error = ""
        failures = limit_waits = 0
        while True:
            if last_error:
                if _is_usage_limit(last_error) and limit_waits < self.max_limit_waits:
                    limit_waits += 1
                    print(f"usage limit hit; waiting {self.limit_wait_s / 60:.0f} min "
                          f"({limit_waits}/{self.max_limit_waits}): {last_error[:120]}",
                          flush=True)
                    time.sleep(self.limit_wait_s)
                else:
                    failures += 1
                    if failures > self.max_retries:
                        break
                    time.sleep(self.backoff_s * 2 ** (failures - 1))
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", check=False,
                    timeout=self.timeout_s, cwd=self.cwd,
                )
            except subprocess.TimeoutExpired:
                last_error = f"timed out after {self.timeout_s}s"
                continue
            duration = time.monotonic() - start
            try:
                response = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_error = f"exit {proc.returncode}, non-JSON output: {proc.stdout[:300]!r} " \
                             f"{proc.stderr[:300]!r}"
                continue
            if proc.returncode != 0 or response.get("is_error"):
                last_error = f"exit {proc.returncode}: {str(response.get('result'))[:300]}"
                continue
            return LLMResult(response.get("result", ""), _main_model(response, model), duration,
                             response)
        raise LLMError(f"claude -p failed after {self.max_retries + 1} attempts: {last_error}")


def _is_usage_limit(error: str) -> bool:
    text = error.lower()
    return "limit" in text and any(w in text for w in ("usage", "rate", "reset", "reached"))


def _main_model(response: dict, requested: str) -> str:
    """The model that wrote the answer. Claude Code also makes small side calls (e.g., to Haiku),
    so `modelUsage` can list several models; the answer comes from the one with most output."""
    usage = response.get("modelUsage") or {}
    if not usage:
        return requested
    return max(usage, key=lambda m: usage[m].get("outputTokens", 0))
