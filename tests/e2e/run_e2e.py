#!/usr/bin/env python3
"""Standalone end-to-end runner for Kiwibro.

No pytest needed::

    python3 tests/e2e/run_e2e.py                 # full fake-LM suite
    python3 tests/e2e/run_e2e.py --no-probes     # skip in-process unit probes
    python3 tests/e2e/run_e2e.py --only chat     # one phase
    python3 tests/e2e/run_e2e.py --list          # list checks and exit

Every phase owns a real ``python3 app.py`` process and a scratch data dir under
``.dev/qa/runtime/``; the fake LM server keeps the suite deterministic and
completely offline (no LM Studio, no internet).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fake_lm import FakeLM, MODELS_NO_TOOLS, MODELS_TINY_CONTEXT  # noqa: E402
from harness import RUNTIME_ROOT, AppServer, ApiClient, Runner, free_port  # noqa: E402
import checks_api  # noqa: E402
import checks_chat  # noqa: E402
import checks_compaction  # noqa: E402
import probes_static  # noqa: E402


class Env:
    """Everything a check needs: the app process, a client, workspace, fake LM."""

    def __init__(self, app: AppServer, fake: FakeLM | None, *, offline: bool = False) -> None:
        self.app = app
        self.fake = fake
        self.offline = offline
        self.client = ApiClient(app.base_url)
        self.workspace = app.workspace

    def restart(self) -> None:
        self.app.stop()
        self.app.start()
        self.client = ApiClient(self.app.base_url)


# ---------------------------------------------------------------------------
# check registry: (phase, name, function)
# ---------------------------------------------------------------------------

API_CHECKS = [
    ("api", "models_select_validation", checks_api.models_select_validation),
    ("api", "conversations_crud", checks_api.conversations_crud),
    ("api", "conversation_cascade", checks_api.conversation_cascade),
    ("api", "memories_crud", checks_api.memories_crud),
    ("api", "settings_validation", checks_api.settings_validation),
    ("api", "workspace_listing", checks_api.workspace_listing),
    ("api", "api_headers", checks_api.api_headers),
    ("api", "host_header_guard", checks_api.host_header_guard),
    ("api", "error_shapes", checks_api.error_shapes),
    ("api", "chat_validation", checks_api.chat_validation),
    ("api", "static_shell", checks_api.static_shell),
    ("api", "restart_persistence", checks_api.restart_persistence),
]

OFFLINE_CHECKS = [
    ("offline", "health_offline", checks_api.health_offline),
    ("offline", "bootstrap_offline", checks_api.bootstrap_offline),
    ("offline", "models_offline", checks_api.models_offline),
    ("offline", "chat_offline_is_graceful", checks_api.chat_offline_is_graceful),
]

CHAT_CHECKS = [
    ("chat", "plain_stream", checks_chat.plain_stream),
    ("chat", "utf8_stream", checks_chat.utf8_stream),
    ("chat", "markdown_stream", checks_chat.markdown_stream),
    ("chat", "reasoning_stream", checks_chat.reasoning_stream),
    ("chat", "no_usage_stream", checks_chat.no_usage_stream),
    ("chat", "empty_stream", checks_chat.empty_stream),
    ("chat", "title_event", checks_chat.title_event),
    ("chat", "title_echo_rejected", checks_chat.title_echo_rejected),
    ("chat", "memory_extraction", checks_chat.memory_extraction),
    ("chat", "tool_write_allow", checks_chat.tool_write_allow),
    ("chat", "tool_write_deny", checks_chat.tool_write_deny),
    ("chat", "tool_write_always", checks_chat.tool_write_always),
    ("chat", "tool_errors", checks_chat.tool_errors),
    ("chat", "tool_remember", checks_chat.tool_remember),
    ("chat", "security_write_traversal", checks_chat.security_write_traversal),
    ("chat", "security_read_traversal", checks_chat.security_read_traversal),
    ("chat", "security_ssrf", checks_chat.security_ssrf),
    ("chat", "security_command_blocklist", checks_chat.security_command_blocklist),
    ("chat", "busy_409_and_stop", checks_chat.busy_409_and_stop),
    ("chat", "client_disconnect_aborts", checks_chat.client_disconnect_aborts),
    ("chat", "context_limit_recovery", checks_chat.context_limit_recovery),
    ("chat", "context_limit_unrecoverable", checks_chat.context_limit_unrecoverable),
    ("chat", "tools_unsupported_retry", checks_chat.tools_unsupported_retry),
    ("chat", "model_swap_preserves_state", checks_chat.model_swap_preserves_state),
]

COMPACTION_CHECKS = [
    ("compaction", "compaction_happens", checks_compaction.compaction_happens),
    ("compaction", "compaction_keeps_chatting", checks_compaction.compaction_keeps_chatting),
    ("compaction", "summary_reaches_the_model", checks_compaction.summary_reaches_the_model),
    ("compaction", "meta_is_first_event", checks_compaction.meta_is_first_event),
]

NO_TOOLS_CHECKS = [
    ("notools", "no_tools_model", checks_chat.no_tools_model),
]

PHASES = ("unit", "offline", "api", "chat", "compaction", "notools")


def _run_probes(runner: Runner) -> None:
    for name, func in probes_static.PROBES:
        runner.check("unit", name, func)


def _run_phase(runner: Runner, env: Env, checks: list[tuple[str, str, Any]], verbose: bool) -> None:
    for phase, name, func in checks:
        runner.check(phase, name, func, env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=PHASES, action="append", help="run only the given phase(s)")
    parser.add_argument("--no-probes", action="store_true", help="skip in-process unit probes")
    parser.add_argument("--out", default=str(RUNTIME_ROOT / "evidence.json"), help="evidence JSON path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    args = parser.parse_args(argv)

    selected = set(args.only or PHASES)
    verbose = not args.quiet

    if args.list:
        for phase, name, _func in OFFLINE_CHECKS + API_CHECKS + CHAT_CHECKS + COMPACTION_CHECKS + NO_TOOLS_CHECKS:
            print(f"{phase:12s} {name}")
        for name, _func in probes_static.PROBES:
            print(f"{'unit':12s} {name}")
        return 0

    started = time.time()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    runner = Runner(verbose=verbose)

    if "unit" in selected and not args.no_probes:
        print("\n== unit probes (in-process) ==")
        _run_probes(runner)

    offline_url = f"http://127.0.0.1:{free_port()}"  # closed port by design
    if "offline" in selected:
        print("\n== phase: offline LM Studio ==")
        with AppServer("offline", lmstudio_url=offline_url) as app:
            env = Env(app, None, offline=True)
            _run_phase(runner, env, OFFLINE_CHECKS, verbose)

    if "api" in selected:
        print("\n== phase: HTTP API (fake LM) ==")
        with FakeLM() as fake, AppServer("api", lmstudio_url=fake.url) as app:
            env = Env(app, fake)
            _run_phase(runner, env, API_CHECKS, verbose)

    if "chat" in selected:
        print("\n== phase: chat / tools / security (fake LM) ==")
        with FakeLM() as fake, AppServer("chat", lmstudio_url=fake.url) as app:
            env = Env(app, fake)
            _run_phase(runner, env, CHAT_CHECKS, verbose)

    if "compaction" in selected:
        print("\n== phase: compaction (tiny context fake LM) ==")
        with FakeLM(MODELS_TINY_CONTEXT) as fake, AppServer("compaction", lmstudio_url=fake.url) as app:
            env = Env(app, fake)
            _run_phase(runner, env, COMPACTION_CHECKS, verbose)

    if "notools" in selected:
        print("\n== phase: model without tool support ==")
        with FakeLM(MODELS_NO_TOOLS) as fake, AppServer("notools", lmstudio_url=fake.url) as app:
            env = Env(app, fake)
            _run_phase(runner, env, NO_TOOLS_CHECKS, verbose)

    elapsed = time.time() - started
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_evidence(out_path)

    passed = len(runner.results) - len(runner.failures)
    print(f"\n{passed}/{len(runner.results)} checks passed in {elapsed:.1f}s")
    print(f"evidence: {out_path} (+ {out_path.with_suffix('.md')})")
    if runner.failures:
        print("\nFAILURES:")
        for failure in runner.failures:
            print(f"  - [{failure.group}] {failure.name}: {failure.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
