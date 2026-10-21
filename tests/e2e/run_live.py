#!/usr/bin/env python3
"""Live verification runner for Kiwibro (task-3).

Starts the latest app code on port 8799 with scratch state (``.dev/qa-data`` +
``.dev/qa-workspace``), runs serialized checks against real LM Studio, optionally
runs the live browser pass (which writes the committed screenshots), then stops.

Gated: only run after the Lead confirms LM Studio is healthy and the fixes are in.

    python3 tests/e2e/run_live.py                    # full live run + browser screenshots
    python3 tests/e2e/run_live.py --no-browser       # server/API live checks only
    python3 tests/e2e/run_live.py --base http://127.0.0.1:8799   # attach to a server
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import requests  # noqa: E402

from harness import REPO_ROOT, RUNTIME_ROOT, AppServer, ApiClient, Runner  # noqa: E402
import checks_live  # noqa: E402

LIVE_ROOT = REPO_ROOT / ".dev" / "qa" / "live"
DATA_DIR = REPO_ROOT / ".dev" / "qa-data"
WORKSPACE = REPO_ROOT / ".dev" / "qa-workspace"
DEFAULT_PORT = 8799


class Env:
    def __init__(self, app: AppServer | None, base_url: str, conversation_id: str = "") -> None:
        self.app = app
        self.base_url = base_url
        self.client = ApiClient(base_url)
        self.workspace = Path(app.workspace) if app else WORKSPACE
        self.conversation_id = conversation_id

    def restart(self) -> None:
        if not self.app:
            raise RuntimeError("cannot restart an attached server")
        self.app.stop()
        self.app.start()
        self.client = ApiClient(self.app.base_url)


# ---------------------------------------------------------------------------
# direct live probes (no model): search providers and page fetching
# ---------------------------------------------------------------------------


def direct_search_provider(env: Env) -> str:
    from kiwibro.search import SearchEngine

    engine = SearchEngine()
    started = time.monotonic()
    results = engine.web_search("kiwi bird conservation news", 6)
    elapsed = time.monotonic() - started
    assert results, "web search returned no results from any provider"
    snippets = [result.get("snippet", "") for result in results]
    non_empty = [snippet for snippet in snippets if len(snippet) >= 30]
    assert non_empty, f"search snippets are weak/empty: {snippets[:3]}"
    assert all(result.get("url", "").startswith("http") for result in results), "result URLs"
    return (
        f"{len(results)} results in {elapsed:.1f}s, "
        f"{len(non_empty)} non-empty snippets (first: {results[0]['url']})"
    )


def direct_fetch_page(env: Env) -> str:
    from kiwibro.search import SearchEngine

    page = SearchEngine().fetch_page("https://en.wikipedia.org/wiki/Kiwi_(bird)", 4000)
    assert page["title"], "page title extracted"
    assert len(page["text"]) > 500, f"readable text length {len(page['text'])}"
    assert "kiwi" in page["text"].lower(), "page content mentions kiwi"
    return f"fetched {page['title']!r} ({len(page['text'])} chars, truncated={page['truncated']})"


DIRECT_CHECKS: list[tuple[str, Any]] = [
    ("direct_search_provider", direct_search_provider),
    ("direct_fetch_page", direct_fetch_page),
]


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def lm_studio_probe(url: str, timeout: float = 12.0) -> tuple[bool, str]:
    try:
        response = requests.get(f"{url.rstrip('/')}/api/v0/models", timeout=(5, timeout))
    except requests.RequestException as error:
        return False, f"LM Studio unreachable at {url}: {error}"
    if response.status_code != 200:
        return False, f"LM Studio /api/v0/models returned HTTP {response.status_code}"
    try:
        data = response.json().get("data") or []
    except ValueError:
        return False, "LM Studio returned invalid JSON"
    chat = [model for model in data if model.get("type") in ("llm", "vlm")]
    loaded = [model["id"] for model in chat if model.get("state") == "loaded"]
    return True, f"LM Studio up: {len(chat)} chat models, loaded={loaded}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base", default="", help="attach to an already-running server instead of spawning")
    parser.add_argument("--lm-url", default="http://127.0.0.1:1234")
    parser.add_argument("--browser", dest="browser", action="store_true", default=True)
    parser.add_argument("--no-browser", dest="browser", action="store_false")
    parser.add_argument("--shots-out", default="docs/screenshots")
    parser.add_argument("--out", default=str(LIVE_ROOT / "evidence.json"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-slow", action="store_true", help="skip compaction (costly) and screenshot flows")
    args = parser.parse_args(argv)

    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    runner = Runner(verbose=not args.quiet)
    started = time.time()

    healthy, message = lm_studio_probe(args.lm_url)
    print(f"LM Studio probe: {'OK' if healthy else 'FAILED'} — {message}")
    if not healthy:
        print("Aborting: run the deterministic suite instead until LM Studio is confirmed healthy.")
        return 2

    app: AppServer | None = None
    try:
        if args.base:
            base_url = args.base.rstrip("/")
            print(f"Attached live run against {base_url}")
            env = Env(None, base_url)
        else:
            # fresh scratch state for the live pass (never the user's real data)
            for directory in (DATA_DIR, WORKSPACE):
                shutil.rmtree(directory, ignore_errors=True)
            app = AppServer(
                "live",
                lmstudio_url=args.lm_url,
                port=args.port,
                data_dir=DATA_DIR,
                workspace=WORKSPACE,
            )
            app.start()
            base_url = app.base_url
            print(f"Live app on {base_url} (data={DATA_DIR}, workspace={WORKSPACE})")
            env = Env(app, base_url)

        # live checks must run on the documented model, not just "first loaded"
        models = env.client.get("/api/models?refresh=1").data
        ids = [model["id"] for model in models.get("models") or []]
        assert "qwen3.8-2b" in ids, f"qwen3.8-2b is not available in LM Studio (have {ids})"
        selected = env.client.post("/api/models/select", {"id": "qwen3.8-2b"})
        assert selected.status == 200, f"model select failed: {selected.status}"
        print("Using model qwen3.8-2b for all live checks")

        # a conversation shared by the read-only live checks
        created = env.client.post("/api/conversations", {"title": "QA live verification"})
        assert created.status == 201, f"conversation create failed: {created.status} {created.text[:200]}"
        env.conversation_id = created.data["id"]

        checks = list(DIRECT_CHECKS) + list(checks_live.LIVE_CHECKS)
        if args.skip_slow:
            checks = [entry for entry in checks if entry[0] not in ("live_context_compaction",)]
        print("\n== live checks (serialized, one model call at a time) ==")
        for name, func in checks:
            runner.check("live", name, func, env)

        if args.browser:
            browser_script = HERE / "browser" / "run_browser.mjs"
            cmd = [
                "node",
                str(browser_script),
                "--mode",
                "live",
                "--base",
                base_url,
                "--out",
                args.shots_out,
            ]
            print(f"\n== live browser pass -> {args.shots_out} ==")
            try:
                process = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=900)
                runner.check(
                    "browser",
                    "live_browser_rig",
                    (lambda: "browser checks + screenshots passed")
                    if process.returncode == 0
                    else _raise_browser,
                    env,
                )
            except subprocess.TimeoutExpired:
                runner.check("browser", "live_browser_rig", _raise_browser_timeout, env)
    except Exception as error:  # noqa: BLE001 - always produce evidence
        print(f"live run aborted: {type(error).__name__}: {error}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            print(f"\napp log tail:\n{app.log_tail(20)}")
            app.stop()

    passed = len(runner.results) - len(runner.failures)
    elapsed = time.time() - started
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_evidence(out_path)
    print(f"\n{passed}/{len(runner.results)} live checks passed in {elapsed:.1f}s")
    print(f"evidence: {out_path} (+ {out_path.with_suffix('.md')})")
    if runner.failures:
        print("\nFAILURES:")
        for failure in runner.failures:
            print(f"  - {failure.name}: {failure.error}")
        return 1
    return 0


def _raise_browser() -> str:
    raise AssertionError("live browser rig failed (see output above)")


def _raise_browser_timeout() -> str:
    raise AssertionError("live browser rig timed out after 900s")


if __name__ == "__main__":
    sys.exit(main())
