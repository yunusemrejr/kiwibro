"""Kiwi Mind's host never outlives the app: the real app is started, asked one thing, then stopped the hard way."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from mind_pack import ROOT, make_pack, node_path

pytestmark = pytest.mark.skipif(node_path() is None, reason="node is not installed")

TEST_PORTS = (8791, 8792, 8793, 8794)  # never 8777: that is the user's own app
GONE_WITHIN_S = 3.0


def free_port() -> int:
    for port in TEST_PORTS:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    pytest.skip("ports 8791-8794 are all in use")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is a stopped process that nobody has collected yet
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def wait_gone(pid: int, seconds: float) -> float | None:
    """Seconds until the process was gone, or None if it still ran after ``seconds``."""
    began = time.monotonic()
    while time.monotonic() - began < seconds:
        if not alive(pid):
            return time.monotonic() - began
        time.sleep(0.02)
    return None if alive(pid) else time.monotonic() - began


def wait_health(port: int) -> dict:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            return requests.get(f"http://127.0.0.1:{port}/health", timeout=2).json()
        except requests.RequestException:
            time.sleep(0.2)
    raise AssertionError("the app did not come up")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL], ids=["sigterm", "sigkill"])
def test_the_host_is_gone_within_seconds_when_the_app_stops(tmp_path: Path, sig: signal.Signals) -> None:
    data = tmp_path / "data"
    make_pack(data)
    port = free_port()
    env = {
        **os.environ,
        "KIWIBRO_DATA_DIR": str(data), "KIWIBRO_WORKSPACE": str(tmp_path / "work"), "KIWIBRO_PORT": str(port),
        "KIWIBRO_AUTO_UPDATE": "0", "KIWIBRO_WARMUP": "0", "KIWIMIND_AUTOSTART": "0", "KIWIBRO_OFFLINE": "0",
    }
    env.pop("KIWIBRO_SERVER_URL", None)
    log = (tmp_path / "app.log").open("wb")
    app = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env, stdout=log, stderr=log,
                           start_new_session=True)
    host_pid = 0
    app_pid = 0
    try:
        health = wait_health(port)
        app_pid = int(health["pid"])
        assert health["mind"] == {"usable": True, "running": False, "pid": None}  # nothing runs before it is needed
        reply = requests.post(f"http://127.0.0.1:{port}/api/chat", json={"message": "hello there"}, stream=True,
                              timeout=60)
        assert reply.status_code == 200
        deadline = time.monotonic() + 30
        for line in reply.iter_lines():  # the first words of the answer: the host has been asked something
            if line.startswith(b"data:") and b'"brain"' in line:
                assert json.loads(line[5:])["brain"] == "kiwimind"
                break
            assert time.monotonic() < deadline, "no answer from Kiwi Mind"
        mind = requests.get(f"http://127.0.0.1:{port}/health", timeout=5).json()["mind"]
        assert mind["running"] and mind["pid"], mind
        host_pid = int(mind["pid"])
        assert alive(host_pid) and host_pid != app_pid
        os.kill(app_pid, sig)
        reply.close()
        assert wait_gone(app_pid, 5) is not None, "the app itself did not stop"
        gone = wait_gone(host_pid, GONE_WITHIN_S)
        print(f"\n{sig.name}: app stopped, Kiwi Mind host gone {gone if gone is None else round(gone, 2)} s later")
        assert gone is not None, f"Kiwi Mind host {host_pid} still runs {GONE_WITHIN_S} s after {sig.name}"
    finally:
        for pid in (host_pid, app_pid):
            if pid and alive(pid):
                os.kill(pid, signal.SIGKILL)
        try:
            app.wait(5)
        except subprocess.TimeoutExpired:
            app.kill()
        log.close()
