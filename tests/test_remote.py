"""Remote server tool: validation, the exact ssh/scp argv, diagnostics and the host policy."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiwibro import config, remote, safety, tools
from kiwibro.sandbox import Workspace


@pytest.fixture()
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """ssh/scp stand-ins that record their argv; exit code and output come from files
    (remote._env() passes ssh only a whitelist of variables, by design)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "argv.log"
    for name in ("ssh", "scp"):
        script = bin_dir / name
        script.write_text(
            f'#!/bin/sh\nprintf "%s\\n" "{name}" "$@" > "{log}"\n'
            f'cat "{tmp_path}/output" 2>/dev/null || printf "remote says hi"\n'
            f'exit "$(cat "{tmp_path}/exit" 2>/dev/null || echo 0)"\n'
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(workspace=Workspace(tmp_path / "ws"))


def test_registered_as_an_ask_by_default_tool() -> None:
    assert "remote" in tools.TOOL_SPECS
    assert config.permission_defaults()["remote"] == "ask"


@pytest.mark.parametrize(
    "args",
    [
        {"action": "run", "host": "-oProxyCommand=sh", "command": "ls"},
        {"action": "run", "host": "prod; rm -rf ~", "command": "ls"},
        {"action": "run", "host": "prod", "command": ""},
        {"action": "upload", "host": "prod", "path": "a.txt", "remote_path": "-r"},
        {"action": "explode", "host": "prod"},
    ],
)
def test_validation_rejects_injection_and_nonsense(args: dict) -> None:
    with pytest.raises(ValueError):
        remote.validate(args)


def test_config_hosts_skips_wildcards(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.write_text("Host prod staging\n  User deploy\nHost *.internal\nhost github.com\n")
    assert remote.config_hosts(cfg) == ["prod", "staging", "github.com"]


def test_run_builds_a_batch_mode_argv(tmp_path: Path, fake_ssh: Path) -> None:
    args = remote.validate({"action": "run", "host": "deploy@prod", "command": "tail -n 5 /var/log/app.log"})
    result = remote.execute(args, _ctx(tmp_path))
    argv = fake_ssh.read_text().splitlines()
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and argv[-3:] == ["deploy@prod", "--", "tail -n 5 /var/log/app.log"]
    assert result.ok and "remote says hi" in result.content and "(exit 0)" in result.content


def test_upload_and_download_stay_in_the_workspace(tmp_path: Path, fake_ssh: Path) -> None:
    ctx = _ctx(tmp_path)
    (ctx.workspace.root / "site.tar.gz").write_bytes(b"x")
    up = remote.execute(remote.validate({"action": "upload", "host": "prod", "path": "site.tar.gz", "remote_path": "/srv/site.tar.gz"}), ctx)
    argv = fake_ssh.read_text().splitlines()
    assert up.ok and argv[0] == "scp" and argv[-2:] == [str(ctx.workspace.root / "site.tar.gz"), "prod:/srv/site.tar.gz"]
    escape = remote.execute(remote.validate({"action": "download", "host": "prod", "path": "../../x", "remote_path": "/etc/hosts"}), ctx)
    assert not escape.ok and "outside the workspace" in escape.content


def test_failures_explain_what_to_do(tmp_path: Path, fake_ssh: Path) -> None:
    (tmp_path / "exit").write_text("255")
    (tmp_path / "output").write_text("deploy@prod: Permission denied (publickey).")
    result = remote.execute(remote.validate({"action": "run", "host": "prod", "command": "ls"}), _ctx(tmp_path))
    assert not result.ok and "ssh-copy-id" in result.content


def test_host_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote, "config_hosts", lambda path=None: ["prod"])
    smart = {"autonomy": "smart"}
    assert safety.decide(smart, "remote", {"action": "run", "host": "prod", "command": "ls /srv"}).mode == "auto"
    unknown = safety.decide(smart, "remote", {"action": "run", "host": "evil.example", "command": "ls"})
    assert (unknown.mode, unknown.level) == ("ask", "high")
    allowed = {"autonomy": "smart", "remote_hosts": ["me@10.0.0.5"]}
    assert safety.decide(allowed, "remote", {"action": "run", "host": "me@10.0.0.5", "command": "ls"}).mode == "auto"
    assert safety.decide(smart, "remote", {"action": "run", "host": "prod", "command": "git pull && npm ci"}).mode == "ask"
    assert safety.decide({"autonomy": "autonomous"}, "remote", {"action": "run", "host": "prod", "command": "npm ci"}).mode == "auto"
    assert safety.decide(smart, "remote", {"action": "run", "host": "prod", "command": "sudo systemctl restart app"}).mode == "ask"
    assert safety.decide(smart, "remote", {"action": "run", "host": "prod", "command": "rm -rf /"}).mode == "deny"


def test_remote_hosts_setting_is_validated() -> None:
    assert config.validate_settings_patch({"remote_hosts": ["prod", "prod"]}, {}) == {"remote_hosts": ["prod"]}
    with pytest.raises(ValueError):
        config.validate_settings_patch({"remote_hosts": ["-oProxyCommand=x"]}, {})
