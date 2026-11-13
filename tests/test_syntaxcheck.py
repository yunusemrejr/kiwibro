"""The check run on a file the agent just wrote: parse only, never run, and silent when it cannot tell."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kiwibro import syntaxcheck
from kiwibro.syntaxcheck import Problem, check


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_python_file_that_does_not_compile_is_named_with_its_line(tmp_path: Path) -> None:
    path = _write(tmp_path, "words.py", "def reverse(text):\n    return text\n\nvalue = {'a': 1\n}\nbroken = (1, 2}\n")
    problem = check(path, "words.py")
    assert problem is not None
    assert (problem.path, problem.line) == ("words.py", 6)
    assert problem.for_tool() == (
        f"Syntax error in words.py, line 6: {problem.message}. Fix the file before you say it is done."
    )
    assert problem.for_reply() == f"Checked by Kiwibro: `words.py` has a syntax error on line 6 ({problem.message})."


def test_a_python_file_that_compiles_has_no_problem(tmp_path: Path) -> None:
    assert check(_write(tmp_path, "ok.py", "def greet(name):\n    return f'Hello, {name}'\n"), "ok.py") is None


def test_the_file_is_parsed_and_never_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    path = _write(tmp_path, "side.py", f"open({str(marker)!r}, 'w').write('x')\n")
    assert check(path, "side.py") is None
    assert not marker.exists()


def test_a_null_byte_is_a_problem_without_a_line(tmp_path: Path) -> None:
    path = tmp_path / "nul.py"
    path.write_bytes(b"x = 1\x00\n")
    problem = check(path, "nul.py")
    assert problem is not None and problem.line == 0
    assert " on line " not in problem.for_reply() and ", line" not in problem.for_tool()


def test_json_and_toml(tmp_path: Path) -> None:
    assert check(_write(tmp_path, "a.json", '{"a": 1}\n'), "a.json") is None
    bad_json = check(_write(tmp_path, "b.json", '{\n  "a": 1\n  "b": 2\n}\n'), "b.json")
    assert bad_json is not None and (bad_json.path, bad_json.line) == ("b.json", 3)
    if syntaxcheck.tomllib is None:
        pytest.skip("tomllib needs Python 3.11")
    assert check(_write(tmp_path, "a.toml", 'name = "kiwi"\n'), "a.toml") is None
    bad_toml = check(_write(tmp_path, "b.toml", 'name = "kiwi"\nsize = \n'), "b.toml")
    assert bad_toml is not None and bad_toml.line == 2 and "(at line" not in bad_toml.message


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_shell_script_is_parsed_not_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    assert check(_write(tmp_path, "ok.sh", f"touch {marker}\n"), "ok.sh") is None
    assert not marker.exists()
    problem = check(_write(tmp_path, "bad.sh", "echo start\nif true; then\necho inside\n"), "bad.sh")
    assert problem is not None and problem.path == "bad.sh" and problem.line > 0


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_javascript_file_is_parsed_not_run(tmp_path: Path) -> None:
    marker = tmp_path / "ran.txt"
    assert check(_write(tmp_path, "ok.js", f"require('fs').writeFileSync({str(marker)!r}, 'x');\n"), "ok.js") is None
    assert not marker.exists()
    problem = check(_write(tmp_path, "bad.js", "const a = 1;\nconst b = ;\n"), "bad.js")
    assert problem is not None and (problem.path, problem.line) == ("bad.js", 2)


def test_the_command_goes_through_the_confinement_it_is_given(tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    seen: list[list[str]] = []

    def confine(argv: list[str]) -> list[str]:
        seen.append(argv)
        return argv

    check(_write(tmp_path, "ok.sh", "echo hi\n"), "ok.sh", confine=confine)
    assert len(seen) == 1 and seen[0][1:] == ["-n", str(tmp_path / "ok.sh")]


def test_other_types_missing_and_oversized_files_give_no_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert check(_write(tmp_path, "notes.md", "# {{ not code\n"), "notes.md") is None
    assert check(tmp_path / "gone.py", "gone.py") is None
    assert check(tmp_path, "folder.py") is None
    monkeypatch.setattr(syntaxcheck, "MAX_BYTES", 10)
    assert check(_write(tmp_path, "big.py", "x = (\n" * 5), "big.py") is None


def test_a_checker_that_is_not_installed_gives_no_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(syntaxcheck.shutil, "which", lambda name: None)
    assert check(_write(tmp_path, "a.js", "const = ;\n"), "a.js") is None
    assert check(_write(tmp_path, "a.sh", "if\n"), "a.sh") is None


def test_the_two_lines_read_plainly_without_a_line_number() -> None:
    problem = Problem("data.json", 0, "the file is not UTF-8 text")
    assert problem.for_tool() == "Syntax error in data.json: the file is not UTF-8 text. Fix the file before you say it is done."
    assert problem.for_reply() == "Checked by Kiwibro: `data.json` has a syntax error (the file is not UTF-8 text)."
