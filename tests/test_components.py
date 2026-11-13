"""Tests for the component library (kiwibro/components.py) and its tool."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiwibro import checkpoints, components, config, safety, tools  # noqa: E402
from kiwibro.sandbox import Workspace  # noqa: E402
from kiwibro.tools import ToolContext  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "kiwibro" / "components" / "library"
TRAIN = REPO / "scripts" / "mind" / "train.py"
MANIFESTS = sorted(LIBRARY.glob("*/*/component.json"))
HOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
LEFTOVER = re.compile(r"\{\{[a-z_]+\}\}")
# A line that only prints is free to show a command; anything else that says
# "remove this" is a kit that removes files, which no kit may do.
PRINTS = re.compile(r"\s*(?:[A-Za-z_][\w.]*=\S+\s+)*(?:printf|echo|log|print|warn|say|hint)\b")
QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
DELETES = re.compile(
    r"(?:^|[\s;&|(`])(?:rm|rmdir|unlink|shred)\b"
    r"|\bfind\b[^\n]*\s-delete\b|\bgit\s+clean\b"
    r"|\bos\.remove\b|\bshutil\.rmtree\b|\.unlink\("
    r"|\bDROP\s+TABLE\b",
    re.IGNORECASE,
)

PAGE_LINE = ("web/page (kit) - One responsive page: semantic HTML, phone-first CSS with design tokens, no framework. "
             "Blanks: title, description")
SITE_LINE = ("web/site (kit) - Small multi-page static site: shared header and footer, home, about and contact "
             "pages, one stylesheet. Blanks: site_name")
RESEARCH_LINE = ("browser/research (recipe) - Answer a question from several sources: search with two or three "
                 "queries, read the best pages, cite numbered sources.")
PAGE_NEXT = ("Write the files, replace the text marked KIWIBRO with the project's own, then open index.html at 360 px "
             "and at desktop width to check the layout.")
RESEARCH_STEPS = (
    "Rewrite the question as three short search phrases covering different angles and call multi_search with queries "
    "and max_results 6.",
    "Pick three to five promising URLs from the ranked list, favouring official and dated pages, and call fetch_pages "
    "with those urls.",
    "Re-read the numbered evidence pack and note the exact sentence behind each figure you plan to use, with its "
    "source number.",
    "If two sources give different figures, call fetch_page on the stronger one and check which figure is newer and "
    "better sourced.",
    "Drop any claim that no fetched passage supports, then call note action save with the question and the numbered "
    "source list.",
    "If a fetch fails with a host or timeout error, say the page could not be read and use search snippets only, "
    "marked unverified.",
    "Answer in plain sentences, cite each fact as [1] or [2] in order of first use, and list the URLs at the end.",
)
RESEARCH_NEXT = ("The final answer must state the answer in plain sentences with numbered citations [1], [2] and the "
                 "URL list below.")
LIST_HINT = 'To use one: component(action="use", name="<name>", path="<folder>", values={...})'


def manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def train_module():
    spec = importlib.util.spec_from_file_location("kiwibro_mind_train_components", TRAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kit_files(folder: Path) -> list[str]:
    return [item.relative_to(folder).as_posix() for item in folder.rglob("*") if item.is_file()]


def lines_that_delete(text: str) -> list[str]:
    """Lines of a kit file that remove something, ignoring a command shown by a print line."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if not DELETES.search(line):
            continue
        bare = QUOTED.sub("", line) if PRINTS.match(line) else line
        if DELETES.search(bare):
            found.append(f"{number}: {line.strip()}")
    return found


@pytest.fixture(scope="module")
def train():
    return train_module()


@pytest.fixture()
def tiny(tmp_path: Path) -> components.Library:
    """A three component library: two kits and a recipe, copied from the real one."""
    root = tmp_path / "library"
    for name in ("web/page", "web/site", "browser/research"):
        shutil.copytree(LIBRARY / name, root / name)
    return components.Library(root)


# --- the library itself ---------------------------------------------------------------


def test_every_manifest_loads_with_the_required_keys():
    assert len(MANIFESTS) == 38
    for path in MANIFESTS:
        data = manifest(path)
        for key in ("name", "kind", "title", "summary", "when", "blanks", "next"):
            assert key in data, f"{path} has no {key}"
        assert data["name"] == "/".join(path.relative_to(LIBRARY).parent.parts), path
        assert data["kind"] in components.KINDS, path
        assert data["title"] and data["summary"] and data["when"] and data["next"], path
        if data["kind"] == components.KIT:
            assert kit_files(path.parent / "files"), f"{path} is a kit without files"
            assert data["blanks"], f"{path} is a kit without blanks"
            for key, blank in data["blanks"].items():
                assert set(blank) == {"about", "default"}, f"{path}: {key}"
                assert blank["about"] and blank["default"], f"{path}: {key}"
        else:
            assert data["blanks"] == {}, path
            assert 4 <= len(data["steps"]) <= 8, path
            assert all(isinstance(step, str) and step.strip() for step in data["steps"]), path
            assert not (path.parent / "files").exists(), f"{path} is a recipe with files"


def test_library_matches_the_reference_catalog(train):
    reference = {name: (blanks, summary) for name, blanks, summary in train.COMPONENTS}
    library = {component.name: component for component in components.Library(LIBRARY).all()}
    assert set(library) == set(reference)
    for name, (blanks, summary) in reference.items():
        wanted = [key.strip() for key in blanks.split(",") if key.strip()]
        assert list(library[name].blanks) == wanted, name
        assert library[name].summary == summary, name
        assert library[name].kind == (components.KIT if wanted else "recipe"), name


def test_kit_placeholders_are_declared_blanks():
    for path in MANIFESTS:
        data = manifest(path)
        if data["kind"] != components.KIT:
            continue
        blanks = set(data["blanks"])
        for relative in kit_files(path.parent / "files"):
            used = set(HOLDER.findall(relative))
            used |= set(HOLDER.findall((path.parent / "files" / relative).read_text(encoding="utf-8")))
            assert used <= blanks, f"{path}: {relative} uses {sorted(used - blanks)}"


def test_kit_files_survive_their_defaults(tmp_path: Path):
    node = shutil.which("node")
    for path in MANIFESTS:
        data = manifest(path)
        if data["kind"] != components.KIT:
            continue
        defaults = {key: blank["default"] for key, blank in data["blanks"].items()}
        source_root = path.parent / "files"
        out = tmp_path / data["name"].replace("/", "-")
        for relative in kit_files(source_root):
            source = source_root / relative
            name = HOLDER.sub(lambda match: defaults[match.group(1)], relative)
            text = HOLDER.sub(lambda match: defaults[match.group(1)], source.read_text(encoding="utf-8"))
            assert not LEFTOVER.search(text + name), f"{path}: {name} keeps a placeholder"
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            if target.suffix == ".py":
                compile(text, str(target), "exec")
            elif target.suffix == ".json":
                json.loads(text)
            elif target.suffix == ".sh":
                done = subprocess.run(["bash", "-n", str(target)], capture_output=True, text=True)
                assert done.returncode == 0, f"{target}: {done.stderr}"
            elif target.suffix in (".js", ".mjs") and node:
                done = subprocess.run([node, "--check", str(target)], capture_output=True, text=True)
                assert done.returncode == 0, f"{target}: {done.stderr}"
            assert len(text.splitlines()) <= 200, f"{target} is over 200 lines"
        size = sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file())
        assert size <= 16 * 1024, f"{path.parent} is {size} bytes"


def test_library_stays_small():
    size = sum(item.stat().st_size for item in LIBRARY.rglob("*") if item.is_file())
    assert size < 400 * 1024, f"the library is {size} bytes"


def test_no_kit_file_removes_anything():
    offenders = []
    for path in MANIFESTS:
        data = manifest(path)
        if data["kind"] != components.KIT:
            continue
        for relative in kit_files(path.parent / "files"):
            found = lines_that_delete((path.parent / "files" / relative).read_text(encoding="utf-8"))
            offenders += [f"{path.parent.name}/{relative}: {line}" for line in found]
    assert not offenders, offenders


def test_a_printed_command_is_not_a_removal():
    assert lines_that_delete('log "remove it with: rm -- $(printf \'%q\' "$file")"\n') == []
    assert lines_that_delete("printf 'To undo, run: rm -rf -- %q\\n' \"$dir\"\n") == []
    assert lines_that_delete("rm -rf -- \"$dir\"\n") == ["1: rm -rf -- \"$dir\""]
    assert lines_that_delete('echo "done" && rm -rf -- "$dir"\n') == ['1: echo "done" && rm -rf -- "$dir"']
    assert lines_that_delete("    shutil.rmtree(folder)\n") == ["1: shutil.rmtree(folder)"]
    assert lines_that_delete("    os.remove(path)\n") == ["1: os.remove(path)"]
    assert lines_that_delete("DROP TABLE notes;\n") == ["1: DROP TABLE notes;"]


# --- the text the model reads ---------------------------------------------------------


def test_search_names_the_best_matches(tiny: components.Library):
    expected = "\n".join([
        '3 components match "web page":',
        PAGE_LINE,
        SITE_LINE,
        RESEARCH_LINE,
        'To use one: component(action="use", name="web/page", path="<folder>", values={...})',
    ])
    assert components.search("web page", tiny) == expected


def test_one_hit_reads_as_a_single(tiny: components.Library):
    expected = "\n".join([
        '1 component matches "stylesheet":',
        SITE_LINE,
        'To use one: component(action="use", name="web/site", path="<folder>", values={...})',
    ])
    assert components.search("stylesheet", tiny) == expected


def test_search_says_so_when_nothing_matches(tiny: components.Library):
    assert components.search("kubernetes", tiny) == (
        'No component matches "kubernetes". Try fewer or other words, or component(action="list") to see all.')


def test_a_recipe_hit_needs_no_folder(tiny: components.Library):
    expected = "\n".join([
        '1 component matches "several sources":',
        RESEARCH_LINE,
        'To use one: component(action="use", name="browser/research")',
    ])
    assert components.search("several sources", tiny) == expected


def test_list_shows_every_component(tiny: components.Library):
    expected = "\n".join(["3 components:", RESEARCH_LINE, PAGE_LINE, SITE_LINE, LIST_HINT])
    assert components.catalogue(None, tiny) == expected


def test_list_narrows_to_an_area(tiny: components.Library):
    expected = "\n".join(["2 components:", PAGE_LINE, SITE_LINE, LIST_HINT])
    assert components.catalogue("web", tiny) == expected


def test_show_prints_a_kit(tiny: components.Library):
    expected = "\n".join([
        "web/page (kit) - One responsive page: semantic HTML, phone-first CSS with design tokens, no framework.",
        "When: Use when the project needs a single public page such as a landing, info or documentation page.",
        'Blanks: title (default "Project page"): The page title, shown in the browser tab, the header and the main '
        'heading.; description (default "A small, fast page that stays readable on phones and desktops."): One '
        "sentence that describes the page; used as the meta description and the line under the heading.",
        "Files: index.html, README.md, styles.css",
        f"Next: {PAGE_NEXT}",
    ])
    assert components.show("web/page", tiny) == expected


def test_show_prints_a_recipe(tiny: components.Library):
    expected = "\n".join([
        "browser/research (recipe) - Answer a question from several sources: search with two or three queries, read "
        "the best pages, cite numbered sources.",
        "Steps:",
        *(f"{number}. {step}" for number, step in enumerate(RESEARCH_STEPS, 1)),
        f"Then: {RESEARCH_NEXT}",
    ])
    assert components.show("browser/research", tiny) == expected


def test_use_of_a_recipe_prints_its_steps(tmp_path: Path, tiny: components.Library):
    text = components.use("browser/research", workspace=Workspace(tmp_path / "ws"), library=tiny)
    assert text == components.show("browser/research", tiny)


def test_use_writes_a_kit_into_a_folder(tmp_path: Path, tiny: components.Library):
    ws = Workspace(tmp_path / "ws")
    text = components.use(
        "web/page", {"title": "Recipe box", "description": "Our recipes, by season."}, workspace=ws, path="site",
        library=tiny,
    )
    assert sorted(kit_files(ws.root)) == ["site/README.md", "site/index.html", "site/styles.css"]
    page = (ws.root / "site" / "index.html").read_text(encoding="utf-8")
    assert "Recipe box" in page and "Our recipes, by season." in page
    assert "{{title}}" not in page and "Project page" not in page
    expected = "\n".join([
        "Used web/page in site: 3 files written",
        f"site/index.html ({len(page.splitlines())} lines)",
        "site/README.md (9 lines)",
        f"site/styles.css ({len((ws.root / 'site' / 'styles.css').read_text(encoding='utf-8').splitlines())} lines)",
        f"Next: {PAGE_NEXT}",
    ])
    assert text == expected


def test_use_without_a_path_writes_at_the_top(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    text = components.use("web/page", {}, workspace=ws)
    assert text.splitlines()[0] == "Used web/page in the workspace: 3 files written"
    assert "index.html" in text.splitlines()[1]
    assert sorted(kit_files(ws.root)) == ["README.md", "index.html", "styles.css"]


def test_use_keeps_the_executable_bit(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    components.use("script/bash", {"name": "report", "purpose": "Count the rows"}, workspace=ws)
    script = ws.root / "report.sh"
    assert script.is_file() and "Count the rows" in script.read_text(encoding="utf-8")
    assert os.access(script, os.X_OK)
    assert not os.access(ws.root / "README.md", os.X_OK)


def test_a_second_use_into_the_same_folder_changes_nothing(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    components.use("web/page", {"title": "Keep me"}, workspace=ws, path="site")
    before = {name: (ws.root / name).read_bytes() for name in kit_files(ws.root)}
    with pytest.raises(components.ComponentError) as caught:
        components.use("web/page", {"title": "Other"}, workspace=ws, path="site")
    assert str(caught.value) == (
        "site/index.html already exists. Nothing was written. Choose another folder with path, or move that file "
        "first.")
    assert {name: (ws.root / name).read_bytes() for name in kit_files(ws.root)} == before


def test_a_kit_is_never_written_halfway(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "site").mkdir(parents=True)
    (ws.root / "site" / "styles.css").write_text("mine\n", encoding="utf-8")
    with pytest.raises(components.ComponentError) as caught:
        components.use("web/page", {}, workspace=ws, path="site")
    assert str(caught.value).startswith("site/styles.css already exists.")
    assert not (ws.root / "site" / "index.html").exists()
    assert (ws.root / "site" / "styles.css").read_text(encoding="utf-8") == "mine\n"


# --- errors ---------------------------------------------------------------------------


def test_an_unknown_component_names_its_closest():
    with pytest.raises(components.ComponentError) as caught:
        components.default_library().get("web/pag")
    assert str(caught.value) == (
        'there is no component named "web/pag". Closest: web/page, web/pwa. component(action="list") shows all.')


def test_an_unknown_blank_lists_the_real_ones(tmp_path: Path):
    with pytest.raises(components.ComponentError) as caught:
        components.use("web/page", {"colour": "red"}, workspace=Workspace(tmp_path / "ws"))
    assert str(caught.value) == 'web/page has no blank named "colour". Its blanks: title, description.'


def test_a_missing_word_says_what_is_missing(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    for call, text in (
        (lambda: components.search("  "), "search needs query."),
        (lambda: components.show(""), "show needs name."),
        (lambda: components.use("", workspace=ws), "use needs name."),
    ):
        with pytest.raises(components.ComponentError) as caught:
            call()
        assert str(caught.value) == text


def test_a_value_in_a_file_name_cannot_carry_a_path(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    for value in ("a/b", ".hidden"):
        with pytest.raises(components.ComponentError) as caught:
            components.use("script/bash", {"name": value}, workspace=ws)
        assert str(caught.value) == (
            'the value for "name" is used in a file name and cannot contain "/" or start with a dot.')
    assert not kit_files(ws.root)


def test_a_folder_outside_the_workspace_is_refused(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    with pytest.raises(components.ComponentError) as caught:
        components.use("web/page", {}, workspace=ws, path="../outside")
    assert "outside the workspace" in str(caught.value)
    assert not kit_files(ws.root)


def test_undo_takes_the_written_files_back(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    data = tmp_path / "data"
    components.use("web/page", {"title": "Draft"}, workspace=ws, path="site", conversation_id="c1", data_dir=data)
    assert (ws.root / "site" / "index.html").exists()
    reports = checkpoints.store_for(data).undo("c1")
    assert reports and {report["action"] for report in reports} == {"removed (it did not exist before)"}
    assert not (ws.root / "site" / "index.html").exists()


# --- the tool layer -------------------------------------------------------------------


def test_the_tool_is_registered(train):
    defs = {item["function"]["name"]: item["function"] for item in tools.tool_defs()}
    assert defs["component"] == train.COMPONENT_TOOL["function"]
    assert "component" in tools.select_tool_names("auto")
    assert "component" in tools.select_tool_names("full")
    assert "component" in config.DEFAULT_PERMISSIONS


def test_the_tool_validates_its_arguments():
    assert tools.validate_args("component", {"action": "SEARCH", "query": " web page "}) == {
        "action": "search", "query": "web page", "name": None, "path": None, "values": {},
    }
    assert tools.validate_args("component", {"action": "use", "name": "web/page", "values": {"title": "Box"}}) == {
        "action": "use", "query": None, "name": "web/page", "path": None, "values": {"title": "Box"},
    }
    for bad in ({"action": "browse"}, {"action": "use", "values": ["title"]},
                {"action": "use", "values": {"title": 3}}):
        with pytest.raises(ValueError):
            tools.validate_args("component", bad)


def test_the_tool_answers_through_execute_tool(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    ctx = ToolContext(workspace=ws, memory=None, search=None, conversation_id="c1")  # type: ignore[arg-type]
    call = tools.validate_args("component", {"action": "search", "query": "web page"})
    found = tools.execute_tool("component", call, ctx)
    assert found.ok and found.content == components.search("web page")
    used = tools.execute_tool(
        "component", tools.validate_args("component", {"action": "use", "name": "web/page", "path": "site"}), ctx)
    assert used.ok and used.content.startswith("Used web/page in site: 3 files written")
    broken = tools.execute_tool(
        "component", tools.validate_args("component", {"action": "show", "name": "web/nothing"}), ctx)
    assert not broken.ok and broken.content.startswith("Error: there is no component named")


# --- safety ---------------------------------------------------------------------------


def test_only_a_kit_use_is_graded_as_a_write(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    (ws.root / "site").mkdir(parents=True)
    (ws.root / "site" / "index.html").write_text("mine\n", encoding="utf-8")
    reads = (
        {"action": "search", "query": "web page"},
        {"action": "list"},
        {"action": "show", "name": "web/page"},
        {"action": "use", "name": "browser/research"},
    )
    for args in reads:
        graded = safety.assess("component", args, workspace=ws.root, spec_risk="write")
        assert graded.kind == "read" and graded.level == "safe", args
    fresh = safety.assess("component", {"action": "use", "name": "web/page", "path": "new/page"},
                          workspace=ws.root, spec_risk="write")
    new_file = safety.assess("write_file", {"path": "new/page/index.html", "content": ""},
                             workspace=ws.root, spec_risk="write")
    assert (fresh.kind, fresh.level) == (new_file.kind, new_file.level) == ("write", "low")
    existing = safety.assess("component", {"action": "use", "name": "web/page", "path": "site"},
                             workspace=ws.root, spec_risk="write")
    edited = safety.assess("write_file", {"path": "site/index.html", "content": ""},
                           workspace=ws.root, spec_risk="write")
    assert (existing.kind, existing.level) == (edited.kind, edited.level) == ("write", "medium")
    outside = safety.assess("component", {"action": "use", "name": "web/page", "path": str(Path.home() / "elsewhere")},
                            workspace=ws.root, spec_risk="write")
    assert outside.level == "high"


def test_the_user_is_asked_like_for_any_other_file_write(tmp_path: Path):
    ws = Workspace(tmp_path / "ws")
    settings = {"autonomy": "supervised", "permissions": {}}
    kit = safety.decide(settings, "component", {"action": "use", "name": "web/page", "path": "site"},
                        workspace=ws.root, risk="write")
    written = safety.decide(settings, "write_file", {"path": "site/index.html", "content": ""},
                            workspace=ws.root, risk="write")
    assert kit.mode == written.mode == "ask"
    default = safety.decide({}, "component", {"action": "search", "query": "web page"}, workspace=ws.root, risk="write")
    assert default.mode == "auto"
