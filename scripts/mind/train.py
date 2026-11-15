#!/usr/bin/env python3
"""Train Kiwi Mind, Kiwibro's own small model stack, on a GPU machine (written for Google Colab).

Stages, each a sub-command:

  gen     a large teacher model writes conversations for every skill from Kiwibro's real tool list
  trunk   Qwen3-0.6B is fine-tuned on all of them with its weights held on the 4-bit grid the
          engine stores (quantisation-aware training), then snapped to that grid
  skills  one low-rank expertise layer per skill is trained on the frozen trunk, only in the
          layers from SPLIT up, so the layers below are shared by every skill
  head    the control head learns to pick the skill and to flag heavy requests from the state
          the shared layers produce
  eval    held-out numbers: loss per skill with and without its layer, tool-call accuracy
  export  packs the trunk, the skills and the head into the files the engine reads

The conversations are rendered with kiwibro/mind/render.py, the same code Kiwibro uses at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("KIWIMIND_WORK", "/content/km"))
BASE = os.environ.get("KIWIMIND_BASE", "/content/models/base")
TEACHER = os.environ.get("KIWIMIND_TEACHER", "/content/models/teacher")
HERE = Path(__file__).resolve().parent

SKILLS = ("chat", "web", "code", "terminal", "computer", "tools", "write")
HEAD_NAMES = [*SKILLS, "hard"]
SPLIT = 16  # skills change layers SPLIT.. only; the layers below are shared by all of them
RANK = 16
ALPHA = 32.0
GROUP = 32
MAX_TOKENS = 2560
CASE_BATCH = 16  # prompts answered at once when replies are generated for a report
FIRST_TOKENS = 3  # the first tokens of a reply decide between a tool call and words: they count more
WEIGHTS = (0.0, 1.0, 1.5, 3.0)  # a token that is only shown, a word of an answer, a token of a tool call, a first token
B_RATE = 4.0  # the B side of a skill starts at zero and learns this much faster than the A side (LoRA+)
SEED = 7
MATRICES = {"q": ("self_attn", "q_proj"), "k": ("self_attn", "k_proj"), "v": ("self_attn", "v_proj"),
            "o": ("self_attn", "o_proj"), "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"),
            "down": ("mlp", "down_proj")}


def load_module(name: str, *candidates: Path):
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(name, candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"{name}: none of {[str(c) for c in candidates]} exists")


RENDER = load_module("kiwimind_render", HERE / "render.py", HERE.parent.parent / "kiwibro" / "mind" / "render.py")


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ------------------------------------------------------------------ the tool catalog

COMPONENT_TOOL = {"type": "function", "function": {
    "name": "component",
    "description": ("Kiwibro's library of ready-made building blocks: kits (starter files for a web page, an API, a "
                    "desktop app, a script, a package) and recipes (proven step lists for browser, computer and "
                    "terminal jobs). Start from one instead of writing from scratch. search or list to find one, "
                    "show to read it, use to put a kit's files into the workspace or to get a recipe's steps."),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["search", "list", "show", "use"]},
        "query": {"type": "string", "description": "search: what you need, in a few words"},
        "name": {"type": "string", "description": "show/use: the component's name, e.g. web/page"},
        "path": {"type": "string", "description": "use: folder in the workspace for the files (default: its top level)"},
        "values": {"type": "object", "description": "use: values for the kit's blanks, e.g. {\"title\": \"Recipe box\"}"},
    }, "required": ["action"]}}}

COMPONENTS = [  # name, blanks, summary: the library in kiwibro/components/library implements exactly these
    ("web/page", "title, description", "One responsive page: semantic HTML, phone-first CSS with design tokens, no framework."),
    ("web/site", "site_name", "Small multi-page static site: shared header and footer, home, about and contact pages, one stylesheet."),
    ("web/app-shell", "app_name", "Single-page app shell in plain JavaScript: views, state, hash routing and a fetch helper."),
    ("web/form", "title", "Accessible form with client-side validation and clear error states."),
    ("web/flask-api", "name, resource", "Flask JSON API with SQLite, input validation and tests."),
    ("web/fastapi-service", "name, resource", "FastAPI service with typed models, a health route and tests."),
    ("web/pwa", "app_name", "Installable web app that works offline: manifest, service worker and an offline page."),
    ("web/scene-3d", "title, scene, sky, ground", "One HTML file with a three.js 3D scene made from plain lines: things (kiwi, tree, house, car and about 60 more) and what they do (walks to, runs to, flies to, drives to, follows, orbits)."),
    ("code/python-package", "name", "Python package: pyproject.toml, src layout and pytest tests."),
    ("code/python-cli", "name", "Command-line tool in Python: argparse subcommands, logging, exit codes and tests."),
    ("code/node-cli", "name", "Node.js command-line tool without dependencies: argument parsing and node:test tests."),
    ("code/sqlite-store", "table", "Small SQLite data layer in Python: schema, migrations, typed queries and tests."),
    ("code/tests-python", "module", "pytest starter for an existing module: fixtures, parametrised cases and tmp_path use."),
    ("code/c-program", "name", "C program with a Makefile, a header, warnings on and a test target."),
    ("ubuntu/gtk-app", "app_name, app_id", "Native Ubuntu app in Python with GTK 4 and libadwaita: window, header bar, actions and a desktop entry."),
    ("ubuntu/desktop-entry", "name, command", "Launcher (.desktop file), icon and an install script for any program."),
    ("ubuntu/systemd-service", "name, command, schedule", "User service and timer (systemd --user) that run a program at login or on a schedule."),
    ("ubuntu/deb-package", "name, version", "Build script and control files that package a program as a .deb."),
    ("script/bash", "name, purpose", "Bash script with strict mode, usage text, argument parsing, a cleanup trap and logging."),
    ("script/python", "name, purpose", "Single-file Python script: argparse, pathlib, logging and a main guard."),
    ("script/backup", "source, target", "Dated tar backup of a folder with a verification step; lists old backups instead of removing them."),
    ("script/batch-rename", "pattern", "Batch rename with a preview: dry run by default, applied with a flag."),
    ("script/watch-run", "command", "Run a command again whenever files change (inotifywait, with a polling fallback)."),
    ("script/cron-job", "name, schedule", "Shell job with a lock file, a size-capped log and a crontab line."),
    ("browser/research", "", "Answer a question from several sources: search with two or three queries, read the best pages, cite numbered sources."),
    ("browser/extract-table", "", "Pull a table or list from a web page into a CSV file and check the rows."),
    ("browser/follow-docs", "", "Find the official documentation page for a library or tool, read the exact section and quote the command or API."),
    ("browser/compare", "", "Compare products or options: one row per option, one column per criterion, a source for each cell."),
    ("browser/monitor-page", "", "Check a page for a change (price, version, status) and report the difference from last time, kept in a note."),
    ("computer/organise-folder", "", "Sort a messy folder into dated or typed subfolders: list, plan, make folders, move; nothing is deleted."),
    ("computer/find-files", "", "Find files by name, type, size or age and report them, largest or newest first."),
    ("computer/free-space", "", "See what takes disk space and hand the user the commands that would free it."),
    ("computer/convert-documents", "", "Convert documents between formats (Markdown, HTML, DOCX, PDF) and open the result."),
    ("computer/screenshot-explain", "", "Take a screenshot, describe what is on screen and answer a question about it."),
    ("computer/daily-brief", "", "Collect notes, open tasks and saved facts into one short summary for the day."),
    ("terminal/diagnose", "", "Find why a command or service fails: read the error, check versions, logs and ports, try one fix at a time."),
    ("terminal/project-setup", "", "Set up a project so it runs: detect the stack, create the environment, install, build, test."),
    ("terminal/git-review", "", "Review what changed in a repository: status, diff, log, risky files, then a summary."),
    ("terminal/port-and-process", "", "Find which process uses a port or the most memory, and hand the user the command that would stop it."),
    ("browser/find-download", "", "Find the official download or documentation link for a program: the maker's site first, check the version and date."),
    ("browser/news", "", "Today's news on a topic: search with dates, read dated sources, keep only items from the asked period, cite each."),
    ("browser/price-check", "", "Find what something costs now: search shops and the maker's page, read each price with its date and currency, cite."),
    ("browser/read-page", "", "Read one web page and answer from it: fetch it, find the passage that answers, quote the figure, give the URL."),
    ("code/add-feature", "", "Add a feature to existing code: read how the neighbours do it, change the fewest files, add a test, run it."),
    ("code/api-client", "name", "Python client for a JSON web API: timeout, bounded retries with backoff, clear errors, standard library only, with tests."),
    ("code/docker", "name", "Container setup for a Python web app: Dockerfile with a non-root user, compose file with a health check, ignore file."),
    ("code/explain-project", "", "Explain an unfamiliar project: entry point, folder layout, main modules and how to run and test it."),
    ("code/failing-tests", "", "Make failing tests pass: run them, read the first failure, decide whether code or test is wrong, fix, rerun."),
    ("code/fix-bug", "", "Fix a bug: reproduce it, find the line that causes it, change only that, run the tests, report what changed."),
    ("code/node-server", "name", "HTTP JSON API in Node with no dependencies: routing, input checks, error replies and node:test tests."),
    ("code/refactor", "", "Restructure code without changing what it does: tests first, one small move at a time, tests after each."),
    ("code/review", "", "Review a change or a file: read the diff, check errors, edge cases and tests, list findings by severity with lines."),
    ("code/scraper", "name", "Python page scraper on the standard library: fetch with a timeout, pull links and tables out of the HTML, save as CSV."),
    ("code/tests-node", "name", "Test starter for Node with node:test: a sample module, table-driven cases and an async case."),
    ("data/analyse-table", "", "Answer questions about a CSV or spreadsheet with exact figures: inspect columns, compute with the table tool, show the numbers."),
    ("data/csv-report", "name", "Python script that reads a CSV, checks its columns, groups and totals a column and writes a Markdown report."),
    ("data/json-tool", "name", "Python script that reads JSON, picks and renames fields, filters records and writes JSON or CSV."),
    ("ui/chart", "title", "Bar and line chart drawn as SVG from a data array: axes, labels, a text table for screen readers, no library."),
    ("ui/dashboard", "title", "Dashboard page: side navigation, stat tiles, a data table and an SVG bar chart drawn from one data file, no framework."),
    ("ui/flowchart", "title", "Flowchart drawn as one inline SVG from a list of steps and links: boxes, decisions, layered layout and loops, no library."),
    ("ui/data-table", "title", "Data table that sorts by column, filters as you type and pages its rows, from a JSON array, plain JavaScript."),
    ("ui/dialog", "title", "Accessible modal dialog and toast messages built on the dialog element: focus stays inside, Escape closes."),
    ("ui/gallery", "title", "Responsive image gallery: a grid of thumbnails and a full-size viewer with keyboard and swipe navigation."),
    ("ui/nav", "site_name", "Responsive site header: navigation links, a menu button on phones, a skip link and a mark on the current page."),
    ("ui/review", "", "Check a web page's UI: phone width, contrast, keyboard focus, states and overflow, with a fix for each finding."),
    ("ui/settings", "title", "Settings screen: grouped options with switches, selects and a text field, saved in the browser and restored on load."),
    ("write/email", "", "Draft an email or message: purpose first, the facts the user gave and nothing invented, a clear ask, a fitting tone."),
    ("write/readme", "project", "README for a project: what it is, install, use, configure, test and licence sections ready to fill."),
    ("write/report", "title", "Report in Markdown: summary first, findings with evidence, a table, open questions and numbered sources."),
    ("write/summary", "", "Summarise a long file or document: index it, read it section by section, keep figures exact, state what was left out."),
]

COMPONENT_NOTES = """The component library (tool "component")
""" + "\n".join(f"- {name} ({'kit' if blanks else 'recipe'}): {summary}" + (f" Blanks: {blanks}." if blanks else "")
                for name, blanks, summary in COMPONENTS) + """
Results of the component tool look like this.
search or list:
3 components match "web page":
web/page (kit) - One responsive page: semantic HTML, phone-first CSS with design tokens, no framework. Blanks: title, description
web/site (kit) - Small multi-page static site: shared header and footer, home, about and contact pages, one stylesheet. Blanks: site_name
To use one: component(action="use", name="web/page", path="<folder>", values={...})
use of a kit:
Used web/page in site: 3 files written
site/index.html (46 lines)
site/styles.css (112 lines)
site/script.js (18 lines)
Next: fill the places marked KIWIBRO: in the files, then open site/index.html.
show or use of a recipe:
browser/research (recipe) - Answer a question from several sources: search with two or three queries, read the best pages, cite numbered sources.
Steps:
1. Call multi_search with two or three differently worded queries.
2. ...
After using a kit the assistant adapts it with one or two edits (replace_in_file or apply_patch) and checks the result; after reading a recipe it follows the steps with the named tools."""

SKILL_TOOLS = {
    "chat": [],
    "web": ["web_search", "multi_search", "fetch_page", "fetch_pages", "web_research", "browse_page", "browser",
            "note", "component"],
    "code": ["read_file", "read_files", "list_files", "glob_files", "grep_files", "file_outline", "write_file",
             "replace_in_file", "apply_patch", "run_command", "multi_command", "run_python", "run_javascript", "git",
             "undo_changes", "component"],
    "terminal": ["run_command", "multi_command", "terminal_session", "system_info", "disk_usage", "remote", "git",
                 "write_file", "read_file", "component"],
    "computer": ["system_info", "disk_usage", "open_path", "clipboard", "notify", "screenshot", "describe_image",
                 "file_manage", "list_files", "workspace", "convert_document", "model_info", "component"],
    "tools": ["remember", "recall", "note", "task_board", "calculate", "analyze_table", "doc_search", "doc_read",
              "doc_add", "doc_list", "study_file", "delegate"],
    "write": ["write_file", "read_file", "note"],
}
FORBIDDEN_CHOICES = {("file_manage", "action"): {"delete"}}  # the teacher never shows a deletion


def load_catalog(path: Path) -> dict[str, dict]:
    tools = json.loads(path.read_text())
    catalog = {tool["function"]["name"]: tool for tool in tools}
    catalog.setdefault("component", COMPONENT_TOOL)
    return catalog


def strict_schema(tool: str, node: dict, key: str = "") -> dict:
    """A parameter schema reduced to what a decoding grammar enforces."""
    if not isinstance(node, dict):
        return {"type": "string"}
    kind = node.get("type")
    if isinstance(kind, list):
        kind = next((one for one in kind if one != "null"), "string")
    if node.get("enum"):
        banned = FORBIDDEN_CHOICES.get((tool, key), set())
        return {"type": kind or "string", "enum": [choice for choice in node["enum"] if choice not in banned]}
    if kind == "object" or "properties" in node:
        properties = node.get("properties") or {}
        if not properties and key:
            return {"type": "object"}
        return {"type": "object", "properties": {name: strict_schema(tool, spec, name) for name, spec in properties.items()},
                "required": [name for name in node.get("required") or [] if name in properties],
                "additionalProperties": False}
    if kind == "array":
        return {"type": "array", "items": strict_schema(tool, node.get("items") or {"type": "string"}, key)}
    return {"type": kind if kind in ("string", "integer", "number", "boolean") else "string"}


def steps_schema(names: list[str], catalog: dict, mode: str, most: int = 4, least: int = 0) -> dict:
    """The tool steps of one turn: calls to the named tools only, with their real parameters in strict mode."""
    if mode == "strict":
        call = {"anyOf": [{"type": "object", "properties": {
            "name": {"type": "string", "enum": [name]},
            "arguments": strict_schema(name, catalog[name]["function"].get("parameters") or {})},
            "required": ["name", "arguments"], "additionalProperties": False} for name in names]}
    else:
        call = {"type": "object", "properties": {"name": {"type": "string", "enum": names},
                                                 "arguments": {"type": "object"}},
                "required": ["name", "arguments"]}
    step = {"type": "object", "properties": {
        "calls": {"type": "array", "items": call, "minItems": 1, "maxItems": 2},
        "results": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}},
        "required": ["calls", "results"], "additionalProperties": False}
    steps = {"type": "array", "items": step, "maxItems": most}
    if least:
        steps["minItems"] = least
    return steps


def family_tools(skill: str, other: str = "") -> list[str]:
    """The tools a teacher prompt shows: the skill's own, and a second skill's when a conversation crosses over."""
    names = list(SKILL_TOOLS[skill])
    return names + [name for name in SKILL_TOOLS.get(other, []) if name not in names]


def family_schema(family: str, names: list[str], catalog: dict, count: int, mode: str) -> dict:
    """What a teacher reply must look like for the families beyond the single request:
    a session of several turns, a repair after a bad first reply, a real message from a data set."""
    text = {"type": "string"}

    def thing(fields: dict) -> dict:
        return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}

    if family == "session":
        turn = {"user": text}
        if names:
            turn["steps"] = steps_schema(names, catalog, mode, most=3)
        turn["answer"] = text
        one = thing({"turns": {"type": "array", "items": thing(turn), "minItems": 2, "maxItems": 4}})
    elif family == "repair":
        one = thing({"user": text, "bad_answer": text, "objection": text,
                     "steps": steps_schema(names, catalog, mode, most=3, least=1), "answer": text})
    else:
        one = thing({"user": text, "work": {"type": "string", "enum": ["chat", "write", "web"]},
                     "steps": steps_schema(names, catalog, mode, most=3), "answer": text})
    return thing({"conversations": {"type": "array", "items": one, "minItems": count, "maxItems": count}})


def conversation_schema(skill: str, catalog: dict, count: int, mode: str) -> dict:
    fields: dict[str, dict] = {"earlier": {"type": "string"}, "user": {"type": "string"},
                               "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}
    names = SKILL_TOOLS[skill]
    if names:
        fields["steps"] = steps_schema(names, catalog, mode)
    fields["answer"] = {"type": "string"}
    fields["followup_user"] = {"type": "string"}
    fields["followup_answer"] = {"type": "string"}
    one = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    return {"type": "object", "properties": {"conversations": {"type": "array", "items": one, "minItems": count,
                                                               "maxItems": count}},
            "required": ["conversations"], "additionalProperties": False}


# ------------------------------------------------------------------ scenario seeds (all made up)

TOPICS = ["sourdough baking", "the James Webb telescope", "compound interest", "learning the guitar",
          "houseplants in low light", "sleep and caffeine", "how vaccines work", "the Ottoman Empire", "electric cars",
          "marathon training", "black holes", "stock market basics", "Turkish grammar", "chess openings",
          "home espresso", "recycling plastics", "Roman aqueducts", "password managers", "bird migration",
          "the metric system", "volcanoes", "budget travel in Japan", "cat behaviour", "photosynthesis",
          "the history of the internet", "healthy breakfasts", "how planes fly", "the French Revolution",
          "learning to swim as an adult", "mechanical keyboards", "tea varieties", "earthquakes", "renewable energy",
          "job interviews", "raising a puppy", "the Silk Road", "meditation", "3D printing", "the periodic table",
          "public speaking", "football tactics", "cycling to work", "the moon landing", "saving for a house",
          "spring allergies", "writing a CV", "the Mediterranean diet", "time zones", "photography basics",
          "why the sky is blue", "bees and pollination", "board games for two", "tides", "first aid for burns",
          "inflation", "the printing press", "city gardening", "probability in card games", "Linux file permissions",
          "how DNS works", "HTTP status codes", "Git branching", "what an API is", "recursion"]
LIFE = ["changing jobs", "moving to another city", "a noisy flatmate", "exam week", "a first marathon",
        "learning to code at 35", "a tight monthly budget", "starting a small online shop", "taking a gap year",
        "buying a used car", "burnout at work", "asking for a raise", "making friends in a new town"]
WEBTHINGS = ["the latest stable Python release", "the newest Ubuntu LTS point release", "the price of a Raspberry Pi 5",
             "the current Node.js LTS version", "the newest Blender version", "the euro to Turkish lira rate",
             "the latest Rust stable version", "the next total solar eclipse visible in Europe",
             "the latest Django release", "the men's marathon world record", "the newest PostgreSQL major version",
             "the latest Firefox release", "the price of Bitcoin", "the latest LibreOffice version",
             "the newest Linux kernel release", "the next Formula 1 race", "the latest Godot release",
             "the most recent Nobel Prize in Physics", "the newest GNOME release", "the Bank of England interest rate",
             "the latest Kubernetes release", "the next SpaceX launch", "the latest numpy release",
             "the train times between Ankara and Istanbul", "the newest Fairphone model"]
FIELDS = ["space exploration", "open-source software", "electric vehicles", "European football", "AI research",
          "climate policy", "smartphones", "video games", "medicine", "cybersecurity", "renewable energy", "chess"]
PAIRS = ["Fedora and Ubuntu for a developer laptop", "SQLite and PostgreSQL for a small web app", "FastAPI and Flask",
         "a Kindle and a Kobo", "Vue and React for a beginner", "two small VPS providers", "Obsidian and Logseq",
         "ext4 and btrfs", "Proton Mail and Tutanota", "Podman and Docker", "two budget Android phones",
         "two entry-level road bikes"]
TECH = ["pytest fixtures", "CSS grid", "systemd user timers", "Python argparse", "git rebase", "SQLite full-text search",
        "the Fetch API", "GTK 4 with Python", "bash arrays", "Flask blueprints", "Rust error handling",
        "an nginx reverse proxy", "cutting a clip with ffmpeg", "rsync over SSH", "Docker Compose volumes",
        "Python virtual environments", "TypeScript generics", "service workers", "cron syntax", "pandas groupby",
        "the desktop entry format", "curl with JSON", "Go modules", "JavaScript regular expressions", "meson"]
PROJECTS = ["a Flask to-do API (app.py, models.py, tests/test_api.py, requirements.txt)",
            "a static blog built by a Python script (build.py, posts/*.md, templates/base.html)",
            "a Node.js chat bot (index.js, commands/, package.json)",
            "a React shopping-cart app (src/App.jsx, src/components/, package.json)",
            "a Rust tool that renames photos (Cargo.toml, src/main.rs)",
            "a data-cleaning project (clean.py, data/sales.csv, README.md)",
            "a Django library site (manage.py, library/models.py, library/views.py)",
            "a Go URL shortener (main.go, store/store.go, go.mod)",
            "a budget exporter (budget.py, exports/, config.toml)",
            "a C program that parses sensor logs (parser.c, parser.h, Makefile)",
            "a backup toolkit in shell (backup.sh, restore.sh, crontab.txt)",
            "a FastAPI weather proxy (main.py, routers/forecast.py, Dockerfile)",
            "a Vue recipe app (src/views/, src/store.js, package.json)",
            "a LaTeX thesis (thesis.tex, chapters/, refs.bib)",
            "a unit-conversion package (pyproject.toml, src/convert/, tests/)",
            "a small Pygame game (game.py, assets/, levels/level1.json)",
            "a TypeScript Express server (src/server.ts, src/routes/, tsconfig.json)",
            "a price scraper (scrape.py, output/prices.json)",
            "a GTK notes app (notes/window.py, notes/store.py, data/org.example.Notes.desktop)",
            "a Markdown notes folder (inbox.md, projects/, journal/2026-11.md)"]
NEW_THINGS = ["a landing page for a bakery", "a personal portfolio site", "a habit-tracker web app", "a contact form",
              "a small JSON API for books", "an offline-capable shopping-list app", "a Python package for date maths",
              "a command-line tool that tidies downloads", "a Node tool that checks links", "a GTK app for quick notes",
              "a launcher for a script", "a nightly job that exports data", "a .deb for a small tool",
              "a backup script for the Documents folder", "a script that renames holiday photos",
              "a watcher that reruns tests on save", "a C tool that counts words", "tests for an existing module"]
SCRIPT_JOBS = ["rename photos by date", "count words in every Markdown file", "merge several CSV files",
               "find duplicate files by hash", "convert a folder of PNGs to JPEG", "check which URLs in a list are down",
               "extract email addresses from a text file", "print the ten largest files", "turn a JSON file into CSV",
               "sum a column of a spreadsheet export", "split a large log by day", "validate a YAML config"]
LANGS = ["Python", "JavaScript", "Bash", "Rust", "Go", "C", "SQL", "TypeScript", "CSS", "HTML"]
SYSTEM_QUESTIONS = ["how much disk space is free", "which process uses the most memory", "what listens on port 8000",
                    "which Ubuntu version this is", "how long the machine has been up", "how big the Downloads folder is",
                    "which Python and Node versions are installed", "whether a service is running",
                    "how many CPU cores there are", "what the local IP address is"]
SHELL_JOBS = ["find all files larger than 100 MB", "count lines of code by file type", "show the last 20 lines of a log",
              "list the ten most recent files", "create a tar.gz of a folder", "replace a word in every .txt file",
              "find TODO comments in a project", "make every .sh file executable", "show git branches sorted by date",
              "download a file and check its hash", "set up a virtual environment and install requirements",
              "sort a CSV by its second column"]
EMAILS = ["asking a landlord to fix the heating", "declining a meeting politely", "following up on a job application",
          "asking a professor for an extension", "thanking a colleague", "requesting a refund for a late delivery",
          "introducing yourself to a new team", "asking a carpenter for a quote", "apologising for a missed deadline",
          "inviting neighbours to a street party", "resigning from a part-time job", "asking a client for overdue payment"]
DELETE_JOBS = ["delete the old build folder", "remove all .log files", "clean out the Downloads folder",
               "delete a stale git branch", "remove node_modules to start fresh", "drop a database table",
               "uninstall a package", "empty the cache directory", "delete duplicate photos",
               "remove every file older than a year", "wipe the temp files of a project", "delete an unused module"]
VOICES = ["one short lowercase line without punctuation", "a polite full sentence", "terse, like a command",
          "two sentences: the request and the reason", "casual, with a typo or two", "starts mid-thought (\"ok so\")",
          "precise: names exact files, numbers or formats", "a plain question", "simple words, slightly broken English",
          "three sentences with some background"]
FOLDERS = ["~/Kiwibro", "~/projects/todo-api", "~/code/blog", "~/Desktop/shop-app", "~/dev/renamer", "~/work/reports",
           "~/Documents/thesis", "~/src/weather-proxy", "~/Kiwibro/notes", "~/projects/game", "~/code/library-site"]

# What Kiwibro's web tools return, and what its context note says, while there is no connection.
OFFLINE_ERROR = ("Error: no internet connection right now. Web tools cannot reach anything until it is back; "
                 "local files, notes, saved memory and indexed documents still work.")
OFFLINE_LINE = ("Internet: not reachable right now. Web tools will fail until it is back: answer from what you know and "
                "from local files, notes, memory and documents, and say what may be outdated.")
WEB_TOOLS = set(SKILL_TOOLS["web"]) - {"note", "component"}

# (weight, kind, text). kind: "" | hard | delete | pre | exact | facts | comp | offline | offnote
TASKS = {
    "chat": [
        (5, "", "asks a factual question about {topic} that general knowledge answers in two or three sentences"),
        (4, "", "wants a simple explanation of one idea in {topic} for a beginner"),
        (3, "", "greets Kiwibro or makes small talk; the answer is one or two warm sentences"),
        (2, "", "asks who Kiwibro is or what it can do (an AI agent running on the user's own computer, with tools for the web, files, commands, notes and memory)"),
        (3, "", "asks for advice about {life}"),
        (3, "", "asks for a short list (3 to 5 items) about {topic}"),
        (3, "", "asks what time or date it is, or how many days until a date; the answer comes from the local time given in the context"),
        (4, "facts", "asks something where the saved facts about the user matter (their name, city, job, pet or a preference)"),
        (4, "exact", "asks a calculation in their own words; the context already carries the exact result and the answer states it as it is"),
        (3, "", "asks which of two options in {topic} to pick and gets a brief, honest comparison"),
        (2, "", "asks the meaning, spelling or usage of a word or phrase"),
        (3, "", "asks something the assistant cannot know (private, in the future or made up); it says so plainly and offers what it can do"),
        (2, "", "vents about {life}; the assistant answers kindly and briefly, without a lecture"),
        (3, "", "asks how to do something about {topic}; the answer is 3 to 5 short steps"),
        (2, "", "asks for a joke, a fun fact or a riddle about {topic}"),
        (3, "", "writes in Turkish, Spanish, German or French about {topic}; the assistant answers in that language"),
        (2, "", "asks a quick programming question about {lang} that needs no tool; the answer has a short code block"),
        (2, "offnote", "asks about {topic} while the internet is down, and Kiwibro already knows it is down: the question needs no web, so the assistant simply answers from what it knows, without mentioning the connection"),
        (2, "long", "is deep in a long conversation about {topic} and refers back to something the earlier record covers; the record is enough, so the assistant answers from it"),
        (3, "hard", "asks a multi-step logic or maths word problem that needs careful reasoning; the answer shows the steps"),
        (3, "hard", "asks for a deep comparison with trade-offs about {topic}, several paragraphs"),
        (2, "hard", "asks for a detailed plan with many constraints (a study plan, a four-week training plan, a project roadmap)"),
    ],
    "web": [
        (5, "", "wants {webthing}; one web_search, then the fact with its source"),
        (4, "", "asks about recent events in {field}; search, then answer from the results with sources"),
        (3, "", "gives a URL and asks what the page says; fetch_page, then a short summary"),
        (3, "", "wants {pair} compared; multi_search with two or three queries, then a short comparison"),
        (3, "", "asks how to do something with {tech}; search, fetch the documentation page, answer with the steps"),
        (3, "", "wants a researched answer with sources about {topic}; web_research in one call"),
        (2, "", "wants a site read step by step: browse_page, follow one numbered link with browse_page again, answer"),
        (5, "pre", "asks for {webthing}; Kiwibro already searched before the answer, so the only step is that web_search and the assistant answers from its results"),
        (2, "", "asks something where the first search returns nothing useful; the assistant tries one different query and then says honestly what it found"),
        (2, "", "asks for two unrelated current facts at once; multi_search, then a short list"),
        (2, "", "wants a claim about {topic} checked; search, compare two sources, give a verdict with links"),
        (2, "", "wants one table or section pulled from a page with the browser tool (action open with a selector, or inspect)"),
        (2, "", "wants the findings kept: search, then save them with the note tool, then confirm"),
        (2, "comp", "asks for a job that matches a browser recipe (research, compare, extract a table, follow the docs); the assistant reads the recipe with the component tool, then follows its first steps"),
        (3, "offline", "asks for {webthing} while the internet is down: the one web call returns exactly \"" + OFFLINE_ERROR + "\"; the assistant does not try again, says there is no connection right now, gives what it knows from memory marked as possibly outdated, and offers to check again later"),
        (3, "offnote", "asks for {webthing} or about recent events in {field} while the internet is down, and Kiwibro already knows it is down: the assistant makes no tool call, says it cannot check right now, gives what it knows from memory marked as possibly outdated, and offers to look once the connection is back"),
        (3, "hard", "wants a thorough research report comparing several sources about {topic}"),
    ],
    "code": [
        (5, "", "wants a bug fixed in {project}: read the file, change it with replace_in_file or apply_patch, run the tests"),
        (4, "", "wants a small function or endpoint added to {project}: file_outline or read_file first, then apply_patch, then run it"),
        (3, "", "asks what a file or function in {project} does: read_file or file_outline, then a brief explanation"),
        (3, "", "wants to know where something is defined or used in {project}: grep_files, answer with file and line"),
        (3, "", "wants a new file in {project} (a test, a config, a helper): write_file, then confirm what it contains"),
        (3, "", "asks for a quick script to {scriptjob}: write_file, run_command, show the output"),
        (2, "", "wants an idea checked by running code: run_python or run_javascript, then explain the result"),
        (2, "", "wants a rename across two files in {project}: grep_files, read_files, apply_patch"),
        (3, "", "tests fail in {project}: run them, read the failing file, fix it, run again"),
        (2, "", "asks for a review of one file in {project}: read_file, then three concrete findings, no edits"),
        (2, "", "wants to see what changed in the repository: git status, diff or log, then a summary"),
        (1, "", "asks to undo the last edit: undo_changes, confirm"),
        (2, "", "names a file that does not exist in {project}: the read fails, the assistant finds the right file with glob_files or list_files and continues"),
        (2, "", "asks a programming question about {lang} that needs no tool; the answer has a short code block"),
        (6, "comp", "wants {newthing} started: the assistant finds the fitting kit (component search, or use directly when the name is obvious), uses it into a folder with values for its blanks, adapts one file, and says what was created"),
        (3, "delete", "asks to {deletejob} in {project}"),
        (3, "hard", "wants a larger feature designed and built in {project} across several files"),
        (2, "hard", "asks for a tricky algorithm implemented with tests"),
    ],
    "terminal": [
        (5, "", "wants to know {sysq}: run_command or system_info, answer with the number"),
        (4, "", "wants a command run in {project} (install, build, lint or test): run_command, report briefly"),
        (3, "", "wants a build and then the tests: multi_command, report"),
        (3, "", "wants a development server started and checked: terminal_session start, then status, report the address"),
        (4, "", "wants to {shelljob}: run_command, show the result"),
        (3, "", "runs into a failing command (a missing package, a permission error, a typo): the assistant reads the error and fixes it or says what is needed"),
        (2, "", "wants logs or files searched with grep, find or awk: run_command, summarise the matches"),
        (2, "", "wants git work done in the terminal (a new branch, a commit, a stash): run_command, confirm"),
        (2, "", "wants a long job watched: terminal_session start, status with wait_ms, report"),
        (2, "", "wants a server checked over SSH (uptime, disk, a service): remote with action run, report"),
        (2, "", "wants versions and environment details: multi_command, a short list"),
        (2, "", "wants the size of a folder or to see what fills it: disk_usage with that path, answer with the biggest entries"),
        (2, "", "asks what a shell command does; it is explained without running it"),
        (2, "", "wants a package installed or a repository cloned while the internet is down: the command fails with a network error (could not resolve host); the assistant says the connection is the cause and what can still be done without it"),
        (3, "comp", "wants {newthing} set up as a script or service: the assistant uses the fitting kit from the component library, makes it executable or checks it, and reports"),
        (2, "comp", "has a problem that matches a terminal recipe (diagnose a failure, set up a project, find what uses a port): the assistant reads the recipe with the component tool and follows its first steps"),
        (5, "delete", "asks to {deletejob}"),
        (2, "hard", "asks for a robust multi-step deployment or backup script with error handling"),
    ],
    "computer": [
        (4, "", "wants a file, folder or web address opened on the desktop: open_path"),
        (3, "", "wants the clipboard read, or a text copied to it: clipboard"),
        (3, "", "wants a screenshot taken and described: screenshot, then describe_image"),
        (3, "", "asks about this computer (CPU, load, disk, desktop session): system_info"),
        (5, "", "wants to know what takes up the disk, which folders are the biggest, or what filled the disk today, this week or lately: disk_usage (with changed when a time is named, with path when a folder is named), then the largest entries with their sizes and, when a time was named, the largest new files"),
        (2, "", "wants a desktop notification now: notify"),
        (4, "", "wants files sorted into folders: list_files, file_manage mkdir, file_manage move, list_files to confirm"),
        (3, "", "wants Kiwibro to work in another folder they name: workspace with action switch, then list_files"),
        (3, "", "wants a document converted (Markdown to HTML, DOCX to PDF): convert_document, then open_path"),
        (2, "", "asks which models are installed: model_info"),
        (2, "", "asks about an image in the workspace: describe_image"),
        (2, "", "wants a file renamed or copied before editing: file_manage move or copy"),
        (3, "comp", "has an everyday job that matches a computer recipe (organise a folder, find files, free space, a daily brief): the assistant reads the recipe with the component tool and follows its first steps"),
        (2, "comp", "wants {newthing} for their Ubuntu desktop: the assistant uses the fitting kit from the component library and says what was created and how to try it"),
        (4, "delete", "asks to {deletejob} on their computer"),
        (2, "", "asks for something the tools cannot do on the desktop (click inside an app, change the wallpaper, log in to a site); the assistant says so plainly and offers what it can do"),
    ],
    "tools": [
        (3, "facts", "tells Kiwibro a fact about themselves; the saved facts already show it, so the assistant confirms in one short sentence without a tool"),
        (2, "", "corrects a remembered fact: remember with action update and the old wording"),
        (3, "", "asks what Kiwibro remembers about something or an earlier conversation: recall, then answer"),
        (4, "", "wants a note saved, extended, read or searched: note"),
        (4, "", "wants a to-do list handled: task_board add, update, list or summary"),
        (4, "", "asks for exact maths, a unit conversion or date arithmetic: calculate, give the result as returned"),
        (3, "", "wants a CSV or JSON table analysed (filter, group, aggregate): analyze_table, then one or two sentences"),
        (3, "", "asks about the contents of their documents: doc_search, then doc_read, answer naming the document"),
        (2, "", "wants a document or folder indexed, or the indexed documents listed: doc_add or doc_list"),
        (2, "", "asks a question about one long file: study_file"),
        (1, "", "wants a second opinion from another local model: delegate with a capability"),
        (3, "", "wants two things at once (save a note and add a task; calculate and note the result): two calls in one step"),
        (2, "", "wants a multi-step job planned: task_board add for several tasks with priorities, then summary"),
        (2, "", "hits a tool error (note not found, a missing table column): the assistant fixes the call or says what is missing"),
        (2, "offnote", "asks a question their own notes or indexed documents can answer while the internet is down, and Kiwibro already knows it is down: note search or doc_search, then the answer naming the local source"),
        (4, "long", "is deep in a long conversation and asks for an exact detail from much earlier that the earlier record only mentions (a number, a file name, a command's output, a decision): recall with a few words to find or with \"step N\", then the exact answer. The recall result lists exact excerpts, one per line, like \"- [this chat, step 14, tool result] <exact text>\""),
    ],
    "write": [
        (5, "", "wants an email written: {email}; 80 to 150 words, ready to send"),
        (4, "", "pastes a rough paragraph (60 to 100 words, included in the message) and wants it clearer, shorter, more formal or friendlier"),
        (4, "", "pastes a text of 120 to 180 words and wants a summary in three bullet points or two sentences"),
        (3, "", "wants a short text translated to or from Turkish, Spanish, German or French; the text is in the message"),
        (3, "", "wants a short social post, product description, bio or announcement about {topic}"),
        (3, "", "wants an outline for an article, talk or report about {topic}"),
        (3, "", "wants grammar and spelling fixed in a pasted paragraph; the corrected text comes back"),
        (2, "", "wants a cover-letter paragraph, a reference, a complaint or a thank-you note"),
        (2, "", "wants five title or name ideas about {topic}"),
        (2, "", "wants a short poem, story or toast (under 120 words) about {topic}"),
        (3, "", "asks for a short poem, a haiku or a rhyme and names no subject; the assistant picks a gentle everyday subject (rain, tea, the sea, a cat, autumn) and writes it at once, with nothing about the user in it"),
        (2, "", "wants a reply drafted to a landlord, colleague or friend in a given tone"),
        (2, "", "pastes meeting notes and wants action items"),
        (2, "", "wants a draft saved to a file: the text is written with write_file, then the path is confirmed"),
        (2, "", "wants a document in the workspace read and summarised: read_file, then the summary"),
        (3, "", "gives exact constraints (word count, number of bullets, a phrase that must appear); the answer honours each one"),
        (4, "hard", "wants a long, detailed article, essay or report of 800 words or more about {topic}"),
        (2, "hard", "wants a full technical document (a specification or a proposal) with sections"),
    ],
}
STEP_HINTS = {"chat": "", "write": "", "web": "one to three tool steps", "code": "two to four tool steps",
              "terminal": "one to three tool steps", "computer": "one to three tool steps",
              "tools": "one or two tool steps"}
DELETE_RULE = (" The assistant must not delete anything and must not run a deleting command: it may look first "
               "(list, du, git status), then it gives the exact command in a code block for the user to run "
               "themselves, with one line on what it removes.")


def local_time(rng: random.Random) -> str:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    stamp = time.gmtime(1767225600 + rng.randrange(0, 730 * 86400))  # 2026-2027
    return (f"{days[stamp.tm_wday]}, {stamp.tm_mday} {months[stamp.tm_mon - 1]} {stamp.tm_year}, "
            f"{stamp.tm_hour:02d}:{rng.randrange(60):02d} {rng.choice(['+03', '+00', '+01', '+02', '-05', '-08', '+09'])}")


def exact_line(rng: random.Random) -> str:
    kind = rng.randrange(5)
    if kind == 0:
        share, whole = rng.choice([5, 8, 12, 15, 17, 18, 20, 22.5, 30]), rng.randrange(40, 9000)
        return f"{share}% of {whole} = {round(share * whole / 100, 4):g}"
    if kind == 1:
        a, b = rng.randrange(12, 900), rng.randrange(3, 99)
        return f"{a} * {b} = {a * b}"
    if kind == 2:
        a, b = rng.randrange(100, 99999), rng.randrange(2, 60)
        return f"{a} / {b} = {round(a / b, 6):g}"
    if kind == 3:
        km = rng.randrange(2, 400)
        return f"{km} km in miles = {round(km * 0.621371192237, 6):g} mi"
    a, b, c = rng.randrange(10, 5000), rng.randrange(10, 5000), rng.randrange(10, 5000)
    return f"{a} + {b} + {c} = {a + b + c}"


def fill(text: str, rng: random.Random) -> str:
    return text.format(topic=rng.choice(TOPICS), life=rng.choice(LIFE), webthing=rng.choice(WEBTHINGS),
                       field=rng.choice(FIELDS), pair=rng.choice(PAIRS), tech=rng.choice(TECH),
                       project=rng.choice(PROJECTS), newthing=rng.choice(NEW_THINGS), scriptjob=rng.choice(SCRIPT_JOBS),
                       lang=rng.choice(LANGS), sysq=rng.choice(SYSTEM_QUESTIONS), shelljob=rng.choice(SHELL_JOBS),
                       email=rng.choice(EMAILS), deletejob=rng.choice(DELETE_JOBS))


# What a user sends after the first answer. (weight, needs a tool, text)
FOLLOW_MOVES = [
    (4, True, "sends a short follow-up that only makes sense after the answer before it (\"and the biggest one?\", \"same for last week\", \"what about the tests folder?\"); it needs the same kind of tool again with other arguments"),
    (3, True, "asks to go one step further with what was just found or made (open it, save it, fix it, run it, look inside it), which needs another tool call"),
    (2, True, "changes one detail of the request (another folder, file, place, number or format); the assistant does the work again with that change"),
    (2, False, "asks something the results already shown can answer; the assistant answers from them without a tool"),
    (2, False, "sends a short follow-up that only makes sense after the answer before it (\"and in Python?\", \"make it shorter\", \"why?\", \"give me another\")"),
    (1, False, "reacts in a word or two (thanks, ok, nice); the assistant replies with one short friendly sentence"),
]
# How a first reply goes wrong, and how the user answers that. A small model does all of these.
FLAWS = [
    (4, "said it cannot do that or has no access to the user's computer, files or the web, although one of its tools does exactly this"),
    (3, "asked whether it should go ahead (\"Want me to check?\") instead of doing it"),
    (2, "answered about a different thing than was asked (it misread a word, or took a question about this computer or these files for a general or web question)"),
    (2, "gave a vague general explanation instead of looking at the real files, system or pages"),
    (2, "claimed it had checked or done something without any tool call, with made-up figures"),
    (1, "named the next step but did not take it"),
]
OBJECTIONS = [  # (weight, manner, text)
    (3, "reject", "says in a few words that this is wrong or not what they asked (\"no\", \"that's not it\", \"wrong\", \"try again\")"),
    (3, "clarify", "says in one line what they really meant (\"no I mean on my computer\", \"I meant the folder, not the file\")"),
    (2, "order", "tells the assistant to use its tools or to just do it (\"use your tools\", \"you can check that yourself\", \"yes, do it\")"),
    (2, "nudge", "sends a bare nudge (\"so?\", \"and?\", \"well?\", \"??\", \"go on\")"),
    (1, "repeat", "repeats the request in other words, a little impatient"),
]
NO_TOOL_MARKS = ("no tool", "without a tool", "without running", "tools cannot")


def plain_task(skill: str, rng: random.Random, tool: bool = False) -> str:
    """One everyday task of a skill, filled in; with tool=True one that cannot be done without a tool."""
    tasks = [task for task in TASKS[skill]
             if task[1] == "" and not (tool and any(mark in task[2] for mark in NO_TOOL_MARKS))]
    return fill(rng.choices(tasks, weights=[task[0] for task in tasks])[0][2], rng)


def base_seed(family: str, skill: str, rng: random.Random) -> dict:
    return {"family": family, "skill": skill, "kind": family, "hard": False, "time": local_time(rng), "exact": [],
            "voice": rng.choice(VOICES), "followup": False, "folder": rng.choice(FOLDERS) if rng.random() < 0.85 else ""}


def make_session_seed(skill: str, rng: random.Random, other: str | None = None) -> dict:
    """A conversation of two to four turns: a task, then what users really send next. One turn in three
    sessions moves to another skill's work, so the stack learns to change the layer in the middle of a chat."""
    seed = base_seed("session", skill, rng)
    if other is None:
        other = rng.choice([name for name in SKILLS if name != skill]) if rng.random() < 0.35 else ""
    count = rng.choice([2, 2, 3, 3, 4])
    moves = [move for move in FOLLOW_MOVES if SKILL_TOOLS[skill] or not move[1]]
    said, plan = [], []
    for _ in range(count - 1):
        move = rng.choices(moves, weights=[one[0] for one in moves])[0]
        said.append(move[2])
        plan.append("chat" if move[2].startswith("reacts in a word") else "same")
    if other:
        at = rng.randrange(count - 1)
        said[at], plan[at] = f"turns to something else and {plain_task(other, rng)}", "other"
    seed["other"], seed["plan"] = other, plan
    later = " ".join(f"Turn {number}: the user {move}." for number, move in enumerate(said, 2))
    seed["scenario"] = (f"{count} turns. Turn 1: the user {plain_task(skill, rng)}. {later} Voice: {seed['voice']}. "
                        f"Context: local time {seed['time']}.")
    return seed


def make_repair_seed(skill: str, rng: random.Random) -> dict:
    """A first reply that failed the way a small model fails, the user's objection, and the reply that
    should have come: the assistant acts. The bad reply is shown to the model and never trained on."""
    seed = base_seed("repair", skill, rng)
    flaw = rng.choices(FLAWS, weights=[one[0] for one in FLAWS])[0][1]
    _, manner, how = rng.choices(OBJECTIONS, weights=[one[0] for one in OBJECTIONS])[0]
    seed["manner"] = manner
    seed["scenario"] = (f"The user {plain_task(skill, rng, tool=True)}. The assistant's first reply was a bad one: it {flaw}. "
                        f"The user then {how}. Voice: {seed['voice']}. Context: local time {seed['time']}.")
    return seed


def make_real_seed(ask: str, rng: random.Random) -> dict:
    seed = base_seed("real", "chat", rng)
    seed["ask"] = ask
    seed["scenario"] = (f"The user's message is exactly: {json.dumps(ask, ensure_ascii=False)} "
                        f"Context: local time {seed['time']}.")
    return seed


def make_seed(skill: str, rng: random.Random) -> dict:
    weights = [task[0] for task in TASKS[skill]]
    _, kind, text = rng.choices(TASKS[skill], weights=weights)[0]
    text = fill(text, rng)
    seed = {"family": "single", "skill": skill, "kind": kind, "hard": kind == "hard", "time": local_time(rng), "exact": [],
            "voice": rng.choice(VOICES), "followup": kind in ("", "facts") and rng.random() < 0.28,
            "folder": rng.choice(FOLDERS) if rng.random() < 0.85 else ""}
    if kind == "exact":
        seed["exact"] = [exact_line(rng)]
    scenario = f"The user {text}."
    if kind == "delete":
        scenario += DELETE_RULE
    if kind == "hard":
        scenario += " This is a heavy request: the answer is thorough and well organised (up to about 350 words)."
    if STEP_HINTS[skill] and kind not in ("pre", "delete", "offline", "offnote") and "no tool" not in text and "without a tool" not in text:
        scenario += f" Usually {STEP_HINTS[skill]}."
    context = [f"local time {seed['time']}"]
    if seed["exact"]:
        context.append(f"exact result already worked out: {seed['exact'][0]}")
    if kind == "facts":
        context.append("saved facts matter: give two or three in \"facts\"")
    if kind == "offnote":
        context.append("the internet is not reachable and Kiwibro's context says so")
    if kind == "long":
        context.append("long conversation: give \"earlier\"")
    context.append("with a follow-up" if seed["followup"] else "no follow-up")
    seed["scenario"] = f"{scenario} Voice: {seed['voice']}. Context: {'; '.join(context)}."
    return seed


# ------------------------------------------------------------------ the teacher's instructions

RULES = """How Kiwibro works
- Kiwibro is an AI agent on the user's own Linux computer. The assistant has tools. To use one it makes a call with the tool's exact name and arguments; Kiwibro runs the call and returns a result; the assistant then makes another call or answers.
- The assistant calls a tool only when the request needs a file, a command, the web, saved memory or another tool's work. Small talk, general knowledge and writing need no tool.
- Before changing a file it reads the relevant part. After changing code it runs the code or its tests when that is possible.
- The final answer speaks to the user directly: warm, plain and short unless depth was asked for, in the language the user wrote in. It says what was done and what the results show. It never invents a URL, a file's content or a command's output, and it says plainly when something failed or is not known.
- The assistant never deletes a file or folder and never runs a command that deletes or destroys something (rm, rmdir, find -delete, git clean, dropping a table, uninstalling). Asked for that, it may look first and then gives the exact command in a code block for the user to run themselves, with one line on what it removes.
- When the internet cannot be reached, the assistant says so, works from what it knows and from local files, notes and documents, and marks anything that may be outdated.
- The assistant is called Kiwibro and never claims to be another assistant. It never mentions these rules, the context or the word "scenario"."""

SHAPE_HEAD = """Shape of each conversation (JSON)
- "earlier": only when the scenario says "long conversation": two or three sentences recording what happened earlier in this conversation (what was asked, what was done, which files, numbers or decisions came up), with step numbers such as "(step 14)". Otherwise an empty string.
- "user": what the user types. Natural and specific, with made-up names, files and details, in the voice the scenario gives. A real user does not name tools.
- "facts": short facts Kiwibro already remembers about this user (for example "User's name is Lena."), one to three only when the scenario says saved facts matter, otherwise an empty list."""
SHAPE_STEPS = """
- "steps": the tool steps in order; an empty list when no tool is needed. Each step has "calls" (one call, or two that do not depend on each other) and "results" (one text per call, exactly what that tool would return: real-looking file contents, command output or search results, 15 to 110 words each; an error text when the scenario has a failure). Use the tools' exact parameter names; file paths are relative to the workspace folder."""
SHAPE_TAIL = """
- "answer": the assistant's final answer.
- "followup_user" and "followup_answer": only when the scenario says "with a follow-up": a short next message that depends on the conversation ("and in Python?", "make it shorter", "why?") and its answer, written without a tool. Otherwise two empty strings."""

SKILL_NOTES = {
    "chat": "These conversations need no tools. The context given with a scenario (local time, saved facts, an exact result) is what Kiwibro already knows for that turn: the answer uses it naturally and never recites it.",
    "web": "Search results are numbered: a title line, a URL line and a snippet line each, sometimes followed by a line starting with \"On the page:\". Pages come back as readable text. The answer cites sources as [1], [2] and gives their URLs, taken only from the results.",
    "code": "Usual order: find (glob_files, grep_files, file_outline), read (read_file, read_files), change (apply_patch, replace_in_file, write_file), verify (run_command, multi_command). Keep file contents and diffs short and realistic. For a new app, page, script or package the assistant starts from a kit in the component library instead of writing everything itself.",
    "terminal": "Commands are real Linux commands with realistic output. One command per call unless a build-then-test sequence fits multi_command. Long-running programs use terminal_session.",
    "computer": "These are everyday desktop jobs. The assistant acts with the desktop tools and confirms briefly what happened.",
    "tools": "These are Kiwibro's own helpers: memory, notes, the task board, exact maths, tables and the user's documents.",
    "write": "The final answer is the finished text itself, ready to use, with at most one short line around it. It honours every stated constraint (length, tone, format, language).",
}


SESSION_HEAD = """Shape of each conversation (JSON)
- "turns": the turns of one conversation in order, as many as the scenario says. Each turn has:
  - "user": what the user types in that turn. Natural, with made-up names, files and details, in the voice the scenario gives. Later turns are short and lean on what came before, the way people really write ("and the biggest one?", "same for last week", "ok now fix it"). A real user does not name tools."""
SESSION_STEPS = """
  - "steps": the tool steps of that turn in order; an empty list when the turn needs no tool. Each step has "calls" (one call, or two that do not depend on each other) and "results" (one text per call, exactly what that tool would return: real-looking file contents, command output or search results, 15 to 110 words each). Use the tools' exact parameter names; file paths are relative to the workspace folder."""
SESSION_TAIL = """
  - "answer": the assistant's final answer for that turn. A follow-up gets only what is new: no answer repeats an earlier one.
- When a later turn asks for something a tool can find out, the assistant calls the tool again in that turn, even when an earlier turn used the same tool. It never answers from memory what has to be looked up, never says it cannot do what one of its tools does, and never asks whether it should go ahead with a harmless look."""
REPAIR_SHAPE = """Shape of each conversation (JSON)
- "user": what the user types first. Natural and specific, with made-up names, files and details, in the voice the scenario gives. A real user does not name tools.
- "bad_answer": the assistant's first reply, bad in exactly the way the scenario says, one to three sentences.
- "objection": the user's next message, short, in the manner the scenario says.
- "steps": what the assistant does after the objection: it acts at once, with the right tools, on what the user asked for in the first place. At least one step. Each step has "calls" (one call, or two that do not depend on each other) and "results" (one text per call, exactly what that tool would return, 15 to 110 words each). Use the tools' exact parameter names; file paths are relative to the workspace folder.
- "answer": the assistant's final answer, from the results. It repeats no part of the bad reply, asks for no permission, and apologises in three words at most or not at all."""
REAL_SHAPE = """Shape of each conversation (JSON)
- "user": the user's message, copied exactly as the scenario gives it. These are messages real people wrote to an assistant.
- "work": "web" when a good answer needs current or checkable facts (news, prices, versions, schedules, weather, a named site), "write" when the user wants a text written or reworked, otherwise "chat".
- "steps": for "web" one to three tool steps, each with "calls" (one call, or two that do not depend on each other) and "results" (one text per call, exactly what that tool would return, 15 to 110 words each); for "chat" and "write" an empty list.
- "answer": the assistant's final answer, as short as the message deserves. For something Kiwibro has no tool for (alarms, music, phone calls, smart-home devices, bank accounts) it says so in one sentence and offers the nearest thing it can do. A harmful request is declined in one sentence."""


def shape_text(family: str, names: list[str]) -> str:
    if family == "session":
        return SESSION_HEAD + (SESSION_STEPS if names else "") + SESSION_TAIL
    if family == "repair":
        return REPAIR_SHAPE
    if family == "real":
        return REAL_SHAPE
    return SHAPE_HEAD + (SHAPE_STEPS if names else "") + SHAPE_TAIL


def manual(tool: dict, examples: dict) -> str:
    function = tool["function"]
    schema = function.get("parameters") or {}
    required = set(schema.get("required") or [])
    lines = [f"- {RENDER.signature(tool)}: {' '.join(str(function.get('description') or '').split())}"]
    for name, spec in (schema.get("properties") or {}).items():
        about = " ".join(str(spec.get("description") or "").split())
        lines.append(f"    {name} ({spec.get('type') or 'value'}{'' if name in required else ', optional'})"
                     + (f": {about}" if about else ""))
    for arguments, result in examples.get(function["name"], [])[:2]:
        shown = result[:420].replace("\n", "\n      ")
        lines.append(f"    real result for {json.dumps(arguments, ensure_ascii=False)[:160]}:\n      {shown}")
    return "\n".join(lines)


def teacher_prefix(skill: str, catalog: dict, examples: dict, family: str = "single", other: str = "") -> str:
    """Everything that is the same for a skill, first, so the teacher's server reuses it."""
    names = family_tools(skill, other)
    parts = [RULES, SKILL_NOTES[skill]]
    if other:
        parts.append(SKILL_NOTES[other])
    if names:
        parts.append("Tools for these conversations\n" + "\n".join(manual(catalog[name], examples) for name in names))
        if "component" in names:
            parts.append(COMPONENT_NOTES)
    parts.append(shape_text(family, names))
    return "\n\n".join(parts)


TEACHER_SYSTEM = ("You write training conversations for Kiwibro, an AI agent that runs on a user's own computer. "
                  "Each conversation shows the assistant behaving ideally. You reply with JSON only.")


def teacher_prompt(prefix: str, seeds: list[dict]) -> str:
    lines = [f"{index}. {seed['scenario']}" for index, seed in enumerate(seeds, 1)]
    return (f"{prefix}\n\nWrite {len(seeds)} conversations, one for each scenario below, in the same order. "
            f"Make them differ in subject, length and wording.\n\nScenarios\n" + "\n".join(lines))


# ------------------------------------------------------------------ from the teacher's JSON to a training record

NOTE_HEAD = "[Kiwibro context for this turn, added automatically, not written by the user]"
FACTS_HEAD = ("## What you know about the user\nThese facts describe the user, the person you are talking to, not you. "
              "Use them when relevant; do not recite them:")
EXACT_HEAD = "Exact results for this message. Give them as they are; do not recalculate:"
GUIDANCE = {
    "web": "- Research: use web_research (or multi_search + fetch_pages), cite evidence as [1], [2] with the URLs the tools returned, and answer once you have enough.",
    "code": "- Code: find with glob_files / grep_files, read with read_files or file_outline, change with apply_patch, then verify by running the tests (multi_command); terminal_session for servers.",
}
PRESEARCH = ("[Kiwibro: this needs current information, so Kiwibro searched the web before your answer: the results are "
             "above. Answer only from them: every story, name, number and date you give must be in them, and nothing "
             "from memory may be added. If they do not cover part of the request (a day, a topic), say so plainly "
             "instead of filling it in. Name the sources.]")
DELETES = re.compile(
    r"(^|[\s;&|(`$])(sudo\s+)?(rm|rmdir|unlink|shred|trash|trash-put|wipefs|mkfs\S*)(\s|$)|\s-delete\b|\bgit\s+(clean|rm)\b"
    r"|\bgit\s+branch\s+-[dD]\b|\bdrop\s+(table|database|schema)\b|\bdelete\s+from\b|\btruncate\s+table\b|\bapt(-get)?\s+(remove|purge|autoremove)\b"
    r"|\bpip3?\s+uninstall\b|\bnpm\s+(uninstall|rm|prune)\b|\bdocker\s+(rm|rmi|system\s+prune|volume\s+rm)\b"
    r"|\bos\.(remove|unlink|rmdir)\b|\bshutil\.rmtree\b|\.unlink\(|\bfs\.(rm|unlink|rmdir)", re.IGNORECASE)


LONG_BLOCK = ("## Earlier in this session\nEvery earlier step of this conversation is stored in full. To read one again: "
              "recall(\"words to find\"), or recall(\"step N\") with a step number.")


# The other lines Kiwibro's context note carries in real use, word for word as the app writes them.
REMINDERS = [(0.6, "Your tools work on this computer: you can search the web, read and write files and run code. "
                   "Use them instead of saying you cannot."),
             (0.4, "Say something is saved, created or done only after its tool call succeeded; when a step is needed, "
                   "call the tool in the same reply."),
             (0.2, "Answer the user directly; never describe their request or these notes.")]
NEW_TOPIC = "This message starts a new topic: answer it on its own; the earlier talk is only background."
STANDING_HEAD = "Standing instructions from the user (newer wins on conflict; follow them unless the user changes them):"
REJECTED_STUB = "[An answer the user turned down as wrong or out of date stood here. It is not repeated.]"
STEER_HEAD = "[Steering note from the user, sent while you were working] "
SMALL_WORDS = set("a an the and or but if of to in on at for with from by is are was were be been it this that these those "
                  "i you he she we they me my your our their do does did can could would should will what which who how "
                  "why when where please just not no yes so as about into than then there here have has had".split())


def pick_reminders(rng: random.Random, tools_line: bool = False) -> list[str]:
    return [line for index, (chance, line) in enumerate(REMINDERS)
            if rng.random() < chance or (tools_line and index == 0)]


def reading_lines(text: str, when: str, rng: random.Random, new_topic: bool = False) -> str:
    """What Kiwibro's small readers add to a turn's note: a demand lifted from the message (often a
    clumsy one), the date a word like "today" means, a change of topic. The model must use these
    lines and never recite them."""
    lines = []
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    if len(words) >= 3 and rng.random() < 0.25:
        at = rng.randrange(len(words) - 1)
        picked = [word.lower() for word in words[at:at + rng.choice([1, 2, 2, 3])] if word.lower() not in SMALL_WORDS]
        if picked:
            lines.append(f"The answer must honour: {' '.join(picked)}.")
    if re.search(r"\btoday\b", text, re.IGNORECASE) and when.count(", ") >= 2:
        lines.append(f"Dates the message means: today = {when.split(', ')[1]}.")
    if new_topic:
        lines.append(NEW_TOPIC)
    return "\n".join(lines)


def standing_note(goal: str, turn: int) -> str:
    goal = " ".join(goal.split())
    return f"{STANDING_HEAD}\n- Goal (turn {turn}): {goal if len(goal) <= 140 else goal[:139].rstrip() + '…'}"


def reaction_note(request: str, correcting: bool) -> str:
    request = request.strip()[:400]
    if correcting:
        return (f"The user's new message may be correcting your last answer (they had asked: \"{request}\"). "
                "Answer the new message as it is written. Do not repeat or defend the last answer.")
    return ("The user says the last answer was wrong or not what they asked for. Do not repeat or defend it, and do not "
            f"apologise at length. Their request stands: \"{request}\". Answer that request again, correctly this time.")


def with_note(text: str, when: str, facts=(), exact=(), guidance: str = "", offline: bool = False,
              long: bool = False, standing: str = "", reading: str = "", reaction: str = "", reminders=()) -> str:
    """A user message as the model gets it: the text, then Kiwibro's note in the order the app builds it."""
    parts = [text.strip(), f"{NOTE_HEAD}\nLocal time: {when}" + (f"\n{OFFLINE_LINE}" if offline else "")]
    if long:
        guidance = ""
    if exact:
        parts.append(EXACT_HEAD + "\n" + "\n".join(f"- {line}" for line in exact))
    if facts:
        parts.append(FACTS_HEAD + "\n" + "\n".join(f"- {fact.strip()}" for fact in facts))
    if long:
        parts.append(LONG_BLOCK)
    if standing:
        parts.append(standing)
    if reading:
        parts.append(reading)
    if guidance:
        parts.append("## Guidance for this turn\n" + guidance)
    if reaction:
        parts.append(reaction)
    if reminders:
        parts.append("\n".join(f"- {line}" for line in reminders))
    return "\n\n".join(parts)


def call_commands(call: dict) -> str:
    arguments = call.get("arguments") or {}
    found = [str(arguments.get(key) or "") for key in ("command", "input", "code")]
    found += [str(one) for one in arguments.get("commands") or [] if isinstance(arguments.get("commands"), list)]
    return "\n".join(found)


def leaks(text: str) -> bool:
    return "<tool_call>" in text or "scenario" in text.lower() or NOTE_HEAD in text


def opening(seed: dict, earlier: str = "") -> list[dict]:
    system = []
    if seed["folder"]:
        system.append(f"Your file and command tools work inside the folder `{seed['folder']}`.")
    if earlier:
        system.append("## Earlier conversation summary\n" + earlier)
    return [{"role": "system", "content": "\n\n".join(system)}] if system else []


def check_steps(steps, catalog: dict, offnote: bool = False, offline: bool = False) -> tuple[list[dict] | None, list[str], str]:
    """The messages of one turn's tool steps and the tools they call, or None and why they are refused."""
    messages, used = [], []
    for step in steps if isinstance(steps, list) else [None]:
        if not isinstance(step, dict):
            return None, used, "shape"
        calls, results = step.get("calls") or [], step.get("results") or []
        if not calls or len(calls) != len(results):
            return None, used, "calls and results differ"
        for call in calls:
            name, arguments = call.get("name"), call.get("arguments")
            if name not in catalog or not isinstance(arguments, dict):
                return None, used, "unknown tool"
            schema = catalog[name]["function"].get("parameters") or {}
            if any(key not in arguments for key in schema.get("required") or []):
                return None, used, "missing argument"
            if any(key not in (schema.get("properties") or {}) for key in arguments):
                return None, used, "unknown argument"
            if (name == "file_manage" and arguments.get("action") == "delete") or DELETES.search(call_commands(call)):
                return None, used, "deletes"
            if name == "component" and arguments.get("name") and arguments["name"] not in {c[0] for c in COMPONENTS}:
                return None, used, "unknown component"
            if offnote and name in WEB_TOOLS:
                return None, used, "web call while known to be offline"
            used.append(name)
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for call, result in zip(calls, results):
            if not isinstance(result, str) or not result.strip() or len(result) > 3000:
                return None, used, "result"
            if offline and call["name"] in WEB_TOOLS:
                result = OFFLINE_ERROR  # word for word what Kiwibro's web tools return without a connection
            messages.append({"role": "tool", "content": result.strip()})
    return messages, used, ""


def forget_notes(messages: list[dict], turns: list[dict], skip: set[int]) -> None:
    """The conversation as Kiwibro sends it when only the newest message carries its context note.
    The earlier replies were written while their own note was still there, so they are shown and not trained."""
    last = turns[-1]["at"]
    for index, message in enumerate(messages[:last]):
        if message["role"] == "user" and NOTE_HEAD in message["content"]:
            message["content"] = message["content"].split("\n\n" + NOTE_HEAD)[0]
        elif message["role"] == "assistant":
            skip.add(index)


def finish(messages: list[dict], seed: dict, used: list[str], catalog: dict, rng: random.Random, turns: list[dict],
           skip: set[int], followup: bool = False) -> dict:
    """turns: where each user turn starts and which skill answers it. skip: assistant messages that are
    shown but not trained on (a bad first reply, a reply whose note has since been dropped)."""
    if len(turns) > 1 and rng.random() < 0.5:
        forget_notes(messages, turns, skip)
    everything = list(catalog)
    if rng.random() < 0.6:
        tools = everything
    else:
        extra = rng.sample(everything, rng.randrange(8, 26))
        tools = [name for name in everything if name in set(extra) | set(used) | set(SKILL_TOOLS[seed["skill"]][:4])]
    identity = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": identity, "skill": seed["skill"], "kind": seed["kind"], "hard": seed["hard"], "followup": followup,
            "tools": tools, "messages": messages, "turns": turns, "skip": sorted(skip),
            "split": "held" if int(identity[:4], 16) % 16 == 0 else "train"}


def build_record(conversation: dict, seed: dict, catalog: dict, rng: random.Random) -> tuple[dict | None, str]:
    """A checked training record of one request, or None and the reason it was refused."""
    user, answer = str(conversation.get("user") or "").strip(), str(conversation.get("answer") or "").strip()
    if not 3 <= len(user) <= 2500 or not 2 <= len(answer) <= 5000:
        return None, "length"
    if leaks(answer):
        return None, "leak"
    facts = [fact for fact in conversation.get("facts") or [] if isinstance(fact, str) and fact.strip()][:3]
    if seed["kind"] != "facts":
        facts = []
    guidance = GUIDANCE.get(seed["skill"], "") if rng.random() < 0.4 else ""
    earlier = str(conversation.get("earlier") or "").strip() if seed["kind"] == "long" else ""
    if seed["kind"] == "long" and not 30 <= len(earlier) <= 900:
        return None, "no earlier record"
    reminders = pick_reminders(rng)
    messages = opening(seed, earlier)
    offnote = seed["kind"] == "offnote"
    turns = [{"at": len(messages), "skill": seed["skill"], "hard": seed["hard"]}]
    messages.append({"role": "user", "content": with_note(user, seed["time"], facts, seed["exact"],
                                                          "" if offnote else guidance, offnote, bool(earlier),
                                                          reading=reading_lines(user, seed["time"], rng),
                                                          reminders=reminders)})
    steps = conversation.get("steps") or []
    if seed["kind"] == "pre" and (len(steps) != 1 or not isinstance(steps[0], dict) or not steps[0].get("calls")
                                  or steps[0]["calls"][0].get("name") != "web_search"):
        return None, "presearch shape"
    step_messages, used, why = check_steps(steps, catalog, offnote, seed["kind"] == "offline")
    if step_messages is None:
        return None, why
    messages += step_messages
    if seed["kind"] == "offline" and sum(name in WEB_TOOLS for name in used) != 1:
        return None, "offline shape"
    if seed["kind"] == "pre":
        messages.append({"role": "user", "content": PRESEARCH})
    if seed["kind"] == "delete" and not (DELETES.search(answer) or "```" in answer):
        return None, "no command handed over"
    messages.append({"role": "assistant", "content": answer})
    follow_user = str(conversation.get("followup_user") or "").strip()
    follow_answer = str(conversation.get("followup_answer") or "").strip()
    followup = bool(seed["followup"] and follow_user and follow_answer)
    if followup:
        turns.append({"at": len(messages), "skill": seed["skill"], "hard": False})
        messages.append({"role": "user", "content": with_note(follow_user, seed["time"],
                                                              reading=reading_lines(follow_user, seed["time"], rng),
                                                              reminders=reminders)})
        messages.append({"role": "assistant", "content": follow_answer})
    return finish(messages, seed, used, catalog, rng, turns, set(), followup), ""


def turn_skill(seed: dict, used: list[str], plan: str) -> str | None:
    """Which skill a turn of a session belongs to: the one whose tools it calls, else the one the scenario planned."""
    first, second = (seed["other"], seed["skill"]) if plan == "other" else (seed["skill"], seed["other"])
    if used:
        return next((skill for skill in (first, second) if skill and all(name in SKILL_TOOLS[skill] for name in used)), None)
    return "chat" if plan == "chat" else (first or second)


def build_session(conversation: dict, seed: dict, catalog: dict, rng: random.Random) -> tuple[dict | None, str]:
    given = conversation.get("turns")
    if not isinstance(given, list) or len(given) < 2:
        return None, "turns"
    reminders = pick_reminders(rng)
    messages, turns, used_all, goal = opening(seed), [], [], ""
    for number, turn in enumerate(given[:1 + len(seed["plan"])]):
        if not isinstance(turn, dict):
            return None, "shape"
        user, answer = str(turn.get("user") or "").strip(), str(turn.get("answer") or "").strip()
        if not 2 <= len(user) <= 2500 or not 2 <= len(answer) <= 5000:
            return None, "length"
        if leaks(answer):
            return None, "leak"
        step_messages, used, why = check_steps(turn.get("steps") or [], catalog)
        if step_messages is None:
            return None, why
        plan = seed["plan"][number - 1] if number else "same"
        skill = turn_skill(seed, used, plan)
        if skill is None:
            return None, "tools of two skills in one turn"
        standing = standing_note(goal, 1) if number and rng.random() < 0.3 else ""
        reading = reading_lines(user, seed["time"], rng, new_topic=plan == "other" and rng.random() < 0.7)
        turns.append({"at": len(messages), "skill": skill, "hard": False})
        messages.append({"role": "user", "content": with_note(user, seed["time"], standing=standing, reading=reading,
                                                              reminders=reminders)})
        messages += step_messages
        messages.append({"role": "assistant", "content": answer})
        goal = goal or user
        used_all += used
    answers = [message["content"] for message in messages if message["role"] == "assistant" and message["content"]]
    if len(set(answers)) < len(answers):  # a copied answer is the habit these conversations are there to break
        return None, "repeated answer"
    return finish(messages, seed, used_all, catalog, rng, turns, set()), ""


def build_repair(conversation: dict, seed: dict, catalog: dict, rng: random.Random) -> tuple[dict | None, str]:
    user, answer = str(conversation.get("user") or "").strip(), str(conversation.get("answer") or "").strip()
    bad, objection = str(conversation.get("bad_answer") or "").strip(), str(conversation.get("objection") or "").strip()
    if not 3 <= len(user) <= 2500 or not 2 <= len(answer) <= 5000 or not 5 <= len(bad) <= 700 or not 1 <= len(objection) <= 400:
        return None, "length"
    if leaks(answer) or leaks(bad) or answer == bad:
        return None, "leak"
    step_messages, used, why = check_steps(conversation.get("steps") or [], catalog)
    if step_messages is None:
        return None, why
    if not used or any(name not in SKILL_TOOLS[seed["skill"]] for name in used):
        return None, "no fitting tool after the objection"
    manner = seed["manner"]
    rejected = manner == "reject"
    # after a reply that denied its tools, Kiwibro's note carries the reminder from then on
    reminders = pick_reminders(rng, tools_line=rng.random() < 0.9)
    messages = opening(seed)
    turns = [{"at": len(messages), "skill": seed["skill"], "hard": False}]
    messages.append({"role": "user", "content": with_note(user, seed["time"], reading=reading_lines(user, seed["time"], rng),
                                                          reminders=reminders)})
    bad_at = len(messages)
    messages.append({"role": "assistant", "content": REJECTED_STUB if rejected and rng.random() < 0.5 else bad})
    reaction = ""
    if rejected and rng.random() < 0.85:
        reaction = reaction_note(user, False)
    elif manner in ("clarify", "repeat") and rng.random() < 0.5:
        reaction = reaction_note(user, True)
    standing = standing_note(user, 1) if not reaction and rng.random() < 0.5 else ""
    said = STEER_HEAD + objection if rng.random() < 0.12 else objection
    turns.append({"at": len(messages), "skill": seed["skill"], "hard": False})
    messages.append({"role": "user", "content": with_note(said, seed["time"], standing=standing, reaction=reaction,
                                                          reminders=reminders)})
    messages += step_messages
    messages.append({"role": "assistant", "content": answer})
    return finish(messages, seed, used, catalog, rng, turns, {bad_at}), ""


def build_real(conversation: dict, seed: dict, catalog: dict, rng: random.Random) -> tuple[dict | None, str]:
    user, answer, work = seed["ask"], str(conversation.get("answer") or "").strip(), conversation.get("work")
    if work not in ("chat", "write", "web") or not 2 <= len(answer) <= 5000:
        return None, "length"
    if leaks(answer):
        return None, "leak"
    step_messages, used, why = check_steps(conversation.get("steps") or [], catalog)
    if step_messages is None:
        return None, why
    if (work == "web") != bool(used) or any(name not in SKILL_TOOLS["web"] for name in used):
        return None, "work and steps differ"
    seed = {**seed, "skill": work}
    messages = opening(seed)
    turns = [{"at": len(messages), "skill": work, "hard": False}]
    messages.append({"role": "user", "content": with_note(user, seed["time"], reading=reading_lines(user, seed["time"], rng),
                                                          reminders=pick_reminders(rng))})
    messages += step_messages
    messages.append({"role": "assistant", "content": answer})
    return finish(messages, seed, used, catalog, rng, turns, set()), ""


BUILDERS = {"single": build_record, "session": build_session, "repair": build_repair, "real": build_real}


# ------------------------------------------------------------------ stage: gen

def read_conversations(text: str) -> list:
    """The conversations in a teacher reply; from a reply that was cut off, the complete ones."""
    try:
        found = json.loads(text[text.index("{"):text.rindex("}") + 1])["conversations"]
        return found if isinstance(found, list) else []
    except (ValueError, KeyError, TypeError):
        pass
    at = text.find("[", text.find('"conversations"'))
    if at < 0:
        return []
    decoder, found, at = json.JSONDecoder(), [], at + 1
    while True:
        while at < len(text) and text[at] in " \n\r\t,":
            at += 1
        if at >= len(text) or text[at] != "{":
            return found
        try:
            item, at = decoder.raw_decode(text, at)
        except ValueError:
            return found
        found.append(item)


ASSISTANT_ONLY = {"alarm", "audio", "iot", "play", "music", "email", "social", "takeaway", "calendar", "lists", "transport"}


def load_asks(path: Path) -> list[str]:
    """First messages real people wrote to assistants, from open data sets. They bring the wording, the
    typos and the range of wishes that invented scenarios lack; the teacher writes Kiwibro's answer."""
    if path.exists():
        return json.loads(path.read_text())
    from datasets import load_dataset

    asks: dict[str, None] = {}

    def keep(text, most: int = 320) -> None:
        text = " ".join(str(text or "").split())
        if 6 <= len(text) <= most:
            asks[text] = None

    def oasst():
        for row in load_dataset("OpenAssistant/oasst2", split="train"):
            if row["parent_id"] is None and row["role"] == "prompter" and row["lang"] == "en" and not row.get("deleted"):
                keep(row["text"])

    def dolly():
        for row in load_dataset("databricks/databricks-dolly-15k", split="train"):
            if not str(row["context"] or "").strip():
                keep(row["instruction"])

    def massive():  # spoken commands; the ones only a phone or a smart speaker can carry out are left out
        for row in load_dataset("mteb/amazon_massive_intent", "en", split="train"):
            if str(row.get("label_text") or "").split("_")[0] not in ASSISTANT_ONLY:
                keep(row["text"], 200)

    for name, load in (("oasst2", oasst), ("dolly", dolly), ("massive", massive)):
        before = len(asks)
        try:
            load()
        except Exception as error:  # one data set that cannot be fetched must not stop a paid GPU run
            log(f"asks: {name} could not be loaded: {type(error).__name__}: {str(error)[:200]}")
        log(f"asks: {name} gave {len(asks) - before}")
    found = list(asks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, ensure_ascii=False))
    return found


def stage_gen(args) -> None:
    from vllm import LLM, SamplingParams

    catalog = load_catalog(Path(args.tools))
    examples: dict[str, list] = {}
    if args.examples and Path(args.examples).exists():
        for item in json.loads(Path(args.examples).read_text()):
            examples.setdefault(item["tool"], []).append((item["arguments"], item["result"]))
    asks = load_asks(Path(args.asks)) if args.real > 0 else []
    out = Path(args.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        seen = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    log(f"gen: {len(seen)} conversations already there; target {args.target}; {args.minutes} min; {len(asks)} real asks")
    llm = LLM(model=TEACHER, max_model_len=16384, gpu_memory_utilization=args.gpu, max_num_seqs=args.seqs,
              enable_prefix_caching=True, seed=SEED)
    rng = random.Random(SEED + len(seen))
    rng.shuffle(asks)
    next_ask = itertools.cycle(asks)
    mode = args.schema
    prefixes: dict[tuple, str] = {}
    schemas: dict[tuple, dict] = {}

    def prefix(family: str, skill: str, other: str) -> str:
        key = (family, skill, other)
        if key not in prefixes:
            prefixes[key] = teacher_prefix(skill, catalog, examples, family, other)
        return prefixes[key]

    def schema(family: str, skill: str, other: str, count: int) -> dict | None:
        if mode == "none":
            return None
        key = (mode, family, skill, other, count)
        if key not in schemas:
            schemas[key] = (conversation_schema(skill, catalog, count, mode) if family == "single"
                            else family_schema(family, family_tools(skill, other), catalog, count, mode))
        return schemas[key]

    def params(wanted: dict | None, most: int):
        if wanted is None:
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=most)
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=most,
                                  structured_outputs=StructuredOutputsParams(json=wanted))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=most,
                                  guided_decoding=GuidedDecodingParams(json=wanted))

    wanted = [skill for skill in SKILLS if not args.only or skill in args.only.split(",")]
    # conversations with tool steps are long: fewer of them per reply, so a reply ends before its limit
    per_prompt = {skill: args.per_prompt if skill in ("chat", "write") else max(2, args.per_prompt - 1) for skill in SKILLS}
    deadline = time.time() + args.minutes * 60
    kept, chunk, last = len(seen), 0, 0.0
    counts: dict[str, int] = {}
    per_skill = args.first
    while kept < args.target:
        if chunk:  # size each further chunk to the time that is left: a chunk only ends with its longest reply
            left = deadline - time.time()
            per_skill = min(args.chunk, int(per_skill * left / max(last, 1.0) * 0.85))
            if left < 90 or per_skill < 3:
                break
        jobs = []  # (family, skill, other skill, seeds)
        for skill in wanted:
            for _ in range(max(1, round(per_skill * args.single))):
                jobs.append(("single", skill, "", [make_seed(skill, rng) for _ in range(per_prompt[skill])]))
            for _ in range(round(per_skill * args.sessions)):
                first = make_session_seed(skill, rng)  # both conversations of a prompt share one tool manual
                jobs.append(("session", skill, first["other"], [first, make_session_seed(skill, rng, first["other"])]))
            for _ in range(round(per_skill * args.repairs) if SKILL_TOOLS[skill] else 0):
                jobs.append(("repair", skill, "", [make_repair_seed(skill, rng) for _ in range(3)]))
        for _ in range(round(per_skill * args.real) if asks else 0):
            jobs.append(("real", "web", "chat", [make_real_seed(next(next_ask), rng) for _ in range(4)]))
        rng.shuffle(jobs)
        started = time.time()
        while True:
            talks = [[{"role": "system", "content": TEACHER_SYSTEM},
                      {"role": "user", "content": teacher_prompt(prefix(family, skill, other), seeds)}]
                     for family, skill, other, seeds in jobs]
            try:
                outputs = llm.chat(talks, [params(schema(family, skill, other, len(seeds)),
                                                  args.max_tokens + (900 if family == "session" else 0))
                                           for family, skill, other, seeds in jobs],
                                   use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                break
            except Exception as error:  # a schema the grammar compiler refuses: fall back one level
                log(f"gen: schema mode {mode} failed: {type(error).__name__}: {str(error)[:300]}")
                if mode == "none":
                    raise
                mode = "loose" if mode == "strict" else "none"
        last = time.time() - started
        tokens, refused, wrote, shown = 0, {}, 0, 0
        with out.open("a") as handle:
            for (family, skill, _other, seeds), output in zip(jobs, outputs):
                tokens += len(output.outputs[0].token_ids)
                text = output.outputs[0].text
                conversations = read_conversations(text)
                if len(conversations) < len(seeds):
                    refused["cut short"] = refused.get("cut short", 0) + len(seeds) - len(conversations)
                    if shown < 2:
                        shown += 1
                        log(f"gen: {family} {skill} output cut short after {len(conversations)} of {len(seeds)}, "
                            f"{len(output.outputs[0].token_ids)} tokens; it ends: {text[-260:]!r}")
                for conversation, seed in zip(conversations, seeds):
                    record, why = (BUILDERS[family](conversation, seed, catalog, rng) if isinstance(conversation, dict)
                                   else (None, "shape"))
                    if record is None or record["id"] in seen:
                        why = f"{family}: {why or 'repeat'}"
                        refused[why] = refused.get(why, 0) + 1
                        continue
                    seen.add(record["id"])
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for key in (family, record["skill"]):
                        counts[key] = counts.get(key, 0) + 1
                    wrote += 1
        kept += wrote
        log(f"gen: chunk {chunk} mode {mode}: {len(jobs)} prompts, {wrote} kept, refused {refused}, "
            f"{tokens / max(last, 1e-6):.0f} tok/s, {last:.0f}s; total {kept}; counts {counts}")
        if chunk == 0:
            shown_kinds = set()
            for line in out.read_text().splitlines():
                sample = json.loads(line)
                family = sample["kind"] if sample["kind"] in ("session", "repair", "real") else "single"
                if family not in shown_kinds:
                    shown_kinds.add(family)
                    text = RENDER.render(sample["messages"], [], add_generation_prompt=False)
                    log(f"gen: sample {family} (turns {sample['turns']}, not trained {sample['skip']})\n{text[-2600:]}")
        chunk += 1
    log(f"gen: done with {kept} conversations")


# ------------------------------------------------------------------ data for training

def shown_tools(record: dict, catalog: dict) -> list[dict]:
    """The tool manual a conversation is trained with: Kiwibro's whole catalog in its own order, as the
    app sends it whichever skill is active (so the manual never gives the skill away). One conversation
    in five loses a few tools it does not call: a switched-off tool must not confuse the model."""
    names = list(catalog)
    rng = random.Random(record["id"])
    if rng.random() < 0.2:
        called = {call["function"]["name"] for message in record["messages"] for call in message.get("tool_calls") or []}
        spare = [name for name in names if name not in called]
        gone = set(rng.sample(spare, min(len(spare), rng.randint(3, 10))))
        names = [name for name in names if name not in gone]
    return [catalog[name] for name in names]


def encode(tokenizer, record: dict, catalog: dict, full: bool = False) -> dict:
    """One conversation as the trainer needs it. For each token: its id, how it counts as something the
    model must learn to write (an index into WEIGHTS), the turn it belongs to (-1: the system turn) and
    whether a running session computes it only when the next turn arrives. For each assistant message:
    the position before it, its turn, whether it is a tool call and of which tool, and whether it is
    only shown (a reply the user turned down)."""
    import torch

    messages = record["messages"]
    speak = [(index, message) for index, message in enumerate(messages) if message["role"] in ("user", "assistant")]
    opens = {turn["at"]: number for number, turn in enumerate(record.get("turns") or [])}
    if not opens:
        opens = {next(index for index, message in speak if message["role"] == "user"): 0}
    last = max(opens.values())
    skip, names = set(record.get("skip") or []), list(catalog)
    tools = list(catalog.values()) if full else shown_tools(record, catalog)
    text, pieces, at, turn = "", [], 0, -1
    for role, body in RENDER.turns(messages, tools):
        index, opener = None, False
        if role != "system" and at < len(speak):
            number, message = speak[at]
            if message["role"] == role and (role == "assistant" or RENDER.plain(message.get("content")) == body):
                index, at = number, at + 1
                if number in opens:
                    turn, opener = opens[number], True
        begin = len(text)
        text += f"{RENDER.IM_START}{role}\n"
        inner = len(text)
        text += body
        close = len(text)
        text += RENDER.IM_END + "\n"
        pieces.append({"role": role, "inner": inner, "close": close, "end": close + len(RENDER.IM_END), "stop": len(text),
                       "turn": -1 if role == "system" else turn, "message": index, "opener": opener})
    final = {}  # the last assistant message of each turn: what follows it is computed when the next turn arrives
    for number, piece in enumerate(pieces):
        if piece["role"] == "assistant":
            final[piece["turn"]] = number
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    code, tturn, late = [], [], []
    starts, start_turn, start_act, start_tool, start_skip = [], [], [], [], []
    at, counted = 0, 0
    for index, (begin, end) in enumerate(encoded["offset_mapping"]):
        while at < len(pieces) - 1 and begin >= pieces[at]["stop"]:
            at, counted = at + 1, 0
        piece = pieces[at]
        message = messages[piece["message"]] if piece["message"] is not None else None
        learn = 0
        if piece["role"] == "assistant" and begin >= piece["inner"] and end <= piece["end"] and end > begin:
            if not counted:
                calls = message.get("tool_calls") or []
                name = calls[0]["function"]["name"] if calls else ""
                starts.append(index - 1)
                start_turn.append(piece["turn"])
                start_act.append(bool(calls))
                start_tool.append(names.index(name) if name in names else -1)
                start_skip.append(piece["message"] in skip)
            counted += 1
            if piece["message"] not in skip:
                learn = 3 if counted <= FIRST_TOKENS else (2 if message.get("tool_calls") else 1)
        before_next = 0 <= piece["turn"] < last
        own_late = before_next and final.get(piece["turn"]) == at and begin >= piece["close"]
        shift_late = before_next and not (piece["opener"] and begin < piece["close"])
        code.append(learn)
        tturn.append(piece["turn"])
        late.append((1 if own_late else 0) | (2 if shift_late else 0))
    forgot = any(NOTE_HEAD not in str(messages[index].get("content")) for index, number in opens.items() if number < last)
    return {"ids": torch.tensor(encoded["input_ids"], dtype=torch.int32), "code": torch.tensor(code, dtype=torch.uint8),
            "tturn": torch.tensor(tturn, dtype=torch.int8), "late": torch.tensor(late, dtype=torch.uint8),
            "starts": starts, "start_turn": start_turn, "start_act": start_act, "start_tool": start_tool,
            "start_skip": start_skip, "forgot": forgot}


def load_data(args, tokenizer) -> list[dict]:
    import torch

    cache = Path(args.out) / "encoded.pt"
    if cache.exists() and not args.fresh:
        return torch.load(cache)
    catalog = load_catalog(Path(args.tools))
    rows, dropped, kinds = [], 0, {}
    for line in Path(args.data).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        row = encode(tokenizer, record, catalog)
        if len(row["ids"]) > MAX_TOKENS or not int(row["code"].sum()):
            dropped += 1
            continue
        turns = record.get("turns") or [{"skill": record["skill"], "hard": record["hard"]}]
        family = record["kind"] if record.get("kind") in ("session", "repair", "real") else "single"
        kinds[family] = kinds.get(family, 0) + 1
        rows.append({"id": record["id"], **row, "turn_skill": [SKILLS.index(turn["skill"]) for turn in turns],
                     "turn_hard": [bool(turn.get("hard")) for turn in turns], "skill": SKILLS.index(record["skill"]),
                     "family": family, "split": record["split"]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache)
    lengths = sorted(len(row["ids"]) for row in rows)
    log(f"data: {len(rows)} conversations {kinds} ({dropped} dropped), tokens median {lengths[len(lengths) // 2]}, "
        f"max {lengths[-1]}, total {sum(lengths)}; held {sum(row['split'] == 'held' for row in rows)}; "
        f"turns {sum(len(row['turn_skill']) for row in rows)}; notes dropped from earlier turns in {sum(row['forgot'] for row in rows)}")
    return rows


def reserved(row: dict, share: float) -> bool:
    """Is this training conversation kept away from the trunk, so that its skill learns from text
    the trunk has not read? A skill trained on what the trunk already knows only learns it by heart."""
    return share > 0 and row["split"] == "train" and int(row["id"][8:16], 16) % 1000 < share * 1000


def batches(rows: list[dict], budget: int, rng: random.Random | None) -> list[list[int]]:
    order = list(range(len(rows)))
    if rng:
        rng.shuffle(order)
    made = []
    for begin in range(0, len(order), 512):
        block = sorted(order[begin:begin + 512], key=lambda i: len(rows[i]["ids"]))
        current, longest = [], 0
        for index in block:
            size = len(rows[index]["ids"])
            if current and max(longest, size) * (len(current) + 1) > budget:
                made.append(current)
                current, longest = [], 0
            current.append(index)
            longest = max(longest, size)
        if current:
            made.append(current)
    if rng:
        rng.shuffle(made)
    return made


def mode_of(row: dict) -> str:
    """Which real situation a conversation is trained as. Most of the time the one a running session is
    in: every turn was computed under its own skill ("own"), or, when Kiwibro has taken the context
    notes out of the earlier messages, what follows such a message was computed again when the next turn
    came ("shifted"). Sometimes a session read again from nothing, all of it under the skill of the newest
    turn ("flat"), and that with the system turn under the skill too, as engines before 2.0 do ("flat_sys")."""
    pick = int(row["id"][4:8], 16) % 100
    if pick < 70:
        return "shifted" if row["forgot"] else "own"
    return "flat" if pick < 90 else "flat_sys"


def route_of(row: dict, mode: str):
    """The skill every position is computed under (-1: none), and which tokens that leaves to be learned:
    a token is learned only where the position that predicts it runs under the skill of its own turn."""
    import torch

    turn = row["tturn"].long()
    skills = torch.tensor(row["turn_skill"])
    last = len(row["turn_skill"]) - 1
    if mode in ("flat", "flat_sys"):
        moved = torch.full_like(turn, last)
    else:
        late = (row["late"] & (1 if mode == "own" else 2)) > 0
        moved = torch.clamp(turn + late.long(), max=last)
    route = skills[moved.clamp_min(0)]
    if mode != "flat_sys":
        route[turn < 0] = -1
    learned = torch.zeros_like(turn, dtype=torch.bool)
    learned[1:] = route[:-1] == skills[turn.clamp_min(0)][1:]
    return route, learned


def collate(rows: list[dict], indexes: list[int], pad: int, device, routed: bool = False, mode: str | None = None) -> dict:
    """A batch: ids, attention, how much each token counts in the loss, the skill of the turn each token
    belongs to (-1: none) and, with routed, the skill each position is computed under."""
    import torch

    width = max(len(rows[i]["ids"]) for i in indexes)
    ids = torch.full((len(indexes), width), pad, dtype=torch.long)
    attention = torch.zeros((len(indexes), width), dtype=torch.long)
    weight = torch.zeros((len(indexes), width))
    owner = torch.full((len(indexes), width), -1, dtype=torch.long)
    route = torch.full((len(indexes), width), -1, dtype=torch.long)
    table = torch.tensor(WEIGHTS)
    for row, index in enumerate(indexes):
        item = rows[index]
        size = len(item["ids"])
        ids[row, :size] = item["ids"].long()
        attention[row, :size] = 1
        counts = table[item["code"].long()]
        turn = item["tturn"].long()
        owner[row, :size] = torch.where(turn < 0, torch.tensor(-1), torch.tensor(item["turn_skill"])[turn.clamp_min(0)])
        if routed:
            route[row, :size], learned = route_of(item, mode or mode_of(item))
            counts = counts * learned
        weight[row, :size] = counts
    return {"ids": ids.to(device), "attention": attention.to(device), "weight": weight.to(device),
            "owner": owner.to(device), "route": route.to(device) if routed else None}


# ------------------------------------------------------------------ the model

def fake_quant(weight, bits: int = 4):
    """The weight as the engine will store it (groups of 32 along the input, fp16 scale), with a
    straight-through gradient. Mirrors scripts/mind/formats.py quantize_q4 / quantize_q8."""
    import torch

    top = 7.0 if bits == 4 else 127.0
    groups = weight.reshape(weight.shape[0], -1, GROUP)
    scale = (groups.abs().amax(dim=-1, keepdim=True) / top).clamp_min(2.0 ** -14).to(torch.float16).to(weight.dtype)
    snapped = (torch.clamp(torch.round(groups / scale), -top - (1 if bits == 4 else 0), top) * scale).reshape(weight.shape)
    return weight + (snapped - weight).detach()


def load_model(path: str = BASE, state: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    if state is not None:
        model.load_state_dict(torch.load(state), strict=True)
    model.config.use_cache = False
    return tokenizer, model.cuda()


def linears(model, first: int = 0):
    for index, layer in enumerate(model.model.layers):
        if index >= first:
            for matrix, (holder, name) in MATRICES.items():
                yield index, matrix, getattr(layer, holder), name


def make_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class QLinear(nn.Module):
        """A linear layer whose weight is seen through the 4-bit grid while it trains."""

        def __init__(self, linear):
            super().__init__()
            if linear.bias is not None:
                raise ValueError("the engine stores no biases")
            self.weight = linear.weight

        def forward(self, x):
            return F.linear(x, fake_quant(self.weight, 4))

    class QEmbedding(nn.Module):
        def __init__(self, embedding, bits: int):
            super().__init__()
            self.weight, self.bits = embedding.weight, bits

        def forward(self, ids):
            rows = F.embedding(ids, self.weight)
            return fake_quant(rows.reshape(-1, rows.shape[-1]), self.bits).reshape(rows.shape)

    class SkillLinear(nn.Module):
        """A frozen linear layer plus the low-rank change of every skill: y = W x + scale * B_s (A_s x), where s
        is the skill of the position (route; -1 = none). So one pass runs each turn of a conversation
        under its own skill, as the engine does when it keeps what earlier turns computed."""

        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            out_dim, in_dim = linear.weight.shape
            self.a = nn.Parameter(torch.empty(len(SKILLS), RANK, in_dim))
            self.b = nn.Parameter(torch.zeros(len(SKILLS), out_dim, RANK))
            self.route, self.tail, self.present = None, None, []
            for index in range(len(SKILLS)):
                nn.init.kaiming_uniform_(self.a.data[index], a=math.sqrt(5))

        def forward(self, x):
            y = self.linear(x)
            route = self.route
            if route is None:
                return y
            if route.shape[1] != x.shape[1]:  # while a reply is written: one new position per conversation
                route = self.tail[:, None].expand(x.shape[0], x.shape[1])
            for skill in self.present:
                pick = route == skill
                change = F.linear(F.linear(x[pick], self.a[skill].to(x.dtype)), self.b[skill].to(x.dtype)) * (ALPHA / RANK)
                y = y.index_put((pick,), change.to(y.dtype), accumulate=True)
            return y

    return QLinear, QEmbedding, SkillLinear


def token_losses(model, table, batch: dict):
    """Cross-entropy of every token the batch teaches, and which tokens those are."""
    import torch
    import torch.nn.functional as F

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=batch["ids"], attention_mask=batch["attention"], use_cache=False).last_hidden_state
        wanted = batch["weight"][:, 1:] > 0
        logits = F.linear(hidden[:, :-1][wanted], table.to(hidden.dtype))
    return F.cross_entropy(logits.float(), batch["ids"][:, 1:][wanted], reduction="none"), wanted


def loss_on(model, table, batch: dict):
    """The training loss: every token the assistant writes, the ones that decide what a reply does
    (its first tokens, a tool call) counted more than the words of an answer."""
    losses, wanted = token_losses(model, table, batch)
    weight = batch["weight"][:, 1:][wanted]
    return (losses * weight).sum() / weight.sum().clamp_min(1e-6), int(wanted.sum())


def held_losses(model, table, rows: list[dict], pad: int, budget: int, modules: dict | None = None,
                mode: str | None = None) -> tuple[float, dict[str, float]]:
    """Plain cross-entropy on the held-back conversations: all of it, and per skill (the tokens of the
    turns that belong to that skill). With modules, every position runs under its skill."""
    import torch

    total = torch.zeros(len(SKILLS), dtype=torch.float64)
    count = torch.zeros(len(SKILLS), dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for indexes in batches(rows, budget, None):
            batch = collate(rows, indexes, pad, "cuda", routed=modules is not None, mode=mode)
            if modules is not None:
                set_route(modules, batch["route"])
            losses, wanted = token_losses(model, table() if callable(table) else table, batch)
            owner = batch["owner"][:, 1:][wanted].cpu()
            total.index_add_(0, owner, losses.double().cpu())
            count.index_add_(0, owner, torch.ones(len(owner), dtype=torch.float64))
    model.train()
    return (float(total.sum() / count.sum().clamp_min(1)),
            {skill: float(total[i] / count[i]) for i, skill in enumerate(SKILLS) if count[i]})


def fit_plan(plan: list, step: int, mark: float, seconds: float, name: str) -> tuple[list, float]:
    """Cut a training plan to the time it may take, judged by the pace of ten early steps."""
    import torch

    if not seconds or step not in (2, 12):
        return plan, mark
    torch.cuda.synchronize()
    if step == 2:
        return plan, time.time()
    pace = (time.time() - mark) / 10
    fits = max(20, int(seconds / pace))
    if fits < len(plan):
        log(f"{name}: {pace:.2f} s a step, so {fits} of the planned {len(plan)} steps fit {seconds / 60:.1f} min")
        plan = plan[:fits]
    return plan, mark


def schedule(step: int, steps: int, peak: float) -> float:
    warm = max(1, int(steps * 0.04))
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))


# ------------------------------------------------------------------ stage: trunk

def stage_trunk(args) -> None:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    QLinear, QEmbedding, _ = make_modules()
    tokenizer, model = load_model()
    rows = load_data(args, tokenizer)
    train = [row for row in rows if row["split"] == "train" and not reserved(row, args.skill_share)]
    held = [row for row in rows if row["split"] == "held"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bits = 4 if args.emb == "q4" else 8
    table_weight = model.model.embed_tokens.weight
    base_loss = held_losses(model, table_weight, held, pad, args.budget)[0]
    for _, _, holder, name in linears(model):
        setattr(holder, name, QLinear(getattr(holder, name)))
    model.model.embed_tokens = QEmbedding(model.model.embed_tokens, bits)
    table = lambda: fake_quant(table_weight, bits)  # noqa: E731
    start_loss = held_losses(model, table, held, pad, args.budget)[0]
    log(f"trunk: held loss float base {base_loss:.4f}, on the grid before training {start_loss:.4f}")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.smoke:
        optimizer = torch.optim.SGD(parameters, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    rng = random.Random(SEED)
    plan = [b for _ in range(args.epochs) for b in batches(train, args.budget, rng)]
    if args.smoke:
        plan = plan[:6]
    log(f"trunk: {len(train)} conversations, {len(plan)} steps")
    model.train()
    began, seen, step, mark = time.time(), 0, 0, 0.0
    # A small data set is learned by heart within a few passes, and the held-back loss then rises
    # again. So it is looked at several times and the weights that did best there are the ones kept.
    every = max(1, len(plan) // max(1, args.checks or args.epochs))
    best_loss, best_weights, best_step = start_loss, None, 0
    while step < len(plan):
        indexes = plan[step]
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, len(plan), args.lr)
        loss, size = loss_on(model, table(), collate(train, indexes, pad, "cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += sum(len(train[i]["ids"]) for i in indexes)
        if step % 25 == 0 or step == len(plan) - 1:
            log(f"trunk: step {step}/{len(plan)} loss {float(loss.detach()):.4f} {seen / (time.time() - began):.0f} tok/s")
        if (step + 1) % every == 0 or step == len(plan) - 1:
            now = held_losses(model, table, held, pad, args.budget)[0]
            log(f"trunk: held loss {now:.4f} after step {step + 1}")
            if args.checks and now < best_loss:
                best_loss, best_step = now, step + 1
                best_weights = [parameter.detach().clone() for parameter in parameters]
        plan, mark = fit_plan(plan, step, mark, args.trunk_minutes * 60, "trunk")
        step += 1
    if best_weights is not None:
        with torch.no_grad():
            for parameter, kept in zip(parameters, best_weights):
                parameter.copy_(kept)
        del best_weights
        log(f"trunk: kept the weights of step {best_step}, held loss {best_loss:.4f}")
    with torch.no_grad():  # snap to the grid: from here on the weights are exactly what the engine stores
        for _, _, holder, name in linears(model):
            module = getattr(holder, name)
            module.weight.data = fake_quant(module.weight.data, 4)
        table_weight.data = fake_quant(table_weight.data, bits)
    state = {key.replace(".weight.weight", ".weight"): value.detach().cpu() for key, value in model.state_dict().items()}
    out = Path(args.out)
    torch.save(state, out / "trunk.pt")
    final, per_skill = held_losses(model, lambda: table_weight, held, pad, args.budget)
    (out / "trunk.json").write_text(json.dumps({"held_loss_float_base": base_loss, "held_loss_grid_untrained": start_loss,
                                                "held_loss_trunk": final, "emb": args.emb, "steps": len(plan),
                                                "kept_step": best_step or len(plan), "conversations": len(train),
                                                "held_loss_per_skill": per_skill}, indent=1))
    log(f"trunk: saved; held loss of the snapped trunk {final:.4f}")


# ------------------------------------------------------------------ stage: skills

def attach_skills(model):
    _, _, SkillLinear = make_modules()
    modules = {}
    for index, matrix, holder, name in linears(model, SPLIT):
        module = SkillLinear(getattr(holder, name))
        setattr(holder, name, module)
        modules[(index, matrix)] = module
    return modules


def set_route(modules: dict, route, tail=None) -> None:
    """Which skill each position of the next pass runs under (None: no skill anywhere). tail: the skill
    of the positions a conversation adds while a reply is written."""
    present = []
    if route is not None:
        seen = set(route.unique().tolist()) | (set(tail.unique().tolist()) if tail is not None else set())
        present = sorted(int(skill) for skill in seen if skill >= 0)
    for module in modules.values():
        module.route, module.tail, module.present = route, tail, present


def load_skills(modules: dict, folder: Path) -> list[str]:
    """Put the saved skills into the model; one that was never saved stays without effect."""
    import torch

    found = []
    for number, skill in enumerate(SKILLS):
        path = folder / f"{skill}.pt"
        state = torch.load(path) if path.exists() else None
        found += [skill] if state is not None else []
        with torch.no_grad():
            for (index, matrix), module in modules.items():
                if state is None:
                    module.b[number].zero_()
                else:
                    module.a[number].copy_(state[f"{index}.{matrix}.a"].float())
                    module.b[number].copy_(state[f"{index}.{matrix}.b"].float())
    return found


def stage_skills(args) -> None:
    """All skills in one run. Every conversation is read as a running session computes it: the system turn
    under no skill, each turn under its own. A skill only gets gradient from the positions it runs, so the
    skills stay separate files, and a change of skill in the engine costs nothing: what earlier turns
    computed stays valid."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out)
    tokenizer, model = load_model(state=out / "trunk.pt")
    rows = load_data(args, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = attach_skills(model)
    model.cuda()
    table = model.model.embed_tokens.weight
    train = [row for row in rows if row["split"] == "train" and (not args.skill_share or reserved(row, args.skill_share))]
    held = [row for row in rows if row["split"] == "held"]
    set_route(modules, None)
    before_all, before = held_losses(model, table, held, pad, args.budget)
    a_side, b_side = [module.a for module in modules.values()], [module.b for module in modules.values()]
    parameters = a_side + b_side
    optimizer = torch.optim.AdamW([{"params": a_side, "rate": 1.0}, {"params": b_side, "rate": B_RATE}],
                                  lr=args.skill_lr, betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(SEED)
    plan = [b for _ in range(args.skill_epochs) for b in batches(train, args.budget, rng)]
    if args.smoke:
        plan = plan[:4]
    every = args.skill_check or max(1, len(plan) // 12)
    log(f"skills: {len(train)} conversations, {len(plan)} steps, a look at the held-back loss every {every}; "
        f"trunk alone {before_all:.4f} {json.dumps({k: round(v, 4) for k, v in before.items()})}")
    model.train()
    began, seen, step, mark = time.time(), 0, 0, 0.0
    # A skill starts as no change at all. Each is kept as it was at its own lowest held-back loss, and
    # one that never gets below the trunk alone stays empty: a skill may help, it must not harm.
    best, kept, kept_step = dict(before), {}, {}
    while step < len(plan):
        indexes = plan[step]
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, len(plan), args.skill_lr) * group["rate"]
        batch = collate(train, indexes, pad, "cuda", routed=True)
        set_route(modules, batch["route"])
        loss, size = loss_on(model, table, batch)
        if size:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += sum(len(train[i]["ids"]) for i in indexes)
        if step % 10 == 0:
            log(f"skills: step {step}/{len(plan)} loss {float(loss.detach()):.4f} {seen / (time.time() - began):.0f} tok/s")
        plan, mark = fit_plan(plan, step, mark, args.skill_minutes * 60, "skills")
        step += 1
        if step % every == 0 or step >= len(plan):
            _, now = held_losses(model, table, held, pad, args.budget, modules)
            for number, skill in enumerate(SKILLS):
                if skill in now and now[skill] < best.get(skill, float("inf")):
                    best[skill], kept_step[skill] = now[skill], step
                    kept[skill] = [(module.a[number].detach().clone(), module.b[number].detach().clone())
                                   for module in modules.values()]
            log(f"skills: held loss after step {step}: {json.dumps({k: round(v, 4) for k, v in now.items()})}")
    with torch.no_grad():
        for number, skill in enumerate(SKILLS):
            for place, module in enumerate(modules.values()):
                if skill in kept:
                    module.a[number].copy_(kept[skill][place][0])
                    module.b[number].copy_(kept[skill][place][1])
                else:
                    module.b[number].zero_()  # no change: the trunk alone was better at every look
    after_all, after = held_losses(model, table, held, pad, args.budget, modules)
    (out / "skills").mkdir(exist_ok=True)
    report = {}
    for number, skill in enumerate(SKILLS):
        state = {}
        for (index, matrix), module in modules.items():
            state[f"{index}.{matrix}.a"] = module.a[number].detach().cpu().half()
            state[f"{index}.{matrix}.b"] = module.b[number].detach().cpu().half()
        torch.save(state, out / "skills" / f"{skill}.pt")
        report[skill] = {"kept_step": kept_step.get(skill, 0), "held_loss_trunk": before.get(skill),
                         "held_loss_with_skill": after.get(skill)}
    report["all"] = {"conversations": len(train), "steps": len(plan), "held_loss_trunk": before_all,
                     "held_loss_with_skills": after_all, "b_rate": B_RATE}
    (out / "skills.json").write_text(json.dumps(report, indent=1))
    log("skills: " + json.dumps(report))


# ------------------------------------------------------------------ stage: head

def split_states(model, ids, attention):
    """The state entering layer SPLIT: what the shared layers hand to the skills and to the head."""
    import torch

    class Enough(Exception):
        pass

    caught = {}

    def hook(_module, positional, named):
        caught["x"] = positional[0] if positional else named["hidden_states"]
        raise Enough

    handle = model.model.layers[SPLIT].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.model(input_ids=ids, attention_mask=attention, use_cache=False)
    except Enough:
        pass
    finally:
        handle.remove()
    return caught["x"].float()


def normed(states):
    import torch

    return states / torch.sqrt((states * states).mean(dim=-1, keepdim=True) + 1e-6)


def unit_names(catalog: dict) -> list[str]:
    return ["act", *(f"tool:{name}" for name in catalog)]


def stage_head(args) -> None:
    """Two small readers of the state the shared layers hand over at the position before a reply. The
    head says which skill the turn needs and whether it is hard. The action unit says whether the reply
    will be a tool call, and of which tool: the app can know that before a single word is written."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    out = Path(args.out)
    catalog = load_catalog(Path(args.tools))
    tokenizer, model = load_model(state=out / "trunk.pt")
    model.eval()
    rows = load_data(args, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rng = random.Random(SEED)

    def marks(row: dict, shift: int = 0) -> list[tuple]:
        return [(position + shift, row["turn_skill"][turn], float(row["turn_hard"][turn]), float(act), tool, number == 0)
                for number, (position, turn, act, tool, shown) in enumerate(zip(
                    row["starts"], row["start_turn"], row["start_act"], row["start_tool"], row["start_skip"]))
                if not shown]

    items = [{"ids": row["ids"], "marks": marks(row), "held": row["split"] == "held"} for row in rows]
    # a second copy of some conversations with another one in front, so the readers also see a
    # change of subject in the middle of a chat
    short = [row for row in rows if row["split"] == "train" and len(row["ids"]) < 1500]
    joined = 0
    for row in rows:
        if row["split"] != "train" or not short or rng.random() > 0.4:
            continue
        front, back = rng.choice(short)["ids"].tolist(), row["ids"].tolist()  # the joined text keeps the later system turn
        first_user = next((i for i, token in enumerate(front) if i > 0 and token == front[0]), 0)
        tail_start = next((i for i, token in enumerate(back) if i > 0 and token == back[0]), 0)
        if not first_user or not tail_start or len(back) + len(front) - first_user > MAX_TOKENS:
            continue
        items.append({"ids": torch.tensor(back[:tail_start] + front[first_user:] + back[tail_start:], dtype=torch.int32),
                      "marks": marks(row, len(front) - first_user), "held": False})
        joined += 1
    features, labels = [], []
    for indexes in batches(items, args.budget * 2, None):
        width = max(len(items[i]["ids"]) for i in indexes)
        ids = torch.full((len(indexes), width), pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for place, index in enumerate(indexes):
            size = len(items[index]["ids"])
            ids[place, :size] = items[index]["ids"].long()
            attention[place, :size] = 1
        states = split_states(model, ids.cuda(), attention.cuda())
        for place, index in enumerate(indexes):
            for position, skill, hard, act, tool, first in items[index]["marks"]:
                features.append(states[place, position].cpu())
                labels.append((skill, hard, act, tool, items[index]["held"], first))
    x = normed(torch.stack(features)).cuda()
    y = torch.tensor([label[0] for label in labels]).cuda()
    hard = torch.tensor([label[1] for label in labels]).cuda()
    act = torch.tensor([label[2] for label in labels]).cuda()
    tool = torch.tensor([label[3] for label in labels]).cuda()
    held = torch.tensor([label[4] for label in labels]).cuda()
    first = torch.tensor([label[5] for label in labels]).cuda()
    log(f"head: {int((~held).sum())} training states, {int(held.sum())} held ({joined} joined conversations)")

    def fit(outputs: int, loss_of):
        net = nn.Sequential(nn.Linear(x.shape[1], args.head_hidden), nn.ReLU(), nn.Linear(args.head_hidden, outputs)).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        train = (~held).nonzero().squeeze(1)
        for _ in range(2 if args.smoke else 40):
            order = train[torch.randperm(len(train), device="cuda")]
            for begin in range(0, len(order), 256):
                pick = order[begin:begin + 256]
                loss = loss_of(net(F.dropout(x[pick], 0.1)), pick)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return net.eval()

    def save(net, names: list[str], name: str) -> None:
        torch.save({"w1": net[0].weight.detach().cpu(), "b1": net[0].bias.detach().cpu(), "w2": net[2].weight.detach().cpu(),
                    "b2": net[2].bias.detach().cpu(), "names": names, "layer": SPLIT, "eps": 1e-6}, out / f"{name}.pt")

    net = fit(len(HEAD_NAMES), lambda got, pick: F.cross_entropy(got[:, :len(SKILLS)], y[pick])
              + F.binary_cross_entropy_with_logits(got[:, -1], hard[pick]))
    with torch.no_grad():
        got = net(x[held])
    right = got[:, :len(SKILLS)].argmax(dim=-1) == y[held]
    said_hard, is_hard = got[:, -1] > 0, hard[held] > 0.5
    report = {"states_train": int((~held).sum()), "states_held": int(held.sum()),
              "skill_accuracy": float(right.float().mean()),
              "skill_accuracy_first_turn": float(right[first[held]].float().mean()),
              "skill_accuracy_later_turns": float(right[~first[held]].float().mean()),
              "per_skill": {skill: float(right[y[held] == number].float().mean()) for number, skill in enumerate(SKILLS)
                            if int((y[held] == number).sum())},
              "hard_precision": float((said_hard & is_hard).sum() / max(1, int(said_hard.sum()))),
              "hard_recall": float((said_hard & is_hard).sum() / max(1, int(is_hard.sum()))),
              "hard_share": float(is_hard.float().mean())}
    save(net, HEAD_NAMES, "head")
    (out / "head.json").write_text(json.dumps(report, indent=1))
    log("head: " + json.dumps(report))

    names = unit_names(catalog)

    def unit_loss(got, pick):
        loss = F.binary_cross_entropy_with_logits(got[:, 0], act[pick])
        calls = tool[pick] >= 0
        return loss + (F.cross_entropy(got[calls][:, 1:], tool[pick][calls]) if bool(calls.any()) else 0.0)

    unit = fit(len(names), unit_loss)
    with torch.no_grad():
        got = unit(x[held])
    sure, is_act, gold = torch.sigmoid(got[:, 0]), act[held] > 0.5, tool[held]
    # From which reading on is "a tool call comes" right at least 97 times in 100, and up to which is "no
    # tool call comes"? The app acts on the unit only beyond these two marks.
    high = next((mark for mark in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
                 if int((sure >= mark).sum()) >= 20 and float(is_act[sure >= mark].float().mean()) >= 0.97), None)
    low = next((mark for mark in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.01)
                if int((sure <= mark).sum()) >= 20 and float((~is_act)[sure <= mark].float().mean()) >= 0.97), None)
    calls = is_act & (gold >= 0)
    top3 = got[:, 1:].topk(3, dim=-1).indices
    report = {"states_held": int(held.sum()), "act_share": float(is_act.float().mean()),
              "act_accuracy": float(((sure >= 0.5) == is_act).float().mean()),
              "act_high": high, "act_high_reach": float((sure[is_act] >= high).float().mean()) if high else 0.0,
              "act_low": low, "act_low_reach": float((sure[~is_act] <= low).float().mean()) if low else 0.0,
              "tool_top1": float((top3[calls][:, 0] == gold[calls]).float().mean()),
              "tool_top3": float((top3[calls] == gold[calls][:, None]).any(dim=-1).float().mean())}
    save(unit, names, "units")
    (out / "units.json").write_text(json.dumps(report, indent=1))
    log("units: " + json.dumps(report))


# ------------------------------------------------------------------ stage: eval

def turn_cases(record: dict) -> list[dict]:
    """One case per turn of a held-back conversation: everything up to the first thing the assistant writes
    in that turn, and the tool it should call first ("" = answer in words)."""
    messages, skip = record["messages"], set(record.get("skip") or [])
    turns = record.get("turns") or [{"at": next(i for i, m in enumerate(messages) if m["role"] == "user"),
                                     "skill": record["skill"]}]
    cases = []
    for number, turn in enumerate(turns):
        reply = next((i for i in range(turn["at"] + 1, len(messages)) if messages[i]["role"] == "assistant"), None)
        if reply is None or reply in skip or NOTE_HEAD not in str(messages[turn["at"]].get("content")):
            continue  # a turn whose note Kiwibro has since dropped is not how the turn looked when it was answered
        calls = messages[reply].get("tool_calls") or []
        group = "after a bad reply" if record.get("kind") == "repair" else (f"first/{turn['skill']}" if number == 0 else "later turn")
        cases.append({"id": record["id"], "messages": messages[:reply], "turns": turns[:number + 1], "group": group,
                      "gold": calls[0]["function"]["name"] if calls else "", "skill": SKILLS.index(turn["skill"])})
    return cases


def ready_case(tokenizer, case: dict, catalog: dict, header: list[int]) -> dict:
    """The case as the engine meets it: the tokens of the prompt and the skill each position runs under."""
    import torch

    row = encode(tokenizer, case, catalog, full=True)
    row["turn_skill"] = [SKILLS.index(turn["skill"]) for turn in case["turns"]]
    route, _ = route_of(row, "shifted" if row["forgot"] else "own")
    return {**case, "ids": torch.cat([row["ids"].long(), torch.tensor(header)]),
            "route": torch.cat([route, torch.full((len(header),), case["skill"])])}


def tool_scores(model, tokenizer, modules: dict, cases: list[dict], catalog: dict, pad: int, routed: bool, unit=None) -> dict:
    """Greedy first replies: is it a tool call when it should be, the right tool, a valid call. With unit,
    the action unit's reading of the same prompts next to what the model then wrote."""
    import torch

    stops = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "</tool_call>", "<|endoftext|>")]
    names = list(catalog)
    tally: dict[str, dict[str, int]] = {}
    for begin in range(0, len(cases), CASE_BATCH):
        part = cases[begin:begin + CASE_BATCH]
        width = max(len(case["ids"]) for case in part)
        ids = torch.full((len(part), width), pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        route = torch.full((len(part), width), -1, dtype=torch.long)
        for place, case in enumerate(part):
            size = len(case["ids"])
            ids[place, width - size:], attention[place, width - size:], route[place, width - size:] = case["ids"], 1, case["route"]
        ids, attention = ids.cuda(), attention.cuda()
        reading = None
        if unit is not None:
            with torch.no_grad():
                reading = unit(normed(split_states(model, ids, attention)[:, -1])).cpu()
        set_route(modules, route.cuda() if routed else None,
                  torch.tensor([case["skill"] for case in part]).cuda() if routed else None)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            made = model.generate(input_ids=ids, attention_mask=attention, max_new_tokens=96, do_sample=False,
                                  eos_token_id=stops, pad_token_id=pad, use_cache=True)
        for place, (case, row) in enumerate(zip(part, made[:, width:])):
            text = tokenizer.decode(row, skip_special_tokens=False)
            for token in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(token, "")
            text = case.get("opened", "") + text
            _, calls = RENDER.parse_reply(text)
            name, gold = (calls[0]["name"] if calls else ""), case["gold"]
            for group in (case["group"], "all"):
                count = tally.setdefault(group, {key: 0 for key in ("cases", "decided", "right", "wanted", "named", "valid",
                                                                    "unit_decided", "unit_named", "unit_and_model_differ")})
                count["cases"] += 1
                count["decided"] += bool(name) == bool(gold)
                count["right"] += name == gold
                if gold:
                    count["wanted"] += 1
                    count["named"] += name == gold
                    schema = catalog.get(name, {}).get("function", {}).get("parameters") or {}
                    count["valid"] += bool(name in catalog and all(key in calls[0]["arguments"] for key in schema.get("required") or []))
                if reading is not None:
                    says_act = bool(reading[place, 0] > 0)
                    count["unit_decided"] += says_act == bool(gold)
                    count["unit_and_model_differ"] += says_act != bool(name)
                    count["unit_named"] += bool(gold) and names[int(reading[place, 1:].argmax())] == gold
    set_route(modules, None)
    scores = {}
    for group, count in sorted(tally.items()):
        scores[group] = {"cases": count["cases"], "tool_or_answer_right": count["decided"] / max(1, count["cases"]),
                         "reply_right": count["right"] / max(1, count["cases"]),
                         "cases_wanting_a_tool": count["wanted"], "tool_name_right": count["named"] / max(1, count["wanted"]),
                         "call_valid": count["valid"] / max(1, count["wanted"])}
        if unit is not None:
            scores[group].update({"unit_tool_or_answer_right": count["unit_decided"] / max(1, count["cases"]),
                                  "unit_tool_name_right": count["unit_named"] / max(1, count["wanted"]),
                                  "unit_and_model_differ": count["unit_and_model_differ"] / max(1, count["cases"])})
    return scores


def unit_sure(model, cases: list[dict], unit, pad: int) -> list[float]:
    """Per case, how sure the action unit is that the reply to come is a tool call."""
    import torch

    sure: list[float] = []
    for begin in range(0, len(cases), CASE_BATCH):
        part = cases[begin:begin + CASE_BATCH]
        width = max(len(case["ids"]) for case in part)
        ids = torch.full((len(part), width), pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for place, case in enumerate(part):
            ids[place, width - len(case["ids"]):], attention[place, width - len(case["ids"]):] = case["ids"], 1
        with torch.no_grad():
            sure += torch.sigmoid(unit(normed(split_states(model, ids.cuda(), attention.cuda())[:, -1]))[:, 0]).tolist()
    return sure


def stage_eval(args) -> None:
    import torch
    import torch.nn as nn

    out = Path(args.out)
    catalog = load_catalog(Path(args.tools))
    tokenizer, model = load_model()
    rows = [row for row in load_data(args, tokenizer) if row["split"] == "held"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    report = {"held_conversations": len(rows), "loss": {}, "tools": {}}
    report["loss"]["float base"] = held_losses(model, model.model.embed_tokens.weight, rows, pad, args.budget)[1]
    del model
    torch.cuda.empty_cache()
    header = tokenizer(f"{RENDER.IM_START}assistant\n", add_special_tokens=False)["input_ids"]
    per, taken, cases = (4 if args.smoke else args.cases), {}, []
    for line in Path(args.data).read_text().splitlines():
        record = json.loads(line) if line.strip() else {"split": ""}
        if record["split"] != "held":
            continue
        for case in turn_cases(record):
            most = per if case["group"].startswith("first/") else per * 3
            if taken.get(case["group"], 0) < most:
                taken[case["group"]] = taken.get(case["group"], 0) + 1
                ready = ready_case(tokenizer, case, catalog, header)
                if len(ready["ids"]) <= MAX_TOKENS:
                    cases.append(ready)
    cases.sort(key=lambda case: len(case["ids"]))
    tokenizer, model = load_model(state=out / "trunk.pt")
    modules = attach_skills(model)
    model.cuda().eval()
    table = model.model.embed_tokens.weight
    found = load_skills(modules, out / "skills")
    unit = None
    if (out / "units.pt").exists():
        saved = torch.load(out / "units.pt")
        unit = nn.Sequential(nn.Linear(saved["w1"].shape[1], saved["w1"].shape[0]), nn.ReLU(),
                             nn.Linear(saved["w2"].shape[1], saved["w2"].shape[0])).cuda().eval()
        with torch.no_grad():
            for layer, weight, bias in ((unit[0], "w1", "b1"), (unit[2], "w2", "b2")):
                layer.weight.copy_(saved[weight])
                layer.bias.copy_(saved[bias])
    set_route(modules, None)
    report["loss"]["trunk"] = held_losses(model, table, rows, pad, args.budget)[1]
    report["loss"]["trunk + skills"] = held_losses(model, table, rows, pad, args.budget, modules)[1]
    report["loss"]["trunk + skills, session read again from nothing"] = held_losses(model, table, rows, pad, args.budget, modules, "flat")[1]
    report["loss"]["trunk + skills, engine before 2.0"] = held_losses(model, table, rows, pad, args.budget, modules, "flat_sys")[1]
    report["skills"] = found
    report["tools"]["trunk"] = tool_scores(model, tokenizer, modules, cases, catalog, pad, False, unit)
    report["tools"]["trunk + skills"] = tool_scores(model, tokenizer, modules, cases, catalog, pad, True)
    marks = json.loads((out / "units.json").read_text()) if (out / "units.json").exists() else {}
    report["open_at"] = None
    if unit is not None and marks.get("act_high"):
        # Would it help to open the reply as a tool call wherever the unit is sure that one comes? The pack
        # tells the app to do so only if more of these held-back replies come out right that way.
        start = "<tool_call>\n"
        opener = torch.tensor(tokenizer(start, add_special_tokens=False)["input_ids"])
        sure = unit_sure(model, cases, unit, pad)
        opened = [{**case, "opened": start, "ids": torch.cat([case["ids"], opener]),
                   "route": torch.cat([case["route"], torch.full((len(opener),), case["skill"])])}
                  if reading >= marks["act_high"] else case for case, reading in zip(cases, sure)]
        name = "trunk + skills, a sure reply opened as a tool call"
        report["tools"][name] = tool_scores(model, tokenizer, modules, opened, catalog, pad, True)
        report["opened_share"] = sum(reading >= marks["act_high"] for reading in sure) / max(1, len(cases))
        if report["tools"][name]["all"]["reply_right"] > report["tools"]["trunk + skills"]["all"]["reply_right"]:
            report["open_at"] = marks["act_high"]
    (out / "eval.json").write_text(json.dumps(report, indent=1))
    log("eval: " + json.dumps(report))


# ------------------------------------------------------------------ stage: export

def stage_export(args) -> None:
    import numpy as np
    import torch

    formats = load_module("kiwimind_formats", HERE / "formats.py")
    out = Path(args.out)
    pack = out / "pack"
    (pack / "skills").mkdir(parents=True, exist_ok=True)
    state = torch.load(out / "trunk.pt")
    weights = {key: value.float().numpy() for key, value in state.items()}
    config = formats.config_from_hf(json.loads((Path(BASE) / "config.json").read_text()))
    emb = json.loads((out / "trunk.json").read_text())["emb"]
    formats.write_trunk(pack / "kiwimind-trunk.fkm", config, weights, mode="q4", emb=emb, max_ctx=args.ctx)
    # the packer must find the same grid the trunk was trained on: read it back and compare
    _, back, _ = formats.read_trunk(pack / "kiwimind-trunk.fkm")
    worst = 0.0
    for key in ("model.layers.0.self_attn.q_proj.weight", f"model.layers.{config['n_layers'] - 1}.mlp.down_proj.weight"):
        if key in back:
            worst = max(worst, float(np.abs(back[key] - weights[key]).max()))
    log(f"export: trunk written; largest difference to the trained weights in two sampled tensors {worst:.3e}")
    for path in sorted((out / "skills").glob("*.pt")):
        skill = torch.load(path)
        entries = []
        for key in sorted({name.rsplit(".", 1)[0] for name in skill}):
            layer, matrix = key.split(".")
            entries.append({"layer": int(layer), "matrix": matrix, "a": skill[f"{key}.a"].numpy(),
                            "b": skill[f"{key}.b"].numpy(), "scale": ALPHA / RANK})
        entries.sort(key=lambda e: (e["layer"], list(MATRICES).index(e["matrix"])))
        formats.write_skill(pack / "skills" / f"{path.stem}.kms", path.stem, SPLIT, config["n_layers"], entries)
    head = torch.load(out / "head.pt")
    formats.write_head(pack / "head.kmh", head["w1"].numpy(), head["b1"].numpy(), head["w2"].numpy(), head["b2"].numpy(),
                       head["names"], layer=head["layer"], eps=head["eps"])
    units = torch.load(out / "units.pt") if (out / "units.pt").exists() else None
    if units is not None:  # the same file format as the head; the host reads it beside the engine
        formats.write_head(pack / "units.kmh", units["w1"].numpy(), units["b1"].numpy(), units["w2"].numpy(),
                           units["b2"].numpy(), units["names"], layer=units["layer"], eps=units["eps"])
    shutil.copy(Path(BASE) / "tokenizer.json", pack / "tokenizer.json")
    files = {}
    for path in sorted(pack.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            files[str(path.relative_to(pack))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    manifest = {"name": "kiwimind", "version": args.version, "base": "Qwen/Qwen3-0.6B", "split": SPLIT, "rank": RANK,
                "skills": [p.stem for p in sorted((pack / "skills").glob("*.kms"))], "head": HEAD_NAMES, "ctx": args.ctx,
                # scoped: the skills were trained with every turn under its own skill and the system turn under
                # none, so an engine may keep what it computed when the skill changes
                "scoped": True, "units": units["names"] if units is not None else [], "files": files}
    for name, key in (("trunk", "trunk"), ("skills", "skills_report"), ("head", "head_report"), ("units", "units_report"),
                      ("eval", "eval")):
        if (out / f"{name}.json").exists():
            manifest[key] = json.loads((out / f"{name}.json").read_text())
    # from which reading of the action unit on the app opens a reply as a tool call; None = it never does
    manifest["open_at"] = (manifest.get("eval") or {}).get("open_at") if units is not None else None
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log("export: " + json.dumps({name: info["bytes"] for name, info in files.items()}))


# ------------------------------------------------------------------ stage: parity

def stage_parity(args) -> None:
    """What the trained model picks next on held-back prompts, written down so that the pack's engine can be run
    on the same prompts and compared (scripts/mind/parity.py). A session is given in the steps a running engine
    meets it in: each earlier turn computed under its own skill, then the newest turn."""
    import torch

    out = Path(args.out)
    catalog = load_catalog(Path(args.tools))
    tokenizer, model = load_model(state=out / "trunk.pt")
    modules = attach_skills(model)
    model.cuda().eval()
    load_skills(modules, out / "skills")
    tools = list(catalog.values())
    header = tokenizer(f"{RENDER.IM_START}assistant\n", add_special_tokens=False)["input_ids"]
    end_mark, closing = tokenizer.convert_tokens_to_ids(RENDER.IM_END), RENDER.IM_END + "\n"
    cases, most = [], args.parity // 2
    taken, skipped = {"one turn": 0, "several turns": 0}, {"one turn": 0, "several turns": 0}
    for line in Path(args.data).read_text().splitlines():
        record = json.loads(line) if line.strip() else {"split": ""}
        for case in turn_cases(record)[-1:] if record["split"] == "held" else []:
            kind = "one turn" if len(case["turns"]) == 1 else "several turns"
            if taken[kind] >= most:
                continue
            ready = ready_case(tokenizer, case, catalog, header)
            ids, prompt = ready["ids"].tolist(), RENDER.render(case["messages"], tools)
            # when the next turn arrives the engine has computed all of a turn but the end mark of its last reply
            steps = [{"text": RENDER.render(case["messages"][:following["at"]], tools, add_generation_prompt=False)[:-len(closing)],
                      "skill": turn["skill"]} for turn, following in zip(case["turns"], case["turns"][1:])]
            steps.append({"text": prompt, "skill": case["turns"][-1]["skill"]})
            # the skill each position runs under when an engine takes these steps; only a case where that is
            # the route the model is run with here can be compared
            route = [-1] * (ids.index(end_mark) + 2)
            for step in steps:
                reach = len(tokenizer(step["text"], add_special_tokens=False)["input_ids"])
                route += [SKILLS.index(step["skill"])] * max(0, reach - len(route))
            if (len(ids) > MAX_TOKENS or tokenizer(prompt, add_special_tokens=False)["input_ids"] != ids
                    or route != ready["route"].tolist()):
                skipped[kind] += 1
                continue
            set_route(modules, ready["route"][None].cuda(), torch.tensor([case["skill"]]).cuda())
            with torch.no_grad():
                best = model(input_ids=ready["ids"][None].cuda()).logits[0, -1].float().topk(2)
            cases.append({"kind": kind, "steps": steps, "tokens": len(ids), "next": int(best.indices[0]),
                          "next_text": tokenizer.decode([int(best.indices[0])]),
                          "margin": float(best.values[0] - best.values[1])})
            taken[kind] += 1
    set_route(modules, None)
    (out / "parity.json").write_text(json.dumps({"version": args.version, "cases": cases}))
    log(f"parity: cases {taken}, left out because the steps do not give the trained route {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gen", "trunk", "skills", "head", "eval", "export", "parity"])
    parser.add_argument("--data", default=str(WORK / "data" / "conv.jsonl"))
    parser.add_argument("--out", default=str(WORK / "out"))
    parser.add_argument("--tools", default=str(WORK / "tools_full.json"))
    parser.add_argument("--examples", default=str(WORK / "result_examples.json"))
    parser.add_argument("--smoke", action="store_true", help="a few steps of everything, to find mistakes early")
    parser.add_argument("--fresh", action="store_true", help="encode the conversations again")
    parser.add_argument("--target", type=int, default=7000)
    parser.add_argument("--asks", default=str(WORK / "asks.json"), help="gen: real first messages from open data sets (made when missing)")
    parser.add_argument("--single", type=float, default=0.42, help="gen: prompts per skill and chunk, as a share of the chunk size: one request")
    parser.add_argument("--sessions", type=float, default=0.3, help="the same for conversations of several turns")
    parser.add_argument("--repairs", type=float, default=0.18, help="the same for a bad first reply put right")
    parser.add_argument("--real", type=float, default=0.6, help="the same, once and not per skill, for real messages from data sets")
    parser.add_argument("--minutes", type=float, default=22)
    parser.add_argument("--first", type=int, default=60, help="prompts per skill in the first chunk")
    parser.add_argument("--chunk", type=int, default=150, help="most prompts per skill in a later chunk")
    parser.add_argument("--per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3600)
    parser.add_argument("--schema", default="strict", choices=["strict", "loose", "none"])
    parser.add_argument("--gpu", type=float, default=0.80)
    parser.add_argument("--seqs", type=int, default=192)
    parser.add_argument("--emb", default="q8", choices=["q4", "q8"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checks", type=int, default=6,
                        help="trunk: look at the held-back loss this many times and keep the best weights; 0 = keep the last")
    parser.add_argument("--trunk-minutes", type=float, default=0, help="cut the trunk plan to this time; 0 = no cut")
    parser.add_argument("--skill-minutes", type=float, default=0, help="the same for all skills together")
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--budget", type=int, default=28000, help="tokens per batch")
    parser.add_argument("--skill-epochs", type=int, default=2)
    parser.add_argument("--skill-lr", type=float, default=1e-4)
    parser.add_argument("--skill-check", type=int, default=0,
                        help="skills: look at the held-back loss every this many steps and keep each skill's best state; 0 = twelve looks")
    parser.add_argument("--skill-share", type=float, default=0.3,
                        help="share of the training conversations the trunk never reads; each skill learns from its own part of it")
    parser.add_argument("--only", default="", help="gen: make conversations for just these skills, comma separated")
    parser.add_argument("--head-hidden", type=int, default=192)
    parser.add_argument("--cases", type=int, default=48, help="eval: generated first turns per skill")
    parser.add_argument("--parity", type=int, default=32, help="parity: prompts to write down, half of them sessions")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--version", default="2.0.0")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.smoke:
        global CASE_BATCH
        args.budget, CASE_BATCH = min(args.budget, 1500), 2  # small enough to run beside the teacher
    {"gen": stage_gen, "trunk": stage_trunk, "skills": stage_skills, "head": stage_head, "eval": stage_eval,
     "export": stage_export, "parity": stage_parity}[args.stage](args)


if __name__ == "__main__":
    main()
