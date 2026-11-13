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
    "terminal": ["run_command", "multi_command", "terminal_session", "system_info", "remote", "git", "write_file",
                 "read_file", "component"],
    "computer": ["system_info", "open_path", "clipboard", "notify", "screenshot", "describe_image", "file_manage",
                 "list_files", "workspace", "convert_document", "model_info", "component"],
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


def conversation_schema(skill: str, catalog: dict, count: int, mode: str) -> dict:
    fields: dict[str, dict] = {"earlier": {"type": "string"}, "user": {"type": "string"},
                               "facts": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}
    names = SKILL_TOOLS[skill]
    if names:
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
        fields["steps"] = {"type": "array", "items": step, "maxItems": 4}
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


def make_seed(skill: str, rng: random.Random) -> dict:
    weights = [task[0] for task in TASKS[skill]]
    _, kind, text = rng.choices(TASKS[skill], weights=weights)[0]
    text = text.format(topic=rng.choice(TOPICS), life=rng.choice(LIFE), webthing=rng.choice(WEBTHINGS),
                       field=rng.choice(FIELDS), pair=rng.choice(PAIRS), tech=rng.choice(TECH),
                       project=rng.choice(PROJECTS), newthing=rng.choice(NEW_THINGS), scriptjob=rng.choice(SCRIPT_JOBS),
                       lang=rng.choice(LANGS), sysq=rng.choice(SYSTEM_QUESTIONS), shelljob=rng.choice(SHELL_JOBS),
                       email=rng.choice(EMAILS), deletejob=rng.choice(DELETE_JOBS))
    seed = {"skill": skill, "kind": kind, "hard": kind == "hard", "time": local_time(rng), "exact": [],
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


def teacher_prefix(skill: str, catalog: dict, examples: dict) -> str:
    """Everything that is the same for a skill, first, so the teacher's server reuses it."""
    names = SKILL_TOOLS[skill]
    parts = [RULES, SKILL_NOTES[skill]]
    if names:
        parts.append("Tools for these conversations\n" + "\n".join(manual(catalog[name], examples) for name in names))
        if "component" in names:
            parts.append(COMPONENT_NOTES)
    parts.append(SHAPE_HEAD + (SHAPE_STEPS if names else "") + SHAPE_TAIL)
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


def with_note(text: str, when: str, facts=(), exact=(), guidance: str = "", offline: bool = False,
              long: bool = False) -> str:
    parts = [text.strip(), f"{NOTE_HEAD}\nLocal time: {when}" + (f"\n{OFFLINE_LINE}" if offline else "")]
    if long:
        guidance = ""
    if exact:
        parts.append(EXACT_HEAD + "\n" + "\n".join(f"- {line}" for line in exact))
    if facts:
        parts.append(FACTS_HEAD + "\n" + "\n".join(f"- {fact.strip()}" for fact in facts))
    if long:
        parts.append(LONG_BLOCK)
    if guidance:
        parts.append("## Guidance for this turn\n" + guidance)
    return "\n\n".join(parts)


def call_commands(call: dict) -> str:
    arguments = call.get("arguments") or {}
    found = [str(arguments.get(key) or "") for key in ("command", "input", "code")]
    found += [str(one) for one in arguments.get("commands") or [] if isinstance(arguments.get("commands"), list)]
    return "\n".join(found)


def build_record(conversation: dict, seed: dict, catalog: dict, rng: random.Random) -> tuple[dict | None, str]:
    """A checked training record, or None and the reason it was refused."""
    user, answer = str(conversation.get("user") or "").strip(), str(conversation.get("answer") or "").strip()
    if not 3 <= len(user) <= 2500 or not 2 <= len(answer) <= 5000:
        return None, "length"
    if "<tool_call>" in answer or "scenario" in answer.lower() or NOTE_HEAD in answer:
        return None, "leak"
    facts = [fact for fact in conversation.get("facts") or [] if isinstance(fact, str) and fact.strip()][:3]
    if seed["kind"] != "facts":
        facts = []
    guidance = GUIDANCE.get(seed["skill"], "") if rng.random() < 0.4 else ""
    earlier = str(conversation.get("earlier") or "").strip() if seed["kind"] == "long" else ""
    if seed["kind"] == "long" and not 30 <= len(earlier) <= 900:
        return None, "no earlier record"
    messages: list[dict] = []
    system = []
    if seed["folder"]:
        system.append(f"Your file and command tools work inside the folder `{seed['folder']}`.")
    if earlier:
        system.append("## Earlier conversation summary\n" + earlier)
    if system:
        messages.append({"role": "system", "content": "\n\n".join(system)})
    offnote = seed["kind"] == "offnote"
    messages.append({"role": "user", "content": with_note(user, seed["time"], facts, seed["exact"],
                                                          "" if offnote else guidance, offnote, bool(earlier))})
    used: list[str] = []
    steps = conversation.get("steps") or []
    if seed["kind"] == "pre" and (len(steps) != 1 or steps[0]["calls"][0].get("name") != "web_search"):
        return None, "presearch shape"
    for step in steps:
        calls, results = step.get("calls") or [], step.get("results") or []
        if not calls or len(calls) != len(results):
            return None, "calls and results differ"
        for call in calls:
            name, arguments = call.get("name"), call.get("arguments")
            if name not in catalog or not isinstance(arguments, dict):
                return None, "unknown tool"
            schema = catalog[name]["function"].get("parameters") or {}
            if any(key not in arguments for key in schema.get("required") or []):
                return None, "missing argument"
            if any(key not in (schema.get("properties") or {}) for key in arguments):
                return None, "unknown argument"
            if (name == "file_manage" and arguments.get("action") == "delete") or DELETES.search(call_commands(call)):
                return None, "deletes"
            if name == "component" and arguments.get("name") and arguments["name"] not in {c[0] for c in COMPONENTS}:
                return None, "unknown component"
            if offnote and name in WEB_TOOLS:
                return None, "web call while known to be offline"
            used.append(name)
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for call, result in zip(calls, results):
            if not isinstance(result, str) or not result.strip() or len(result) > 3000:
                return None, "result"
            if seed["kind"] == "offline" and call["name"] in WEB_TOOLS:
                result = OFFLINE_ERROR  # word for word what Kiwibro's web tools return without a connection
            messages.append({"role": "tool", "content": result.strip()})
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
        messages.append({"role": "user", "content": with_note(follow_user, seed["time"])})
        messages.append({"role": "assistant", "content": follow_answer})
    everything = list(catalog)
    if rng.random() < 0.6:
        tools = everything
    else:
        extra = rng.sample(everything, rng.randrange(8, 26))
        tools = [name for name in everything if name in set(extra) | set(used) | set(SKILL_TOOLS[seed["skill"]][:4])]
    identity = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": identity, "skill": seed["skill"], "kind": seed["kind"], "hard": seed["hard"], "followup": followup,
            "tools": tools, "messages": messages, "split": "held" if int(identity[:4], 16) % 16 == 0 else "train"}, ""


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


def stage_gen(args) -> None:
    from vllm import LLM, SamplingParams

    catalog = load_catalog(Path(args.tools))
    examples: dict[str, list] = {}
    if args.examples and Path(args.examples).exists():
        for item in json.loads(Path(args.examples).read_text()):
            examples.setdefault(item["tool"], []).append((item["arguments"], item["result"]))
    prefixes = {skill: teacher_prefix(skill, catalog, examples) for skill in SKILLS}
    out = Path(args.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        seen = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    log(f"gen: {len(seen)} conversations already there; target {args.target}; {args.minutes} min")
    llm = LLM(model=TEACHER, max_model_len=16384, gpu_memory_utilization=args.gpu, max_num_seqs=args.seqs,
              enable_prefix_caching=True, seed=SEED)
    rng = random.Random(SEED + len(seen))
    mode = args.schema

    def params(schema: dict | None):
        if schema is None:
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens)
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens,
                                  structured_outputs=StructuredOutputsParams(json=schema))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens,
                                  guided_decoding=GuidedDecodingParams(json=schema))

    wanted = [skill for skill in SKILLS if not args.only or skill in args.only.split(",")]
    # conversations with tool steps are long: fewer of them per reply, so a reply ends before its limit
    per_prompt = {skill: args.per_prompt if skill in ("chat", "write") else max(2, args.per_prompt - 1) for skill in SKILLS}
    deadline = time.time() + args.minutes * 60
    kept, chunk, last = len(seen), 0, 0.0
    counts = {skill: 0 for skill in SKILLS}
    per_skill = args.first
    while kept < args.target:
        if chunk:  # size each further chunk to the time that is left: a chunk only ends with its longest reply
            left = deadline - time.time()
            per_skill = min(args.chunk, int(per_skill * left / max(last, 1.0) * 0.85))
            if left < 90 or per_skill < 3:
                break
        jobs = []
        for skill in wanted:
            for _ in range(per_skill):
                jobs.append((skill, [make_seed(skill, rng) for _ in range(per_prompt[skill])]))
        rng.shuffle(jobs)
        started = time.time()
        while True:
            schemas = {skill: (None if mode == "none" else conversation_schema(skill, catalog, per_prompt[skill], mode))
                       for skill in wanted}
            talks = [[{"role": "system", "content": TEACHER_SYSTEM},
                      {"role": "user", "content": teacher_prompt(prefixes[skill], seeds)}] for skill, seeds in jobs]
            try:
                outputs = llm.chat(talks, [params(schemas[skill]) for skill, _ in jobs], use_tqdm=False,
                                   chat_template_kwargs={"enable_thinking": False})
                break
            except Exception as error:  # a schema the grammar compiler refuses: fall back one level
                log(f"gen: schema mode {mode} failed: {type(error).__name__}: {str(error)[:300]}")
                if mode == "none":
                    raise
                mode = "loose" if mode == "strict" else "none"
        last = time.time() - started
        tokens, refused, wrote, shown = 0, {}, 0, 0
        with out.open("a") as handle:
            for (skill, seeds), output in zip(jobs, outputs):
                tokens += len(output.outputs[0].token_ids)
                text = output.outputs[0].text
                conversations = read_conversations(text)
                if len(conversations) < len(seeds):
                    refused["cut short"] = refused.get("cut short", 0) + len(seeds) - len(conversations)
                    if shown < 2:
                        shown += 1
                        log(f"gen: {skill} output cut short after {len(conversations)} of {len(seeds)}, "
                            f"{len(output.outputs[0].token_ids)} tokens; it ends: {text[-260:]!r}")
                for conversation, seed in zip(conversations, seeds):
                    record, why = build_record(conversation, seed, catalog, rng) if isinstance(conversation, dict) else (None, "shape")
                    if record is None or record["id"] in seen:
                        refused[why or "repeat"] = refused.get(why or "repeat", 0) + 1
                        continue
                    seen.add(record["id"])
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[skill] += 1
                    wrote += 1
        kept += wrote
        log(f"gen: chunk {chunk} mode {mode}: {len(jobs)} prompts, {wrote} kept, refused {refused}, "
            f"{tokens / max(last, 1e-6):.0f} tok/s, {last:.0f}s; total {kept}; per skill {counts}")
        if chunk == 0:
            sample = json.loads(out.read_text().splitlines()[-1])
            log("gen: sample\n" + RENDER.render(sample["messages"], [catalog[n] for n in sample["tools"][:6]],
                                               add_generation_prompt=False)[:2500])
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


def encode(tokenizer, record: dict, catalog: dict):
    """Token ids, which of them the model must learn to write, and where each assistant turn starts."""
    text, spans = "", []
    for role, body in RENDER.turns(record["messages"], shown_tools(record, catalog)):
        text += f"{RENDER.IM_START}{role}\n"
        start = len(text)
        text += body + RENDER.IM_END
        if role == "assistant":
            spans.append((start, len(text)))
        text += "\n"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, mask, starts, at = encoded["input_ids"], [], [], 0
    for index, (begin, end) in enumerate(encoded["offset_mapping"]):
        while at < len(spans) and begin >= spans[at][1]:
            at += 1
        inside = at < len(spans) and begin >= spans[at][0] and end <= spans[at][1] and end > begin
        if inside and (not starts or starts[-1][1] != at):
            starts.append((index - 1, at))
        mask.append(1 if inside else 0)
    return ids, mask, [position for position, _ in starts]


def load_data(args, tokenizer) -> list[dict]:
    import torch

    cache = Path(args.out) / "encoded.pt"
    if cache.exists() and not args.fresh:
        return torch.load(cache)
    catalog = load_catalog(Path(args.tools))
    rows, dropped = [], 0
    for line in Path(args.data).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ids, mask, starts = encode(tokenizer, record, catalog)
        if len(ids) > MAX_TOKENS or not sum(mask):
            dropped += 1
            continue
        hard = [bool(record["hard"])] * len(starts)
        if record.get("followup") and hard:
            hard[-1] = False
        rows.append({"id": record["id"], "ids": ids, "mask": mask, "starts": starts, "hard": hard,
                     "skill": SKILLS.index(record["skill"]), "split": record["split"]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache)
    lengths = sorted(len(row["ids"]) for row in rows)
    log(f"data: {len(rows)} conversations ({dropped} dropped), tokens median {lengths[len(lengths) // 2]}, "
        f"max {lengths[-1]}, total {sum(lengths)}; held {sum(row['split'] == 'held' for row in rows)}")
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


def collate(rows: list[dict], indexes: list[int], pad: int, device):
    import torch

    width = max(len(rows[i]["ids"]) for i in indexes)
    ids = torch.full((len(indexes), width), pad, dtype=torch.long)
    attention = torch.zeros((len(indexes), width), dtype=torch.long)
    mask = torch.zeros((len(indexes), width), dtype=torch.bool)
    for row, index in enumerate(indexes):
        size = len(rows[index]["ids"])
        ids[row, :size] = torch.tensor(rows[index]["ids"])
        attention[row, :size] = 1
        mask[row, :size] = torch.tensor(rows[index]["mask"], dtype=torch.bool)
    return ids.to(device), attention.to(device), mask.to(device)


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
        """A frozen linear layer plus one skill's low-rank change: y = W x + scale * B (A x)."""

        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            out_dim, in_dim = linear.weight.shape
            self.a = nn.Parameter(torch.empty(RANK, in_dim))
            self.b = nn.Parameter(torch.zeros(out_dim, RANK))
            self.on = True
            self.reset()

        def reset(self):
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
            nn.init.zeros_(self.b)

        def forward(self, x):
            y = self.linear(x)
            if self.on:
                y = y + F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * (ALPHA / RANK)
            return y

    return QLinear, QEmbedding, SkillLinear


def loss_on(model, table, ids, attention, mask):
    """Cross-entropy on the tokens the assistant writes; table is the output matrix to use."""
    import torch
    import torch.nn.functional as F

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
        wanted = mask[:, 1:]
        logits = F.linear(hidden[:, :-1][wanted], table.to(hidden.dtype))
    return F.cross_entropy(logits.float(), ids[:, 1:][wanted]), int(wanted.sum())


def held_loss(model, table, rows: list[dict], pad: int, budget: int) -> float:
    import torch

    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for indexes in batches(rows, budget, None):
            loss, size = loss_on(model, table() if callable(table) else table, *collate(rows, indexes, pad, "cuda"))
            total, count = total + float(loss) * size, count + size
    model.train()
    return total / max(count, 1)


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
    base_loss = held_loss(model, table_weight, held, pad, args.budget)
    for _, _, holder, name in linears(model):
        setattr(holder, name, QLinear(getattr(holder, name)))
    model.model.embed_tokens = QEmbedding(model.model.embed_tokens, bits)
    table = lambda: fake_quant(table_weight, bits)  # noqa: E731
    start_loss = held_loss(model, table, held, pad, args.budget)
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
        loss, size = loss_on(model, table(), *collate(train, indexes, pad, "cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += sum(len(train[i]["ids"]) for i in indexes)
        if step % 25 == 0 or step == len(plan) - 1:
            log(f"trunk: step {step}/{len(plan)} loss {float(loss.detach()):.4f} {seen / (time.time() - began):.0f} tok/s")
        if (step + 1) % every == 0 or step == len(plan) - 1:
            now = held_loss(model, table, held, pad, args.budget)
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
    final = held_loss(model, lambda: table_weight, held, pad, args.budget)
    (out / "trunk.json").write_text(json.dumps({"held_loss_float_base": base_loss, "held_loss_grid_untrained": start_loss,
                                                "held_loss_trunk": final, "emb": args.emb, "steps": len(plan),
                                                "kept_step": best_step or len(plan), "conversations": len(train)}, indent=1))
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


def put_skill(modules: dict, state: dict | None) -> None:
    import torch

    for (index, matrix), module in modules.items():
        module.on = state is not None
        if state is not None:
            with torch.no_grad():
                module.a.copy_(state[f"{index}.{matrix}.a"].float())
                module.b.copy_(state[f"{index}.{matrix}.b"].float())


def stage_skills(args) -> None:
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
    parameters = [p for module in modules.values() for p in (module.a, module.b)]
    (out / "skills").mkdir(exist_ok=True)
    report = {}
    for number, skill in enumerate(SKILLS):
        if args.only and skill not in args.only.split(","):
            continue
        train = [row for row in rows if row["split"] == "train" and row["skill"] == number
                 and (not args.skill_share or reserved(row, args.skill_share))]
        held = [row for row in rows if row["split"] == "held" and row["skill"] == number]
        for module in modules.values():
            module.reset()
            module.on = False
        before = held_loss(model, table, held, pad, args.budget)
        for module in modules.values():
            module.on = True
        optimizer = torch.optim.AdamW(parameters, lr=args.skill_lr, betas=(0.9, 0.99), weight_decay=0.0)
        rng = random.Random(SEED + number)
        plan = [b for _ in range(args.skill_epochs) for b in batches(train, args.budget, rng)]
        if args.smoke:
            plan = plan[:3]
        model.train()
        step, mark = 0, 0.0
        # The skill starts as no change at all. It is kept as it was at its lowest held-back loss, and
        # one that never gets below the trunk alone stays empty: a skill may help, it must not harm.
        best_loss, best_state, best_step = before, None, 0
        while step < len(plan):
            indexes = plan[step]
            for group in optimizer.param_groups:
                group["lr"] = schedule(step, len(plan), args.skill_lr)
            loss, _ = loss_on(model, table, *collate(train, indexes, pad, "cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            plan, mark = fit_plan(plan, step, mark, args.skill_minutes * 60 / len(SKILLS), f"skills: {skill}")
            step += 1
            if args.skill_check and (step % args.skill_check == 0 or step >= len(plan)):
                now = held_loss(model, table, held, pad, args.budget)
                if now < best_loss:
                    best_loss, best_step = now, step
                    best_state = [(module.a.detach().clone(), module.b.detach().clone()) for module in modules.values()]
        if args.skill_check:
            with torch.no_grad():
                if best_state is None:
                    for module in modules.values():
                        module.b.zero_()  # no change: the trunk alone was better at every look
                else:
                    for module, (a, b) in zip(modules.values(), best_state):
                        module.a.copy_(a)
                        module.b.copy_(b)
        after = held_loss(model, table, held, pad, args.budget)
        state = {}
        for (index, matrix), module in modules.items():
            state[f"{index}.{matrix}.a"] = module.a.detach().cpu().half()
            state[f"{index}.{matrix}.b"] = module.b.detach().cpu().half()
        torch.save(state, out / "skills" / f"{skill}.pt")
        report[skill] = {"conversations": len(train), "steps": len(plan), "kept_step": best_step if args.skill_check else len(plan),
                         "held_loss_trunk": before, "held_loss_with_skill": after}
        log(f"skills: {skill}: {len(train)} conversations, {len(plan)} steps, kept step {report[skill]['kept_step']}, "
            f"held loss {before:.4f} -> {after:.4f}")
    (out / "skills.json").write_text(json.dumps(report, indent=1))


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


def stage_head(args) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    out = Path(args.out)
    tokenizer, model = load_model(state=out / "trunk.pt")
    model.eval()
    rows = load_data(args, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rng = random.Random(SEED)
    # a second copy of some conversations with another one in front, so the head also sees a
    # change of subject in the middle of a chat
    extra = []
    short = [row for row in rows if row["split"] == "train" and len(row["ids"]) < 1500]
    for row in rows:
        if row["split"] != "train" or not short or rng.random() > 0.4:
            continue
        front = rng.choice(short)  # the joined text keeps the later conversation's system turn
        first_user =next((i for i, token in enumerate(front["ids"]) if i > 0 and token == front["ids"][0]), 0)
        tail_start = next((i for i, token in enumerate(row["ids"]) if i > 0 and token == row["ids"][0]), 0)
        if not first_user or not tail_start:
            continue
        ids = row["ids"][:tail_start] + front["ids"][first_user:] + row["ids"][tail_start:]
        if len(ids) > MAX_TOKENS:
            continue
        shift = len(front["ids"]) - first_user
        extra.append({"ids": ids, "mask": [0] * len(ids), "starts": [p + shift for p in row["starts"]],
                      "hard": row["hard"], "skill": row["skill"], "split": "train"})
    everything = rows + extra
    features, skills, hards, splits, firsts = [], [], [], [], []
    for indexes in batches(everything, args.budget * 2, None):
        ids, attention, _ = collate(everything, indexes, pad, "cuda")
        states = split_states(model, ids, attention)
        for row, index in enumerate(indexes):
            item = everything[index]
            for turn, position in enumerate(item["starts"]):
                features.append(states[row, position].cpu())
                skills.append(item["skill"])
                hards.append(float(item["hard"][turn]))
                splits.append(item["split"] == "held")
                firsts.append(turn == 0)
    x = torch.stack(features)
    x = x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6)
    y, hard = torch.tensor(skills), torch.tensor(hards)
    held, first = torch.tensor(splits), torch.tensor(firsts)
    log(f"head: {int((~held).sum())} training states, {int(held.sum())} held ({len(extra)} joined conversations)")
    net = nn.Sequential(nn.Linear(x.shape[1], args.head_hidden), nn.ReLU(), nn.Linear(args.head_hidden, len(HEAD_NAMES))).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    tx, ty, th = x[~held].cuda(), y[~held].cuda(), hard[~held].cuda()
    for epoch in range(2 if args.smoke else 40):
        order = torch.randperm(tx.shape[0], device="cuda")
        for begin in range(0, len(order), 256):
            pick = order[begin:begin + 256]
            got = net(F.dropout(tx[pick], 0.1))
            loss = F.cross_entropy(got[:, :len(SKILLS)], ty[pick]) + F.binary_cross_entropy_with_logits(got[:, -1], th[pick])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    net.eval()
    with torch.no_grad():
        got = net(x[held].cuda()).cpu()
    guess = got[:, :len(SKILLS)].argmax(dim=-1)
    right = guess == y[held]
    said_hard, is_hard = got[:, -1] > 0, hard[held] > 0.5
    report = {"states_train": int((~held).sum()), "states_held": int(held.sum()),
              "skill_accuracy": float(right.float().mean()),
              "skill_accuracy_first_turn": float(right[first[held]].float().mean()),
              "per_skill": {skill: float(right[y[held] == number].float().mean()) for number, skill in enumerate(SKILLS)
                            if int((y[held] == number).sum())},
              "hard_precision": float((said_hard & is_hard).sum() / max(1, int(said_hard.sum()))),
              "hard_recall": float((said_hard & is_hard).sum() / max(1, int(is_hard.sum()))),
              "hard_share": float(is_hard.float().mean())}
    torch.save({"w1": net[0].weight.detach().cpu(), "b1": net[0].bias.detach().cpu(), "w2": net[2].weight.detach().cpu(),
                "b2": net[2].bias.detach().cpu(), "names": HEAD_NAMES, "layer": SPLIT, "eps": 1e-6}, out / "head.pt")
    (out / "head.json").write_text(json.dumps(report, indent=1))
    log("head: " + json.dumps(report))


# ------------------------------------------------------------------ stage: eval

def first_turn(record: dict, catalog: dict) -> tuple[str, str]:
    """The prompt up to the first thing the assistant writes, and the tool it should call ("" = answer)."""
    messages, kept = record["messages"], []
    for message in messages:
        if message["role"] == "assistant":
            calls = message.get("tool_calls") or []
            return (RENDER.render(kept, list(catalog.values())),
                    calls[0]["function"]["name"] if calls else "")
        kept.append(message)
    return "", ""


def tool_scores(model, tokenizer, cases: list[tuple[str, str]], catalog: dict) -> dict:
    import torch

    tokenizer.padding_side = "left"
    stops = [tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "</tool_call>", "<|endoftext|>")]
    decided = named = valid = wanted_tool = 0
    for begin in range(0, len(cases), 16):
        part = cases[begin:begin + 16]
        encoded = tokenizer([prompt for prompt, _ in part], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            made = model.generate(**encoded, max_new_tokens=96, do_sample=False, eos_token_id=stops,
                                  pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for (_, gold), row in zip(part, made[:, encoded["input_ids"].shape[1]:]):
            text = tokenizer.decode(row, skip_special_tokens=False)
            for token in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(token, "")
            _, calls = RENDER.parse_reply(text)
            name = calls[0]["name"] if calls else ""
            decided += bool(name) == bool(gold)
            if gold:
                wanted_tool += 1
                named += name == gold
                schema = catalog.get(name, {}).get("function", {}).get("parameters") or {}
                valid += bool(name in catalog and all(key in calls[0]["arguments"] for key in schema.get("required") or []))
    return {"cases": len(cases), "tool_or_answer_right": decided / max(1, len(cases)),
            "tool_name_right": named / max(1, wanted_tool), "call_valid": valid / max(1, wanted_tool),
            "cases_wanting_a_tool": wanted_tool}


def stage_eval(args) -> None:
    import torch

    out = Path(args.out)
    catalog = load_catalog(Path(args.tools))
    records = {}
    for line in Path(args.data).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            if record["split"] == "held":
                records[record["id"]] = record
    tokenizer, model = load_model()
    rows = [row for row in load_data(args, tokenizer) if row["split"] == "held"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    per = 4 if args.smoke else args.cases
    cases = {skill: [first_turn(records[row["id"]], catalog) for row in rows if row["skill"] == number][:per]
             for number, skill in enumerate(SKILLS)}
    by_skill = {skill: [row for row in rows if row["skill"] == number] for number, skill in enumerate(SKILLS)}
    report = {"held_conversations": len(rows), "loss": {}, "tools": {}}
    report["loss"]["float base"] = {skill: held_loss(model, model.model.embed_tokens.weight, by_skill[skill], pad, args.budget)
                                    for skill in SKILLS if by_skill[skill]}
    report["tools"]["float base"] = {skill: tool_scores(model, tokenizer, cases[skill], catalog) for skill in SKILLS if cases[skill]}
    del model
    torch.cuda.empty_cache()
    tokenizer, model = load_model(state=out / "trunk.pt")
    modules = attach_skills(model)
    model.cuda().eval()
    table = model.model.embed_tokens.weight
    put_skill(modules, None)
    report["loss"]["trunk"] = {skill: held_loss(model, table, by_skill[skill], pad, args.budget) for skill in SKILLS if by_skill[skill]}
    report["tools"]["trunk"] = {skill: tool_scores(model, tokenizer, cases[skill], catalog) for skill in SKILLS if cases[skill]}
    report["tools"]["trunk + its skill"] = {}
    for skill in SKILLS:
        path = out / "skills" / f"{skill}.pt"
        if not path.exists():
            continue
        put_skill(modules, torch.load(path))
        report["loss"][f"trunk + {skill}"] = {other: held_loss(model, table, by_skill[other], pad, args.budget)
                                              for other in SKILLS if by_skill[other]}
        if cases[skill]:
            report["tools"]["trunk + its skill"][skill] = tool_scores(model, tokenizer, cases[skill], catalog)
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
                "files": files}
    for name in ("trunk", "skills", "head", "eval"):
        if (out / f"{name}.json").exists():
            manifest[name] = json.loads((out / f"{name}.json").read_text())
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log("export: " + json.dumps({name: info["bytes"] for name, info in files.items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gen", "trunk", "skills", "head", "eval", "export"])
    parser.add_argument("--data", default=str(WORK / "data" / "conv.jsonl"))
    parser.add_argument("--out", default=str(WORK / "out"))
    parser.add_argument("--tools", default=str(WORK / "tools_full.json"))
    parser.add_argument("--examples", default=str(WORK / "result_examples.json"))
    parser.add_argument("--smoke", action="store_true", help="a few steps of everything, to find mistakes early")
    parser.add_argument("--fresh", action="store_true", help="encode the conversations again")
    parser.add_argument("--target", type=int, default=7000)
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
    parser.add_argument("--skill-check", type=int, default=4,
                        help="skills: look at the held-back loss every this many steps and keep the best state; 0 = keep the last")
    parser.add_argument("--skill-share", type=float, default=0.3,
                        help="share of the training conversations the trunk never reads; each skill learns from its own part of it")
    parser.add_argument("--only", default="", help="skills: train just these, comma separated")
    parser.add_argument("--head-hidden", type=int, default=192)
    parser.add_argument("--cases", type=int, default=48, help="eval: generated first turns per skill")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.budget = min(args.budget, 1500)  # small enough to run beside the teacher
    {"gen": stage_gen, "trunk": stage_trunk, "skills": stage_skills, "head": stage_head, "eval": stage_eval,
     "export": stage_export}[args.stage](args)


if __name__ == "__main__":
    main()
