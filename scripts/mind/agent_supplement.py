#!/usr/bin/env python3
"""Build a training-only KiwiBro supplement from recurring failure categories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from scripts.mind import agent_corpus as corpus


DEFAULT_SEED = 91309
NAMES = ("Breakwater", "Saltpan", "Wayfarer", "Copperleaf")
HELPERS = ("model_info", "delegate", "calculate")
WEB = ("web_search", "fetch_page")
FILES = ("remember", "write_file", "read_file")
REPAIR = ("read_file", "apply_patch")
DATA = ("analyze_table", "read_file", "calculate")


def teach(episode: corpus.Episode, *reasoning: str) -> corpus.Episode:
    targets = [message for message in episode.messages if message["role"] == "assistant"]
    if len(targets) != len(reasoning):
        raise ValueError(f"{episode.id}: missing decision rationale")
    for message, decision in zip(targets, reasoning, strict=True):
        message["reasoning"] = decision
    if not corpus.score({"calls": [], "required": episode.required, "forbidden": episode.forbidden}, targets[-1]):
        raise ValueError(f"{episode.id}: answer disagrees with its authored oracle")
    return episode


def helper_unavailable(s: corpus.Sample, _out: Path) -> corpus.Episode:
    primary = f"qa/{s.slug}-layout"
    name = f"{s.label} dock"
    e = corpus.Episode(s.id, "supplement_helper", f"Send only this label to {primary} for a capitalization "
                       f"check: “{name}”. That reviewer alone is approved to see the wording.", HELPERS)
    e.observe("model_info", "Installed local models:\nqa/terse-editor · coding\nqa/quick-text · general")
    e.answer(f"{primary} is unavailable. I did not send the label to another reviewer.",
             (primary, "unavailable", "did not send"), ("qa/quick-text checked", "looks correct"))
    return teach(e, "The wording can go only to the exact reviewer, so check the local list first.",
                 "The requested reviewer is absent and the user allowed no other recipient. Report that no check ran.")


def helper_authorized(s: corpus.Sample, _out: Path) -> corpus.Episode:
    primary = f"lab/{s.slug}-proof"
    backup = f"lab/{s.slug}-second"
    subject = f"The {s.label} repairs crew"
    question = f"Identify the subject of the sentence “{subject} signed last.”"
    e = corpus.Episode(s.id, "supplement_helper", f"Ask {primary} this one grammar question. If it is unavailable, "
                       f"the only approved substitute is {backup}. Return the reviewer and its answer. {question}", HELPERS)
    e.observe("delegate", f"Error: {primary} is not available. No task was sent.",
              model=primary, capability="general", task=question, max_tokens=80)
    e.observe("delegate", f"{backup}: The subject is “{subject}.”",
              model=backup, capability="general", task=question, max_tokens=80)
    e.answer(f"{primary} was unavailable. {backup} identified “{subject}” as the subject.",
             (primary, backup, subject, "unavailable"), ("I used an automatic helper",))
    return teach(e, "Send the bounded grammar question to the selected reviewer first.",
                 "The first reviewer is unavailable and the request names exactly one fallback.",
                 "The approved fallback returned the subject. Name the reviewer that actually answered.")


def helper_local_fallback(s: corpus.Sample, _out: Path) -> corpus.Episode:
    from kiwibro.compute import evaluate

    model = f"budget/{s.slug}-checker"
    expression = f"{s.value} * 14"
    total = s.value * 14
    e = corpus.Episode(s.id, "supplement_helper", f"Ask your local helper {model} to total {s.value} boxes at "
                       "14 credits each. If that exact helper is missing, use your calculator. "
                       "Please keep the calculation on this machine.", HELPERS)
    e.observe("model_info", "Installed local models:\nwarehouse/quick-count · general")
    e.observe("calculate", f"{expression} = {evaluate(expression)['result']}", expression=expression)
    e.answer(f"{model} is unavailable. The local calculator gives {total} credits for {s.value} boxes.",
             (model, "unavailable", f"{total} credits"), ("warehouse/quick-count answered",))
    return teach(e, "Check whether the requested model is present before routing this side question.",
                 "The selected helper is absent. The user explicitly authorized local calculation as the fallback.",
                 "The calculator returned the total. Report that route and its result.")


def research_primary(s: corpus.Sample, _out: Path) -> corpus.Episode:
    permit = f"https://permits.{s.slug}.test/market/stall-rates"
    roster = f"https://permits.{s.slug}.test/market/vendors"
    e = corpus.Episode(s.id, "supplement_research", f"Link the {s.label} market authority's stall deposit and tell me "
                       "whether the vendor roster confirms that it is refunded after cleanup.", WEB)
    e.observe("web_search", f"Market stall rates — {permit}\nLicensed vendor roster — {roster}",
              query=f"{s.label} market stall deposit official roster", max_results=3)
    e.observe_batch(
        ("fetch_page", {"url": permit, "max_chars": 1200}, f"Stall rates\n{permit}\nDeposit: {s.value} credits."),
        ("fetch_page", {"url": roster, "max_chars": 1200},
         f"Licensed vendor roster\n{roster}\nDirectory of stall operators and categories. No refund terms."),
    )
    e.answer(f"The authority lists a deposit of {s.value} credits on its [stall rates page]({permit}). "
             f"The [vendor roster]({roster}) gives no refund terms, so the cleanup refund is unconfirmed.",
             (f"{s.value} credits", permit, roster, "unconfirmed"), ("will be refunded", "nonrefundable"))
    return teach(e, "Find the authority's fee page and the separate source the user named for refund terms.",
                 "Search titles locate the sources. Fetch both pages to check what they actually establish.",
                 "The fee is explicit and the roster leaves the refund unanswered. Cite each source with its claim.")


def research_missing(s: corpus.Sample, _out: Path) -> corpus.Episode:
    index = f"https://library.{s.slug}.test/access/guide"
    contact = f"https://library.{s.slug}.test/access/help"
    e = corpus.Episode(s.id, "supplement_research", f"Can outside readers bring a laptop into the {s.label} "
                       f"map archive? I have its published guide at {index}. Link the policy and give the answer "
                       "only if it's documented.", WEB)
    e.observe("fetch_page", f"Reading-room access guide\n{index}\nOutside readers may register. "
              f"Bag checks apply. For equipment questions see {contact}.",
              url=index, max_chars=1600)
    e.observe("fetch_page", f"Reader assistance\n{contact}\nThe desk handles equipment questions by email. "
              "The page lists no permitted-device categories.", url=contact, max_chars=1600)
    e.answer(f"Outside readers can register under the [access guide]({index}). Laptop permission remains "
             f"unconfirmed. The [access help page]({contact}) directs equipment questions to the desk.",
             (index, contact, "Laptop permission remains unconfirmed"), ("laptops are allowed", "laptops are prohibited"))
    return teach(e, "Use the access policy URL the user supplied.",
                 "The guide links a specific equipment contact page. Check it for the missing permission.",
                 "The pages confirm registration and provide an equipment contact. Neither gives a laptop rule.")


def memory_artifact(s: corpus.Sample, out: Path) -> corpus.Episode:
    path = f"handover/{s.slug}-cadence.json"
    fact = f"For the {s.label} handover, the follow-up interval is {s.value} hours."
    content = json.dumps({"handover": s.label, "follow_up_hours": s.value, "dispatch_ref": s.marker}, indent=2) + "\n"
    e = corpus.Episode(s.id, "supplement_memory", f"Remember this maintenance rule: {fact} Also write "
                       f"{path} with handover name, follow_up_hours, and dispatch_ref {s.marker}. "
                       "Read back the file so I can use it for the shift handoff.", FILES)
    e.observe("remember", f"Saved project memory: {fact}", action="save", kind="project", content=fact)
    e.observe("write_file", f"Wrote {len(content.encode())} bytes to {path}.", path=path, content=content)
    e.observe("read_file", "\n".join(f"{i}: {line}" for i, line in enumerate(content.splitlines(), 1)), path=path)
    e.answer(f"Saved the {s.value}-hour follow-up rule to memory. "
             f"I wrote and checked the [handoff file]({path}), which contains dispatch reference {s.marker}.",
             ("memory", f"{s.value}-hour", path, s.marker))
    corpus.write_fixture(out, e, "json", {path: content})
    return teach(e, "Save the user-supplied handover rule as project memory.",
                 "The memory save succeeded. The handoff JSON is a separate, still-pending deliverable.",
                 "Read the saved file to verify the interval and reference.",
                 "Both surfaces are confirmed. Report the memory change and link the checked file.")


def memory_reference(s: corpus.Sample, out: Path) -> corpus.Episode:
    path = f"service/{s.slug}/receipt.txt"
    contents = f"Bench: {s.label}\nCalibration receipt: {s.marker}\nJob: Closed\n"
    fact = "User prefers maintenance summaries to include the receipt reference."
    e = corpus.Episode(s.id, "supplement_memory", f"For future maintenance summaries, remember to include the "
                       f"receipt reference. Also tell me the current calibration receipt in {path}; "
                       "the receipt file is read-only for this request.", FILES)
    e.observe("remember", f"Saved preference: {fact}", action="save", kind="preference", content=fact)
    e.observe("read_file", "\n".join(f"{i}: {line}" for i, line in enumerate(contents.splitlines(), 1)), path=path)
    e.answer(f"Saved your receipt-reference preference. The current calibration receipt is {s.marker} "
             f"on line 2 of [{path}]({path}).", ("Saved", s.marker, "line 2", path), ("I edited",))
    corpus.write_fixture(out, e, "reference", {path: contents})
    return teach(e, "Store the stable formatting preference. The receipt number belongs to this request.",
                 "The other part of the request is a read of the current receipt file.",
                 "The receipt comes from line 2. Report it alongside the separate preference save.")


def visual_stopping(s: corpus.Sample, out: Path) -> corpus.Episode:
    path = f"frontdesk/{s.slug}-status.html"
    before, after = 'querySelector("#expand-brief")', 'querySelector("#expand-briefing")'
    source = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{s.label} dock status</title>
<style>body{{font:1rem system-ui;margin:1rem}}main{{max-width:32rem}}button{{min-height:44px;font:inherit}}</style>
<main><h1>{s.label} dock status</h1>
<button id="expand-briefing" type="button" aria-expanded="false" aria-controls="brief">Show briefing</button>
<section id="brief" hidden><p>Today's briefing code is {s.marker}.</p></section>
</main>
<script>
const button = document.{after};
const brief = document.querySelector("#brief");
button.addEventListener("click", () => {{
    brief.hidden = !brief.hidden;
    button.setAttribute("aria-expanded", String(!brief.hidden));
}});
</script>
</html>
"""
    bad = source.replace(after, before)
    e = corpus.Episode(s.id, "supplement_visual", f"The Show briefing button in {path} leaves the dock code hidden. "
                       "My preview shows a null addEventListener error. Fix the behavior, preserve the rest of the page, "
                       "and finish once the saved preview is clean and fits a phone.", REPAIR)
    e.observe("read_file", "\n".join(f"{i}: {line}" for i, line in enumerate(bad.splitlines(), 1)), path=path)
    # The app runs visual verification after an edit. Present its observation without
    # supervising the model to issue the app's internal look_at_page call itself.
    e.observe("apply_patch", f"Applied 1 edit to {path}.\n\n-const button = document.{before};\n"
              f"+const button = document.{after};\n\nSaved-page preview at 1280x800 and 360x740. "
              "No script errors. It fits a phone screen.", path=path, edits=[{"old": before, "new": after}])
    e.answer(f"Fixed the button selector in [{path}]({path}). The saved preview passed the script check "
             "and fit the 360 px phone viewport.", (path, "selector", "360 px"), ("restyled", "rewrote"))
    corpus.write_fixture(out, e, "html", {path: source, "before.html": bad})
    return teach(e, "Read the markup and listener binding to find why the user's preview failed.",
                 "The selector has the wrong ID. Change that one literal while preserving the rest.",
                 "The app checked the saved page after the edit. The preview is clean and fits, so report completion.")


def physical_missing(s: corpus.Sample, out: Path) -> corpus.Episode:
    from kiwibro.compute import evaluate

    power_path = f"bench/{s.slug}/supply.txt"
    trial_path = f"bench/{s.slug}/trial.txt"
    minutes = 3 + (s.value % 8)
    volts = 24
    expression = f"{volts} * {minutes} / 60"
    factor = evaluate(expression)["result"]
    power = f"Unit: {s.label}\nDC output: {volts} V constant\n"
    trial = f"Elapsed: {minutes} min\nMean current: not logged\n"
    e = corpus.Episode(s.id, "supplement_physics", f"Use {power_path} and {trial_path} to compute the watt-hours "
                       "sent to the DC load. If a meter reading was missed, give what the observations support.",
                       ("read_file", "calculate"))
    e.observe_batch(
        ("read_file", {"path": power_path}, f"1: Unit: {s.label}\n2: DC output: {volts} V constant"),
        ("read_file", {"path": trial_path}, f"1: Elapsed: {minutes} min\n2: Mean current: not logged"),
    )
    e.observe("calculate", f"{expression} = {factor}", expression=expression)
    e.answer(f"At {volts} V for {minutes} minutes, E = V × I_avg × t_hours = {factor} × I_avg Wh, "
             "with I_avg in amperes. The mean current is missing, so the recorded inputs give no numerical energy.",
             ("I_avg", f"{factor} × I_avg Wh", "current is missing"))
    corpus.write_fixture(out, e, "physics", {power_path: power, trial_path: trial})
    return teach(e, "Read the supply and trial logs together to identify which quantities were measured.",
                 "Voltage and elapsed time are known. Evaluate their product in volt-hours, leaving current symbolic.",
                 "A numerical watt-hour result depends on the missing mean current. Give the coefficient and required input.")


def typed_table(s: corpus.Sample, out: Path) -> corpus.Episode:
    from kiwibro.sandbox import Workspace
    from kiwibro.tools import ToolContext, execute_tool, validate_args

    path = f"inbound/{s.slug}/weighbridge.tsv"
    rows = f"bay\tload_kg\nwest\t{s.value}\neast\t{2 * s.value}\nwest\t{3 * s.value}\nsouth\t{4 * s.value}\n"
    e = corpus.Episode(s.id, "supplement_typed", f"Re-run the {s.label} weighbridge report in {path}. "
                       "The saved filter is for the west bay and the old export used a mass_kg column. "
                       "Give the current total west load and how many rows matched.", DATA)
    corpus.write_fixture(out, e, "tsv", {path: rows})
    context = ToolContext(Workspace(out / e.fixture["folder"]), None, None, e.id)
    stale = {"path": path, "where": 'bay == "west"', "aggregate": "sum:mass_kg", "limit": 12}
    profile = {"path": path}
    corrected = {"path": path, "where": 'bay == "west"', "aggregate": "count,sum:load_kg", "limit": 12}
    results = [execute_tool("analyze_table", validate_args("analyze_table", args), context)
               for args in (stale, profile, corrected)]
    if (
        [result.ok for result in results] != [False, True, True]
        or results[-1].meta != {"rows": 4, "matched": 2}
        or f"| 2 | {4 * s.value} |" not in results[-1].content
    ):
        raise ValueError(f"{s.id}: the typed table recovery no longer matches the runtime")
    for args, result in zip((stale, profile, corrected), results, strict=True):
        e.observe("analyze_table", result.content, **args)
    e.answer(f"The west bay total is {4 * s.value} kg across 2 matched rows out of 4. "
             "The current export names the column load_kg.", (f"{4 * s.value} kg", "2 matched rows", "load_kg"))
    return teach(e, "Use the supplied bay filter and the saved field name for the requested sum.",
                 "The old aggregate failed on an unknown column. Profile the existing table before retrying.",
                 "The profile shows load_kg. Keep the west filter and request count and sum together.",
                 "The corrected report gives the total and selected row count. Report both with units.")


RECIPES = {
    "supp-exact-unavailable": helper_unavailable, "supp-named-fallback": helper_authorized,
    "supp-calculator-fallback": helper_local_fallback, "supp-source-coverage": research_primary,
    "supp-source-absence": research_missing, "supp-memory-artifact": memory_artifact,
    "supp-memory-reference": memory_reference, "supp-render-stop": visual_stopping,
    "supp-missing-current": physical_missing, "supp-table-schema": typed_table,
}


def check(out: Path) -> dict:
    summary = corpus.check(out, training_only=True)
    rows = corpus.read_jsonl(out / "train.jsonl")
    thinking = Counter()
    for row in rows:
        messages = [message for message in row["messages"] if message["role"] == "assistant"]
        if row.get("thinking") is True:
            if not all(isinstance(message.get("reasoning"), str) and 1 <= len(message["reasoning"].split()) <= 35
                       for message in messages):
                raise ValueError(f"{row['id']}: missing or lengthy decision rationale")
            thinking["episodes"] += 1
            thinking["assistant_turns"] += len(messages)
        elif "thinking" in row or any("reasoning" in message for message in messages):
            raise ValueError(f"{row['id']}: unexpected reasoning in a plain trajectory")
    summary["thinking"] = dict(thinking)
    return summary


def verify_render(out: Path) -> None:
    from kiwibro.pagecheck import BUDGET_MS, PHONE, Options, chrome_render, look
    from kiwibro.sandbox import Workspace
    from kiwibro.tools import ToolContext, execute_tool, validate_args

    fixture = next(row for row in corpus.read_jsonl(out / "fixtures.jsonl") if row["kind"] == "html")
    row = next(row for row in corpus.read_jsonl(out / "train.jsonl") if row["id"] == fixture["id"])
    path = next(name for name in fixture["files"] if name != "before.html")
    folder = out / fixture["folder"]
    patch = next(call["function"]["arguments"] for message in row["messages"] for call in message.get("tool_calls", [])
                 if call["function"]["name"] == "apply_patch")
    with tempfile.TemporaryDirectory(prefix="kiwi-supplement-") as temp:
        scratch = Path(temp)
        workspace = Workspace(scratch)
        workspace.write_file(path, (folder / "before.html").read_text(encoding="utf-8"))
        broken = look(workspace, path, Options(phone=True))
        if not any(finding.code == "script" for finding in broken.findings):
            raise RuntimeError(f"Expected startup script error: {broken.report()}")
        context = ToolContext(workspace, None, None, row["id"])
        applied = execute_tool("apply_patch", validate_args("apply_patch", patch), context)
        fixed = (scratch / path).read_text(encoding="utf-8")
        if not applied.ok or fixed != (folder / path).read_text(encoding="utf-8"):
            raise RuntimeError("The patch changed more than the authored selector")
        rendered = look(workspace, path, Options(phone=True))
        if rendered.findings or rendered.could_not or "it fits a phone screen" not in rendered.confirmed:
            raise RuntimeError(rendered.report())
        control = scratch / "control.html"
        control.write_text(fixed.replace("</html>", """
<script>
    const toggle = document.getElementById("expand-briefing");
    toggle.click();
    if (!document.getElementById("brief").hidden && toggle.getAttribute("aria-expanded") === "true") {
        document.body.append("Supplement control verified");
    }
</script>
</html>"""), encoding="utf-8")
        result = chrome_render(control, scratch / "control.png", *PHONE, BUDGET_MS)
        if "Supplement control verified" not in (result.report or {}).get("text", ""):
            raise RuntimeError(f"The repaired control did not reveal the briefing: {result.error}")


def build(out: Path, *, each: int = 4, seed: int = DEFAULT_SEED, live: bool = False) -> dict:
    from kiwibro.tools import tool_defs

    if any((out / name).exists() for name in ("heldout.jsonl", "probes.jsonl")):
        raise ValueError("Use a separate output directory for the training supplement")
    out.mkdir(parents=True, exist_ok=True)
    catalog = {tool["function"]["name"]: tool for tool in tool_defs()}
    episodes = corpus.samples(out, RECIPES, NAMES, each, seed)
    rows = []
    for index, episode in enumerate(episodes):
        row = episode.record(catalog)
        if index % each % 2 == 0:
            row["thinking"] = True
        else:
            for message in row["messages"]:
                message.pop("reasoning", None)
        rows.append(row)
    corpus.jsonl(out / "train.jsonl", rows)
    fixtures = [episode.fixture for episode in episodes if episode.fixture]
    corpus.jsonl(out / "fixtures.jsonl", fixtures)
    summary = check(out)
    summary["fixtures"] = dict(Counter(fixture["kind"] for fixture in fixtures))
    summary["seed"] = seed
    summary["sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ("train.jsonl", "fixtures.jsonl")
    }
    if live:
        verify_render(out)
        summary["browser_fixture_verified"] = True
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("build")
    create.add_argument("output", type=Path)
    create.add_argument("--each", type=int, default=4)
    create.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify = commands.add_parser("check")
    verify.add_argument("output", type=Path)
    for command in (create, verify):
        command.add_argument("--verify-render", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        if args.each < 1:
            parser.error("--each must be positive")
        summary = build(args.output, each=args.each, seed=args.seed, live=args.verify_render)
    else:
        summary = check(args.output)
        if args.verify_render:
            verify_render(args.output)
            summary["browser_fixture_verified"] = True
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
