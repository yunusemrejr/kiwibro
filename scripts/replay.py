#!/usr/bin/env python3
"""Replay scripted conversations against a running Kiwibro and save every reply.

The standing check for answer quality: run it before and after a change, several times
(small models are random), and read the replies side by side.

    KIWIBRO_DATA_DIR=/tmp/replay-data KIWIBRO_WORKSPACE=/tmp/replay-ws KIWIBRO_PORT=8792 \\
        KIWIBRO_AUTO_UPDATE=0 python3 app.py &
    python3 scripts/replay.py http://localhost:8792 <model> tests/replay/scenarios.json out.json [only,names] [--helpers-off]

Every run starts by deleting the instance's memories and notes, so one run cannot answer from
what the run before it stored. It therefore refuses to touch an instance that uses the default
data folder: always point it at a test instance with its own.

The "repo" scenario asks about a checkout named demo-app in the test workspace: clone any
repository there first (git clone <address> /tmp/replay-ws/demo-app).
"""

import json
import sys
import time
from pathlib import Path

import requests

helpers_off = "--helpers-off" in sys.argv
args = [arg for arg in sys.argv[1:] if arg != "--helpers-off"]
base, model, scenario_file, out_file = args[:4]
only = set(args[4].split(",")) if len(args) > 4 else None
scenarios = json.load(open(scenario_file))
results = []

data_dir = Path(requests.get(f"{base}/api/settings", timeout=30).json().get("data_dir") or "")
if data_dir == Path.home() / ".local/share/kiwibro":
    sys.exit(f"{base} uses your own data folder ({data_dir}); start a test instance with KIWIBRO_DATA_DIR set.")
requests.delete(f"{base}/api/memories", timeout=30)
for note in requests.get(f"{base}/api/notes", timeout=30).json().get("notes", []):
    requests.delete(f"{base}/api/notes/{note['id']}", timeout=30)
requests.patch(f"{base}/api/settings", json={"sense_enabled": not helpers_off}, timeout=30)


def chat(conversation_id, message):
    body = {"conversation_id": conversation_id, "message": message, "model": model}
    started = time.time()
    turn = {"user": message, "text": "", "tools": [], "events": []}
    with requests.post(f"{base}/api/chat", json=body, stream=True, timeout=600) as response:
        response.raise_for_status()
        event = None
        for raw in response.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            if raw.startswith("event:"):
                event = raw[6:].strip()
            elif raw.startswith("data:") and event:
                try:
                    data = json.loads(raw[5:].strip())
                except ValueError:
                    continue
                if event == "meta":
                    conversation_id = data.get("conversation_id") or conversation_id
                elif event == "delta":
                    turn["text"] += data.get("text", "")
                elif event == "rewrite":
                    turn["text"] = data.get("text", "")
                    turn["events"].append("rewrite")
                elif event == "retract":
                    turn["text"] = turn["text"].replace(data.get("text", ""), "", 1)
                elif event == "tool_call":
                    turn["tools"].append({"name": data.get("name"), "args": data.get("args")})
                elif event == "tool_result":
                    if turn["tools"]:
                        turn["tools"][-1]["ok"] = data.get("ok")
                        turn["tools"][-1]["preview"] = (data.get("preview") or "")[:300]
                elif event == "approval":
                    requests.post(f"{base}/api/approvals/{data['approval_id']}", json={"decision": "allow"}, timeout=30)
                    turn["events"].append("approval:" + str(data.get("tool")))
                elif event == "memory":
                    turn["memory"] = data
                elif event == "plan":
                    turn["plan"] = data.get("steps")
                elif event == "done":
                    turn["inference"] = data.get("inference")
                    turn["stats"] = data.get("stats")
                    break
                elif event == "error":
                    turn["error"] = data
                    break
    turn["seconds"] = round(time.time() - started, 1)
    return conversation_id, turn


for scenario in scenarios:
    if only and scenario["name"] not in only:
        continue
    conversation_id = None
    record = {"name": scenario["name"], "turns": []}
    for message in scenario["messages"]:
        try:
            conversation_id, turn = chat(conversation_id, message)
        except Exception as exc:  # keep going: one failed turn must not lose the run
            turn = {"user": message, "error": repr(exc)}
        record["turns"].append(turn)
        print(f"[{scenario['name']}] {message[:60]!r} -> {len(turn.get('text', ''))} chars, "
              f"{[t['name'] for t in turn.get('tools', [])]} {turn.get('seconds')}s", flush=True)
        json.dump(results + [record], open(out_file, "w"), ensure_ascii=False, indent=1)
    record["conversation_id"] = conversation_id
    results.append(record)
    json.dump(results, open(out_file, "w"), ensure_ascii=False, indent=1)
try:
    record_memories = requests.get(f"{base}/api/memories", timeout=30).json()
    record_notes = requests.get(f"{base}/api/notes", timeout=30).json()
    json.dump({"scenarios": results, "memories": record_memories, "notes": record_notes}, open(out_file, "w"),
              ensure_ascii=False, indent=1)
except Exception as exc:
    print("could not read memories/notes", exc)
requests.patch(f"{base}/api/settings", json={"sense_enabled": True}, timeout=30)
print("done")
