#!/usr/bin/env python3
"""Build the training data for the Kiwi Sense helpers: real messages, labelled by a teacher model.

Runs on a GPU machine (it was run on Google Colab), never inside Kiwibro:

    python label.py pool      # collect English messages and message pairs from open datasets
    python label.py synth     # the teacher writes messages for the rare cases
    python label.py messages  # the teacher labels every message (intent, web, time, topic, keep, spans)
    python label.py facts     # memory facts written from messages that state them
    python label.py lite      # the teacher labels many more messages, without the spans (for the reading head)
    python label.py shift     # the teacher labels message pairs: follow-up, same topic or new topic
    python label.py threads   # the teacher writes five next messages for one real exchange: two follow-ups,
                              # one on the same subject, two on a new one
    python label.py relate    # the teacher labels (message, saved item) pairs: bears on the answer or not
    python label.py all

Every stage writes one JSONL file under $KS_DATA (default ./data) and is skipped when that
file exists, so a stopped run carries on where it was. Needs: vllm, datasets, scikit-learn.

Sources (all open): allenai/WildChat-1M (ODC-BY), OpenAssistant/oasst2 (Apache-2.0),
HuggingFaceH4/ultrachat_200k (MIT), databricks/databricks-dolly-15k (CC BY-SA 3.0),
mteb/amazon_massive_intent (Apache-2.0, from MASSIVE CC BY 4.0), clinc/clinc_oos (CC BY 3.0),
bavard/personachat_truecased (persona lines). The teacher is Qwen/Qwen3.8-27B-FP8.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("KS_DATA", "data"))
TEACHER = os.environ.get("KS_TEACHER", "/content/models/teacher")
SEED = 7
MAX_CHARS = 1200
# How much the teacher labels (one GPU-hour at about ten messages a second).
N_POOL = int(os.environ.get("KS_N_POOL", 7000))
N_SYNTH = int(os.environ.get("KS_N_SYNTH", 5000))
N_PAIRS = int(os.environ.get("KS_N_PAIRS", 7000))
N_LITE = int(os.environ.get("KS_N_LITE", 20000))
N_THREADS = int(os.environ.get("KS_N_THREADS", 6000))
N_QUERIES = int(os.environ.get("KS_N_QUERIES", 3500))
N_PERSONA = int(os.environ.get("KS_N_PERSONA", 1500))
N_TRIPLES = int(os.environ.get("KS_N_TRIPLES", 2500))

INTENTS = ["chat", "question", "task", "share", "remember", "note", "recall", "feedback", "continue", "meta"]
WEB = ["none", "own", "asked", "live", "lookup"]
TIME = ["none", "clock", "now", "recent", "past", "future"]
TOPICS = ["software", "computers", "ai", "science", "math", "health", "money", "work", "learning", "writing",
          "society", "entertainment", "sports", "places", "home", "people", "kiwibro", "none"]
KEEP = ["none", "fact", "preference", "passing", "unsure"]
SHIFT = ["followup", "same", "new"]

SYSTEM = ("You label messages that people send to a personal AI assistant running on their own computer. "
          "The assistant can search the web, keep long-term memories about its user, keep a notebook of notes, "
          "read and write files and run code. You answer only with JSON.")

GUIDE = """Label the MESSAGE at the end. Judge the message alone, exactly as the user wrote it.

intent: what the user mainly wants
  chat: greeting, thanks, small talk or a reaction, with no request ("hi", "thanks!", "lol nice", "how's your day?")
  question: wants information, an explanation, advice, a recommendation or an opinion
  task: wants something made or changed: write, code, fix, translate, summarise, rewrite, plan, calculate, convert, list
  share: tells something about themselves or their situation and asks for nothing
  remember: asks the assistant to remember, update or forget something about them
  note: asks to write something down, save it as a note or add it to a list, or dictates a note, list or reminder to keep
  recall: asks about something they told, saved or discussed before ("what's my sister's name?", "what did I save about the trip?", "what did we talk about yesterday?")
  feedback: reacts to the assistant's previous answer: it is wrong, too long, not what they meant, or it was good, with no new request
  continue: asks to carry on with what was going on ("go on", "more", "next one", "yes, do it", "ok", "the second option")
  meta: asks about the assistant itself: who or what it is, what it can do, its settings, how it works
  When a message mixes several, pick what the reply must deliver: a statement followed by a question is "question".

web: would a good answer need a web search?
  none: no; general knowledge, reasoning, advice, writing or code is enough
  own: no, it is about the user's own things: their notes, memories, files, computer or this conversation
  asked: yes, the user explicitly asks to search, look up, google, browse or check online
  live: yes, it needs current information that changes: news, weather, prices, rates, scores, results, schedules, who holds a position now, latest versions, recent or upcoming events, anything after 2025
  lookup: a search would make it reliable: a specific detail a model may not know or may misremember (an exact figure, a niche fact, a specific product, person, place or paper, a link)

time: what time the message is about
  none: no particular time
  clock: the current time, date, day or year itself, or counting from now ("what day is it?", "how many days until Christmas?")
  now: the present state of things ("currently", "right now", "still", "today's weather")
  recent: the last days, weeks or months ("this week", "yesterday's match", "lately", "recent news")
  past: a particular earlier time or period ("in 2019", "last summer", "the 90s", "in Roman times")
  future: an upcoming time ("tomorrow", "next week", "this weekend", "in 2030")

topic: the main subject, one of
  software (programming, code, dev tools, databases, websites), computers (hardware, operating systems, devices, networks, apps and how to use them), ai (artificial intelligence, machine learning, language models, chatbots), science (physics, chemistry, biology, space, earth, environment, engineering), math (maths, statistics, calculations, puzzles), health (medicine, the body, fitness, nutrition, mental health), money (finance, investing, prices, business, economy, shopping, crypto), work (jobs, careers, workplace, productivity, management), learning (education, studying, exams, languages and grammar), writing (stories, poems, essays, emails, letters, editing and wording), society (politics, law, government, news, religion, philosophy, history), entertainment (films, TV, music, books, games, art, celebrities, hobbies), sports, places (travel, countries, cities, geography, weather, transport), home (cooking, recipes, house, garden, DIY, pets, cars, everyday life), people (family, relationships, friends, feelings, personal life), kiwibro (the assistant itself or this app), none (no subject: greetings, thanks, bare reactions)

keep: does the message tell something about the user worth keeping in long-term memory?
  none: nothing about the user worth keeping (questions, tasks, small talk, facts about the world)
  fact: a lasting fact about the user or their life: name, age, job, where they live, family, pets, things they own, skills, health conditions, ongoing projects, goals
  preference: a lasting like or dislike, or how they want answers ("I prefer short answers", "I'm vegetarian", "use metric units")
  passing: about the user but temporary or trivial: mood, what they are doing today, what they just ate
  unsure: not a real statement about the user: hypothetical, pretend, joking, quoted, an example, or about someone else ("if I were a doctor", "my friend thinks", "pretend I'm a pirate")

time_phrases: the exact words of the message that name a time or a period, copied as written ("last week", "in 2019", "tomorrow morning", "since March"). Not the words "time" or "date" alone. [] if none.
asks: the exact words that set a requirement for the answer: a format, a length, a number of items, a style or tone, an audience, a language or tool to use, something to include or avoid ("in 3 bullet points", "under 100 words", "in Python", "without pandas", "for a 10-year-old", "as a table", "formal", "include sources", "step by step"). Copied as written. [] if none.
keyphrases: 1 to 3 short phrases copied from the message that name what it is specifically about ("asyncio event loop", "sourdough starter", "flat in Lisbon"). [] for chat.

MESSAGE:
"""

SHIFT_GUIDE = """A person is chatting with an AI assistant. Below are the previous exchange and the person's next message.
How does NEXT relate to the previous exchange?
  followup: NEXT cannot be understood without the previous exchange; it points back to it ("it", "that one", "why?", "and in Rust?", "make it shorter", "yes", "the second one", "what about him?")
  same: NEXT stays on the same topic or task but makes sense on its own
  new: NEXT starts a different topic or task
"""

# The same pair, five ways: what tells a follow-up from a new topic is then the only difference.
THREAD_GUIDE = """Below is the start of a chat between a person and an AI assistant. Write five different possible next messages from the person:
  elliptical: a short follow-up (1 to 6 words) that only makes sense right after this exchange because it leaves out what it is about ("and in Rust?", "why?", "shorter please", "what about the second one?", "ok do it", "and for kids?", "any cheaper ones?")
  pointing: a follow-up that refers back to the exchange (it, that, this, those, the above, your answer, the third step, a detail of the reply) and cannot be answered without it
  same: a new question or request on the same subject that is complete on its own and names what it is about
  new: a message about a completely different subject
  new_short: a short, casual message (2 to 8 words) about a completely different subject
Write the way real people type: vary the tone and the length; some lowercase, some with small typos. English only.
"""
THREAD_KINDS = {"elliptical": "followup", "pointing": "followup", "same": "same", "new": "new", "new_short": "new"}

RELATE_GUIDE = """A personal AI assistant keeps saved items about its user: memories (short facts) and notes. The user now sends a message. Should this saved item be shown to the assistant while it answers?
  yes: the item bears on the answer: it would change, personalise or complete the reply, or the message asks about it
  no: it does not bear on the answer, even when it shares words or the general subject
"""

FACTS_GUIDE = """Write what this message says about its author as short memory facts in the third person, each starting with "User" ("User has a dog named Max.", "User prefers short answers."). Only lasting facts and preferences; leave out moods, today's plans, questions and anything hypothetical. One fact per sentence, at most 3. [] if there is none.
MESSAGE:
"""

TRIPLE_GUIDE = """A personal AI assistant keeps short memory facts about its user. Invent facts for a user who sends the message below:
  relevant: 2 facts that would truly change or personalise a good answer to this message
  unrelated: 2 facts that share words or the general subject with the message but would not matter for answering it
Write each as one short sentence starting with "User" ("User is allergic to peanuts.").
MESSAGE:
"""

SYNTH_GUIDE = """Write {k} different messages that real people might type to a personal AI assistant on their computer. Each message: {theme}
Vary the length (2 to 40 words), the tone, the wording and the subject; some casual, lowercase, with small typos; some polite and complete. Each message stands alone. Write the messages only, in English.
"""

SYNTH_THEMES = [
    "tells the assistant a lasting fact about themselves (job, home town, family, pets, skills, health, what they own)",
    "tells the assistant a lasting preference: what they like or dislike, or how they want answers given",
    "mentions a passing state: their mood, what they are doing right now, today's plans, what they just ate",
    "says something hypothetical, pretend, joking, quoted, or about another person rather than themselves",
    "asks the assistant to remember something about them",
    "asks the assistant to forget or update something it knows about them",
    "asks what the assistant remembers about them, or for a detail they told it before",
    "dictates a note, a list or a reminder for the assistant to save",
    "asks the assistant to save its previous answer, or part of the chat, as a note",
    "asks about their saved notes, or to find something in them",
    "says the assistant's previous answer was wrong or not what they meant",
    "reacts to the assistant's previous answer: too long, too vague, great, thanks that worked",
    "asks the assistant to go on, continue, do the next step, or just says yes or ok to its offer",
    "asks who or what the assistant is, what it can do, how it works, or about its settings",
    "asks the current time, date, day or year, or how many days until or since something",
    "asks about current news, weather, prices, exchange rates, scores, results or who won",
    "explicitly asks the assistant to search the web or look something up online",
    "asks for a specific factual detail: an exact figure, a specification, a date of birth, a niche fact",
    "asks about something that happened in a particular past year or period",
    "asks about upcoming events, deadlines or plans: tomorrow, next week, next month, next year",
    "asks for something with explicit requirements on format, length, number of items, style, tone or audience",
    "asks for code with constraints: the language, libraries to use or avoid, style, tests",
    "is written in English but quotes a few words in another language and asks what they mean, how to say or pronounce them",
    "is very short: one to four words, of any kind",
    "is a greeting or small talk",
    "asks about the user's own files, folders, downloads or computer",
    "mixes a fact about the user with a question that depends on it",
    "only makes sense right after an earlier answer (\"and in Rust?\", \"why?\", \"what about the second one?\")",
    "asks about something said earlier in the same chat",
    "asks two or three different questions at once",
    "uses words like may, march, spring, fall, sunday or second in a sense that has nothing to do with time",
    "asks for a summary, comparison or explanation of a document, article or text they mention",
]


GUIDE_LITE = re.sub(r"\ntime_phrases:.*?(?=\nMESSAGE:)", "", GUIDE, flags=re.S)


def log(*parts) -> None:
    print(*parts, flush=True)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def usable(text: str) -> bool:
    return 1 <= len(text) <= MAX_CHARS and any(ch.isalpha() for ch in text)


class Dedup:
    """Exact duplicates and template spam: the same opening words at most ``per_opening`` times."""

    def __init__(self, per_opening: int = 3) -> None:
        self.seen: set[str] = set()
        self.openings: Counter = Counter()
        self.per_opening = per_opening

    def add(self, text: str) -> bool:
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        opening = " ".join(key.split()[:6])
        if not key or key[:160] in self.seen or (len(key.split()) > 6 and self.openings[opening] >= self.per_opening):
            return False
        self.seen.add(key[:160])
        self.openings[opening] += 1
        return True


# ---------------------------------------------------------------- pool

def pool() -> None:
    from datasets import load_dataset

    rng = random.Random(SEED)
    dedup = Dedup()
    messages: list[dict] = []
    pairs: list[dict] = []
    convos: list[list[str]] = []

    def take(text: str, source: str) -> bool:
        text = clean(text)
        if usable(text) and dedup.add(text):
            messages.append({"text": text, "source": source})
            return True
        return False

    log("WildChat …")
    wild = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20000)
    singles = turns = 0
    for row in wild:
        if row.get("toxic") or row.get("language") != "English":
            continue
        conv = [m for m in row["conversation"] if m.get("content")]
        users = [i for i, m in enumerate(conv) if m["role"] == "user"]
        for i in users[:3]:
            if singles < 20000 and take(conv[i]["content"], "wildchat"):
                singles += 1
        for a, b in zip(users, users[1:]):
            first, reply, second = clean(conv[a]["content"]), "", clean(conv[b]["content"])
            if a + 1 < len(conv) and conv[a + 1]["role"] == "assistant":
                reply = clean(conv[a + 1]["content"])[:400]
            if usable(first) and usable(second) and turns < 9000:
                pairs.append({"prev": first[:800], "reply": reply, "next": second, "source": "wildchat"})
                turns += 1
                break
        if len(users) >= 2:
            convos.append([clean(conv[i]["content"])[:600] for i in users[:6]])
        if singles >= 20000 and turns >= 9000:
            break
    log("  messages", singles, "pairs", turns)

    log("oasst2 …")
    oasst = load_dataset("OpenAssistant/oasst2", split="train")
    by_id = {}
    for row in oasst:
        by_id[row["message_id"]] = row
    count = 0
    for row in oasst:
        if row["role"] != "prompter" or row["lang"] != "en" or row.get("deleted"):
            continue
        if count < 6000 and take(row["text"], "oasst"):
            count += 1
        parent = by_id.get(row["parent_id"]) if row["parent_id"] else None
        grand = by_id.get(parent["parent_id"]) if parent and parent["parent_id"] else None
        if parent and grand and grand["role"] == "prompter" and rng.random() < 0.5:
            first, second = clean(grand["text"]), clean(row["text"])
            if usable(first) and usable(second):
                pairs.append({"prev": first[:800], "reply": clean(parent["text"])[:400], "next": second, "source": "oasst"})
    log("  messages", count)

    log("ultrachat …")
    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=SEED).select(range(12000))
    count = 0
    for row in ultra:
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        replies = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if users and count < 5000 and take(users[0], "ultrachat"):
            count += 1
        if len(users) >= 2 and rng.random() < 0.3:
            first, second = clean(users[0]), clean(users[1])
            if usable(first) and usable(second):
                pairs.append({"prev": first[:800], "reply": clean(replies[0] if replies else "")[:400],
                              "next": second, "source": "ultrachat"})
    log("  messages", count)

    log("dolly, massive, clinc …")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=SEED)
    count = 0
    for row in dolly:
        if count < 3500 and take(row["instruction"], "dolly"):
            count += 1
    massive = load_dataset("mteb/amazon_massive_intent", "en", split="train").shuffle(seed=SEED)
    count = 0
    for row in massive:
        if count < 3500 and take(row["text"], "massive"):
            count += 1
    clinc = load_dataset("clinc/clinc_oos", "plus", split="train").shuffle(seed=SEED)
    count = 0
    for row in clinc:
        if count < 2500 and take(row["text"], "clinc"):
            count += 1

    # Topic switches inside real chats are rarer than in life: some pairs join two chats.
    spliced = 0
    for _ in range(3500):
        a, b = rng.sample(convos, 2)
        first, second = rng.choice(a), rng.choice(b)
        if usable(first) and usable(second):
            pairs.append({"prev": first, "reply": "", "next": second, "source": "spliced"})
            spliced += 1
    log("  spliced pairs", spliced)

    log("persona lines …")
    lines = set()
    try:  # a script dataset: read the Hub's parquet copy of it
        persona = load_dataset("bavard/personachat_truecased", revision="refs/convert/parquet", split="train")
        for row in persona:
            for line in row["personality"]:
                lines.add(line.strip())
            if len(lines) > 6000:
                break
    except Exception as error:  # the facts stage still has the stated facts from real messages
        log("  persona lines unavailable:", error)
    rng.shuffle(messages)
    rng.shuffle(pairs)
    write_jsonl(DATA / "pool_messages.jsonl", messages)
    write_jsonl(DATA / "pool_pairs.jsonl", pairs)
    write_jsonl(DATA / "persona_lines.jsonl", [{"text": line} for line in sorted(lines)])
    write_jsonl(DATA / "conversations.jsonl", [{"turns": c} for c in convos[:6000]])
    log("pool:", len(messages), "messages,", len(pairs), "pairs,", len(lines), "persona lines")


# ---------------------------------------------------------------- teacher

_llm = None


def teacher():
    global _llm
    if _llm is None:
        from vllm import LLM

        _llm = LLM(model=TEACHER, max_model_len=4096, gpu_memory_utilization=0.90, max_num_seqs=256,
                   enable_prefix_caching=True, seed=SEED)
    return _llm


def json_params(schema: dict, temperature: float = 0.0, max_tokens: int = 400):
    from vllm import SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))


def ask(prompts: list[str], schema: dict | list[dict], temperature: float = 0.0, max_tokens: int | list[int] = 400,
        system: str = SYSTEM) -> list[dict | None]:
    """One answer per prompt; ``schema`` and ``max_tokens`` may be lists, one per prompt."""
    conversations = [[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in prompts]
    if isinstance(schema, list):
        tokens = max_tokens if isinstance(max_tokens, list) else [max_tokens] * len(prompts)
        params = [json_params(one, temperature, size) for one, size in zip(schema, tokens)]
    else:
        params = json_params(schema, temperature, max_tokens)
    outputs = teacher().chat(conversations, params, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    answers: list[dict | None] = []
    for out in outputs:
        try:
            answers.append(json.loads(out.outputs[0].text))
        except Exception:
            answers.append(None)
    return answers


def enum(values: list[str]) -> dict:
    return {"type": "string", "enum": values}


def strings(limit: int) -> dict:
    return {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": limit}


MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": enum(INTENTS), "web": enum(WEB), "time": enum(TIME), "topic": enum(TOPICS), "keep": enum(KEEP),
        "time_phrases": strings(4), "asks": strings(6), "keyphrases": strings(3),
    },
    "required": ["intent", "web", "time", "topic", "keep", "time_phrases", "asks", "keyphrases"],
    "additionalProperties": False,
}


def synth() -> None:
    rng = random.Random(SEED + 1)
    prompts, themes = [], []
    for theme in SYNTH_THEMES:
        for _ in range(26):
            prompts.append(SYNTH_GUIDE.format(k=10, theme=theme))
            themes.append(theme)
    schema = {"type": "object", "properties": {"messages": strings(10)}, "required": ["messages"]}
    answers = ask(prompts, schema, temperature=1.0, max_tokens=900,
                  system="You write realistic example messages. You answer only with JSON.")
    dedup = Dedup(per_opening=4)
    rows = []
    for theme, answer in zip(themes, answers):
        for text in (answer or {}).get("messages", []):
            text = clean(text).strip('"')
            if usable(text) and dedup.add(text):
                rows.append({"text": text, "source": "synth", "theme": theme})
    rng.shuffle(rows)
    write_jsonl(DATA / "synth_messages.jsonl", rows)
    log("synth:", len(rows))


def messages() -> None:
    rows = read_jsonl(DATA / "pool_messages.jsonl")[:N_POOL] + read_jsonl(DATA / "synth_messages.jsonl")[:N_SYNTH]
    answers = ask([GUIDE + row["text"] for row in rows], MESSAGE_SCHEMA, max_tokens=300)
    out = [{**row, **answer} for row, answer in zip(rows, answers) if answer]
    write_jsonl(DATA / "labeled_messages.jsonl", out)
    log("labelled messages:", len(out), "of", len(rows))
    for field in ("intent", "web", "time", "topic", "keep"):
        log(field, Counter(row[field] for row in out).most_common())


LITE_SCHEMA = {
    "type": "object",
    "properties": {"intent": enum(INTENTS), "web": enum(WEB), "time": enum(TIME), "topic": enum(TOPICS), "keep": enum(KEEP)},
    "required": ["intent", "web", "time", "topic", "keep"],
    "additionalProperties": False,
}


LITE_BATCH, LITE_CHARS = 8, 4800  # messages per prompt; their total length keeps the prompt in the window


def lite_prompt(texts: list[str]) -> str:
    head = GUIDE_LITE.replace(
        "Label the MESSAGE at the end. Judge the message alone, exactly as the user wrote it.",
        f"Label each of the {len(texts)} MESSAGES at the end, in order. They come from different people: "
        "judge each one alone, exactly as its user wrote it.")
    head = head[:head.rindex("MESSAGE:")]
    return head + "\n".join(f"MESSAGE {i}:\n<<<\n{text}\n>>>" for i, text in enumerate(texts, 1))


def lite() -> None:
    """The five readings (no spans) for many more messages than the full stage labels. The long
    guide is read once per batch of messages: the teacher cannot cache it."""
    rows = read_jsonl(DATA / "pool_messages.jsonl")[:N_LITE] + read_jsonl(DATA / "synth_messages.jsonl")
    batches: list[list[dict]] = [[]]
    for row in rows:
        size = sum(len(r["text"]) for r in batches[-1])
        if len(batches[-1]) >= LITE_BATCH or (batches[-1] and size + len(row["text"]) > LITE_CHARS):
            batches.append([])
        batches[-1].append(row)
    schemas = [{"type": "object", "properties": {"labels": {"type": "array", "items": LITE_SCHEMA,
                                                             "minItems": len(b), "maxItems": len(b)}},
                "required": ["labels"]} for b in batches]
    answers = ask([lite_prompt([r["text"] for r in b]) for b in batches], schemas,
                  max_tokens=[48 * len(b) for b in batches])
    out = []
    for batch, answer in zip(batches, answers):
        labels = (answer or {}).get("labels", [])
        if len(labels) == len(batch):
            out += [{**row, **label} for row, label in zip(batch, labels)]
    write_jsonl(DATA / "labeled_lite.jsonl", out)
    log("labelled (lite):", len(out), "of", len(rows))
    for field in ("intent", "web", "time", "topic", "keep"):
        log(field, Counter(row[field] for row in out).most_common())


def facts() -> None:
    labelled = read_jsonl(DATA / "labeled_messages.jsonl")
    stating = [row["text"] for row in labelled if row["keep"] in ("fact", "preference")]
    persona = [row["text"] for row in read_jsonl(DATA / "persona_lines.jsonl")]
    random.Random(SEED + 5).shuffle(persona)
    persona = persona[:N_PERSONA]
    schema = {"type": "object", "properties": {"facts": strings(3)}, "required": ["facts"]}
    sources = stating + persona
    answers = ask([FACTS_GUIDE + text for text in sources], schema, max_tokens=160)
    facts_out = []
    for text, answer in zip(sources, answers):
        for fact in (answer or {}).get("facts", []):
            fact = clean(fact)
            if fact.startswith("User") and 8 <= len(fact) <= 160:
                facts_out.append({"fact": fact, "from": text})
    # Invented facts around real questions: two that matter, two that only look like they do.
    rng = random.Random(SEED + 2)
    asking = [row["text"] for row in labelled if row["intent"] in ("question", "task") and len(row["text"]) < 400]
    rng.shuffle(asking)
    schema3 = {"type": "object", "properties": {"relevant": strings(2), "unrelated": strings(2)},
               "required": ["relevant", "unrelated"]}
    chosen = asking[:N_TRIPLES]
    answers = ask([TRIPLE_GUIDE + text for text in chosen], schema3, temperature=0.7, max_tokens=220)
    triples = [{"message": text, **answer} for text, answer in zip(chosen, answers) if answer]
    write_jsonl(DATA / "facts.jsonl", facts_out)
    write_jsonl(DATA / "triples.jsonl", triples)
    log("facts:", len(facts_out), "triples:", len(triples))


def shift() -> None:
    pairs = read_jsonl(DATA / "pool_pairs.jsonl")[:N_PAIRS]
    prompts = []
    for pair in pairs:
        reply = f"PREVIOUS (assistant, beginning): {pair['reply'][:400]}\n" if pair["reply"] else ""
        prompts.append(f"{SHIFT_GUIDE}\nPREVIOUS (user): {pair['prev'][:800]}\n{reply}NEXT (user): {pair['next']}")
    schema = {"type": "object", "properties": {"label": enum(SHIFT)}, "required": ["label"]}
    answers = ask(prompts, schema, max_tokens=20)
    out = [{**pair, "label": answer["label"]} for pair, answer in zip(pairs, answers) if answer]
    write_jsonl(DATA / "labeled_shift.jsonl", out)
    log("shift:", Counter(row["label"] for row in out).most_common())


def threads() -> None:
    rng = random.Random(SEED + 4)
    contexts = [p for p in read_jsonl(DATA / "pool_pairs.jsonl") if p["source"] != "spliced"]
    rng.shuffle(contexts)
    contexts = contexts[:N_THREADS]
    prompts = []
    for pair in contexts:
        reply = f"ASSISTANT (beginning): {pair['reply'][:400]}\n" if pair["reply"] else ""
        prompts.append(f"{THREAD_GUIDE}\nPERSON: {pair['prev'][:800]}\n{reply}")
    text = {"type": "string", "maxLength": 240}
    schema = {"type": "object", "properties": {kind: text for kind in THREAD_KINDS},
              "required": list(THREAD_KINDS), "additionalProperties": False}
    answers = ask(prompts, schema, temperature=0.9, max_tokens=260,
                  system="You write realistic example messages. You answer only with JSON.")
    out = []
    for pair, answer in zip(contexts, answers):
        for kind, label in THREAD_KINDS.items():
            message = clean((answer or {}).get(kind, "")).strip('"')
            if usable(message):
                out.append({"prev": pair["prev"][:800], "reply": pair["reply"], "next": message, "label": label,
                            "kind": kind, "source": "thread"})
    write_jsonl(DATA / "labeled_threads.jsonl", out)
    log("threads:", Counter(row["kind"] for row in out).most_common())


def relate() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    rng = random.Random(SEED + 3)
    labelled = read_jsonl(DATA / "labeled_messages.jsonl")
    fact_rows = read_jsonl(DATA / "facts.jsonl")
    triples = read_jsonl(DATA / "triples.jsonl")
    conversations = read_jsonl(DATA / "conversations.jsonl")
    items = sorted({row["fact"] for row in fact_rows})
    # Notes and earlier messages are saved items too: earlier turns of real chats.
    notes = [turn for conv in conversations[:3000] for turn in conv["turns"][:-1] if 20 <= len(turn) <= 500]
    items_all = items + notes
    queries = [row["text"] for row in labelled if row["intent"] not in ("chat",) and len(row["text"]) < 500]
    rng.shuffle(queries)
    queries = queries[:N_QUERIES]
    vec = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(items_all + queries)
    index = NearestNeighbors(n_neighbors=4, metric="cosine").fit(vec.transform(items_all))
    _, near = index.kneighbors(vec.transform(queries))
    pairs = []
    for query, hits in zip(queries, near):
        for hit in hits[:2]:  # close in words: the hard cases either way
            pairs.append({"message": query, "item": items_all[hit], "kind": "near"})
        pairs.append({"message": query, "item": rng.choice(items_all), "kind": "random"})
    # Later turns of a chat against its earlier turns: recall of an earlier exchange.
    for conv in conversations[3000:4000]:
        turns = conv["turns"]
        if len(turns) >= 3:
            pairs.append({"message": turns[-1], "item": rng.choice(turns[:-1]), "kind": "earlier"})
    prompts = [f"{RELATE_GUIDE}\nSAVED ITEM: {p['item']}\nMESSAGE: {p['message']}" for p in pairs]
    schema = {"type": "object", "properties": {"label": enum(["yes", "no"])}, "required": ["label"]}
    answers = ask(prompts, schema, max_tokens=12)
    out = [{**pair, "label": answer["label"]} for pair, answer in zip(pairs, answers) if answer]
    for triple in triples:
        for fact in triple.get("relevant", []):
            out.append({"message": triple["message"], "item": fact, "kind": "made_yes", "label": "yes"})
        for fact in triple.get("unrelated", []):
            out.append({"message": triple["message"], "item": fact, "kind": "made_no", "label": "no"})
    rng.shuffle(out)
    write_jsonl(DATA / "labeled_relate.jsonl", out)
    log("relate:", Counter((row["kind"], row["label"]) for row in out).most_common())


STAGES = {
    "pool": (pool, "pool_messages.jsonl"),
    "synth": (synth, "synth_messages.jsonl"),
    "messages": (messages, "labeled_messages.jsonl"),
    "lite": (lite, "labeled_lite.jsonl"),
    "facts": (facts, "triples.jsonl"),
    "shift": (shift, "labeled_shift.jsonl"),
    "threads": (threads, "labeled_threads.jsonl"),
    "relate": (relate, "labeled_relate.jsonl"),
}


def main() -> None:
    wanted = sys.argv[1:] or ["all"]
    names = list(STAGES) if wanted == ["all"] else wanted
    for name in names:
        run, output = STAGES[name]
        if (DATA / output).exists():
            log(f"[{name}] done already ({output})")
            continue
        log(f"[{name}] …")
        run()


if __name__ == "__main__":
    main()
