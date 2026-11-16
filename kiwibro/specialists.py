"""Small local workflow classifiers. Predictions can add a prompt hint only."""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("specialists-v1.json.gz")
MAX_MODEL_BYTES = 2_000_000
WEIGHT_SCALE = 512
WORD = re.compile(r"[^\W_]+")
FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
# A browser prediction cannot supply its own evidence that the request calls
# for live browsing. Account changes are handled by the app's normal safeguards.
BROWSER_EVIDENCE = re.compile(
    r"https?://|\b(?:web|internet|online|website|homepage|browser|tab|page|link|linked|"
    r"directory|portal|sidebar|index|address|url|html|script|dom|publication|published|"
    r"announcement|interstitial|timetable|release notes|public source)\b|"
    r"\b(?:tarayıcı\w*|sekme\w*|sayfa\w*|bağlantı\w*|internet\w*|site\w*)\b", re.I,
)
ACCOUNT_ACTION = re.compile(
    r"\b(?:sign in to|log in to|authenticate as|enter my password)\b|"
    r"\b(?:buy|order|purchase|pay|send|submit|delete|transfer|cancel(?:lation)?|publish|post|vote)\b"
    r"[^.?!\n]{0,60}\b(?:my|account|subscription|invoice|button|funds|money|for me)\b|"
    r"\b(?:hesabıma giriş|havale gönder|ödeme yap|sipariş ver|oy ver)\b", re.I,
)
EXPLAIN_BROWSER = re.compile(r"\A\s*(?:please\s+)?(?:explain|describe|teach me|how (?:does|do|can))\b", re.I)
CHECK_LIVE = re.compile(r"https?://|\b(?:look up|search the web|check online|browse to|inspect the current)\b", re.I)
HINTS = {
    "browser": {
        "path": (
            "Inspect the current page and follow an observed link or a URL supplied by the user. "
            "Refresh the link list after navigation and verify the destination."
        ),
        "state": (
            "Recheck the current tab, URL and visible state before choosing the next action. "
            "When inspection yields only a script shell, use a rendered read if available."
        ),
    },
    "visual": {
        "scene": (
            "Keep the requested medium, subject and style. For a spatial web scene, frame the subject "
            "across viewport sizes, animate by elapsed time, and render-check two distinct poses."
        ),
        "mechanism": (
            "Map the visible parts and pivots to the mechanism. Drive motion and readings from the same "
            "state, mark illustrative dimensions, and verify the joints and controls across their range."
        ),
        "reveal": (
            "Build the labeled parts from their assembled poses. Check their paths and occlusion through "
            "the reveal, then verify the assembled, intermediate and fully separated views."
        ),
        "repair": (
            "Read the existing implementation and reproduce the visual symptom first. Patch the smallest "
            "responsible code path and check the saved page and affected motion in a browser."
        ),
    },
}


def features(text: str) -> set[str]:
    """Binary word and character features shared by training and inference."""
    sample = text.strip()
    if len(sample) > 1200:
        sample = sample[:840] + " " + sample[-360:]
    sample = FENCE.sub(" ", sample)
    folded = unicodedata.normalize("NFKD", sample.casefold().replace("ı", "i"))
    words = WORD.findall("".join(char for char in folded if not unicodedata.combining(char)))[:180]
    result = {f"w:{word}" for word in words}
    result.update(f"b:{left} {right}" for left, right in zip(words, words[1:]))
    for word in set(words):
        padded = f"^{word[:36]}$"
        result.update(
            f"c:{padded[start:start + width]}"
            for width in (3, 4)
            for start in range(len(padded) - width + 1)
        )
    return result


@lru_cache(maxsize=1)
def _models() -> dict:
    try:
        with gzip.open(MODEL_PATH, "rb") as handle:
            raw = handle.read(MAX_MODEL_BYTES + 1)
        if len(raw) > MAX_MODEL_BYTES:
            return {}
        pack = json.loads(raw)
        if pack["version"] != 1 or pack["origin"] != "google-colab" or pack["scale"] != WEIGHT_SCALE:
            return {}
        heads = pack["heads"]
        return heads if isinstance(heads, dict) else {}
    except (OSError, EOFError, ValueError, KeyError, TypeError):
        return {}


def prediction(model: dict, signals: set[str]) -> tuple[str, float, float]:
    """Return the leading class, softmax score and lead using the exported weights."""
    scores = model["bias"].copy()
    for feature in signals:
        weights = model["weights"].get(feature)
        if weights is not None:
            for index, weight in enumerate(weights):
                scores[index] += weight
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    best, second = ranked[:2]
    relative = [math.exp((value - scores[best]) / WEIGHT_SCALE) for value in scores]
    total = sum(relative)
    return model["labels"][best], 1 / total, (1 - relative[second]) / total


def admissible(head: str, message: str) -> bool:
    if head != "browser":
        return True
    request = FENCE.sub(" ", message[:1600])
    if EXPLAIN_BROWSER.search(request) and not CHECK_LIVE.search(request):
        return False
    return bool(BROWSER_EVIDENCE.search(request) and not ACCOUNT_ACTION.search(request))


def classify(model: dict, signals: set[str]) -> str:
    label, score, lead = prediction(model, signals)
    minimum, separation = model["thresholds"]
    if label in model["advice_labels"] and score >= minimum and lead >= separation:
        return label
    return "none"


def note(message: str, *, browser: bool = True, visual: bool = True) -> str:
    """Return bounded local advice when a supported workflow is recognized."""
    if not isinstance(message, str) or not message.strip() or not (browser or visual):
        return ""
    models = _models()
    if not models:
        return ""
    signals = features(message)
    hints = []
    for name, enabled in (("browser", browser), ("visual", visual)):
        if not enabled or name not in models or not admissible(name, message):
            continue
        try:
            label = classify(models[name], signals)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
        hint = HINTS[name].get(label)
        if hint:
            hints.append(f"- {hint}")
    return "Workflow hints for this request:\n" + "\n".join(hints) if hints else ""
