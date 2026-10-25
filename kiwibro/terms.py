"""Key terms: what a message is about, for matching memories and past messages.

Raw word overlap matched almost everything on function words ("is the build passing"
pulled in 7 unrelated memories through "is" and "the") and missed inflections ("where
do I live" never found "User lives in Istanbul"). Key terms drop function words, fold
English inflections to one stem and add a concept tag for a few everyday topics, so
"recommend a dinner" finds "User's favourite food is lahmacun".
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren as at be because been before being
    below between both but by can cannot could couldn did didn do does doesn doing don done down during each
    else even ever every few for from further get gets getting got had hadn has hasn have haven having he her
    here hers herself him himself his how i if in into is isn it its itself just let lets me might more most
    must mustn my myself no nor not now of off on once only or other ought our ours ourselves out over own
    please really same shall shan she should shouldn so some such than that the their theirs them themselves
    then there these they this those through to too under until up upon us very was wasn way we were weren
    what when where which while who whom why will with won would wouldn yes yet you your yours yourself
    yourselves ok okay hi hello hey thanks thank tell know think want need like make makes made go going
    something anything thing things one ones lot bit much many sure still maybe also well yeah
    user users user's said says say remember recall assistant kiwibro
    ll ve re d s t m
    ve bir bu da de ne mi mı mu mü ile için gibi çok daha ama ben sen o biz siz onlar şu var yok
    """.split()
)

# Everyday topics people ask about without naming the stored word ("dinner" vs "food").
# Each member maps to a shared tag; keep groups unambiguous — a member that is common in
# unrelated requests would pull the whole group into every prompt.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "food": ("food", "eat", "eating", "dinner", "lunch", "breakfast", "meal", "meals", "cook", "cooking",
             "recipe", "recipes", "restaurant", "dish", "cuisine", "snack", "hungry", "diet", "vegan",
             "vegetarian", "dessert"),
    "music": ("music", "song", "songs", "instrument", "instruments", "guitar", "piano", "violin", "drums",
              "drum", "sing", "singing", "band", "musician", "concert"),
    "home": ("live", "lives", "living", "lived", "home", "city", "country", "hometown", "address", "town",
             "neighbourhood", "neighborhood", "moved", "relocate", "relocated"),
    "job": ("job", "jobs", "career", "employer", "company", "profession", "occupation", "workplace",
            "colleague", "colleagues", "boss", "salary", "employed", "hired", "developer", "engineer",
            "employee", "freelancer", "intern", "internship", "startup"),
    "pet": ("pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "kitten", "parrot", "hamster", "rabbit"),
    "family": ("family", "mother", "mom", "mum", "father", "dad", "sister", "brother", "sibling", "siblings",
               "wife", "husband", "partner", "son", "daughter", "kids", "children", "parents", "grandmother",
               "grandfather", "cousin", "aunt", "uncle"),
    "language": ("language", "languages", "speak", "speaks", "fluent", "native", "translate", "accent"),
    "editor": ("editor", "ide", "vim", "neovim", "nvim", "emacs", "vscode", "jetbrains", "pycharm", "sublime",
               "helix", "zed"),
    "os": ("os", "linux", "ubuntu", "windows", "macos", "debian", "fedora", "arch", "distro", "gnome", "kde"),
    "health": ("health", "allergic", "allergy", "allergies", "medication", "medicine", "sick", "illness",
               "condition", "symptoms", "diagnosed"),
    "birthday": ("birthday", "born", "age", "birth", "birthdate"),
    "hobby": ("hobby", "hobbies", "weekend", "weekends", "pastime", "sport", "sports", "gym", "hiking"),
}
_CONCEPT_OF: dict[str, set[str]] = {}
for _tag, _members in CONCEPTS.items():
    for _member in _members:
        _CONCEPT_OF.setdefault(_member, set()).add("#" + _tag)


def words(text: str) -> list[str]:
    """Lower-cased words in order (apostrophes split words: "user's" → "user", "s")."""
    return _WORD_RE.findall((text or "").lower())


def stem(word: str) -> str:
    """Fold common English inflections to one form: plays/playing/played → play,
    lives/living/lived/live → liv, studies/studying → study. Deliberately crude — it
    only has to map a query word and a stored word to the same key."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith(("ses", "xes", "ches", "shes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 3 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 2 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w


def content_words(text: str) -> list[str]:
    """Words that carry meaning (no stop words), in order, without duplicates."""
    seen: list[str] = []
    for word in words(text):
        if len(word) < 2 or word in STOPWORDS or word.isdigit() and len(word) < 3:
            continue
        if word not in seen:
            seen.append(word)
    return seen


def key_terms(text: str) -> set[str]:
    """Stems of the content words plus concept tags (``#food``) — the matching key."""
    keys: set[str] = set()
    for word in content_words(text):
        keys.add(stem(word))
        keys.update(_CONCEPT_OF.get(word, ()))
    return keys


def concept_groups(keys: set[str]) -> list[tuple[str, ...]]:
    """Member words of each concept in ``keys`` (for widening a full-text search)."""
    return [CONCEPTS[key[1:]] for key in sorted(keys) if key.startswith("#") and key[1:] in CONCEPTS]
