"""Turn a title into a URL-friendly slug."""

from __future__ import annotations

import re
import unicodedata

_SEPARATORS = re.compile(r"[^a-z0-9]+")


# KIWIBRO: replace slugify with the first real function of your package
def slugify(text: str, max_length: int = 60) -> str:
    """Return a lowercase slug: words joined by single dashes.

    Raises TypeError when text is not a string and ValueError when
    max_length is not positive.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if max_length < 1:
        raise ValueError(f"max_length must be positive, got {max_length}")

    # Drop accents (é -> e) before removing anything that is not a letter or digit.
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _SEPARATORS.sub("-", plain.lower()).strip("-")
    if len(slug) <= max_length:
        return slug

    # Cut on a word boundary so the slug does not end mid-word.
    cut = slug[:max_length]
    return cut.rsplit("-", 1)[0] if "-" in cut else cut
