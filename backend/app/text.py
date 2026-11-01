import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")

def normalize_text(value: str) -> str:
    """Conservative normalization: preserve negation, numbers, and punctuation."""
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = TAG_RE.sub(" ", value)
    value = URL_RE.sub(" URL ", value)
    return SPACE_RE.sub(" ", value.lower()).strip()
