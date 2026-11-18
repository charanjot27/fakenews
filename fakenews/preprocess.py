"""The two text views.

classical view (BoW / TF-IDF): lowercase, links -> 'urltoken', keep Devanagari, keep '!' '?' and emojis as tokens.
bert view (transformers):      close to the original text; only whitespace is normalised and links replaced.

The tokenizer pattern treats the whole Devanagari block (U+0900-U+097F) as word characters. Without it,
scikit-learn's default pattern splits Hindi words at vowel signs (matras), e.g. 'बारिश' -> 'ब', 'र', 'श'.
"""
from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|in|org|net|ly|co)(/\S*)?\b)", re.IGNORECASE)
WS_RE = re.compile(r"\s+")

DEVANAGARI = "ऀ-ॿ"
EMOJI = "\U0001F300-\U0001FAFF☀-➿"
# word = run of letters/digits/Devanagari (incl. matras); '!!!' and '?' runs; single emojis
TOKEN_PATTERN = rf"(?u)[\w{DEVANAGARI}]+|[!?]+|[{EMOJI}]"
TOKEN_RE = re.compile(TOKEN_PATTERN)

EMOJI_RE = re.compile(f"[{EMOJI}]")
DEVANAGARI_RE = re.compile(f"[{DEVANAGARI}]")


def normalise(text) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    return WS_RE.sub(" ", text).strip()


def classical_view(text) -> str:
    t = normalise(text).lower()
    t = URL_RE.sub(" urltoken ", t)
    return WS_RE.sub(" ", t).strip()


def bert_view(text) -> str:
    t = normalise(text)
    t = URL_RE.sub("URL", t)
    return t


def tokenize(text: str) -> list[str]:
    """Same tokenization the BoW/TF-IDF vectorizers use (for explanations and tests)."""
    return TOKEN_RE.findall(text)


def style_features(text: str) -> dict:
    """Features of writing style only (used by the style-only shortcut probe)."""
    t = str(text)
    letters = [c for c in t if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    words = t.split()
    return {
        "n_chars": len(t),
        "n_words": len(words),
        "n_exclaim": t.count("!"),
        "n_question": t.count("?"),
        "caps_ratio": upper / max(1, len(letters)),
        "n_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()),
        "n_emoji": len(EMOJI_RE.findall(t)),
        "has_link": int(bool(URL_RE.search(t)) or "link" in t.lower()),
        "devanagari_ratio": len(DEVANAGARI_RE.findall(t)) / max(1, len(t)),
    }


HINGLISH_MARKERS = set("""hai hain ka ki ke ko se mein me par nahi nahin kya karo karein kar raha rahi rahe
gaya gayi ho hoga hogi jayega jayenge aur bhi sab sabhi yeh ye wo woh tha thi apna apni kal aaj abhi
sarkar baat log logon wala wali bhejo dekho mat liye wajah""".split())


def detect_language(text: str) -> str:
    """Heuristic used only when your data has no language column: hi / hinglish / en."""
    t = str(text)
    if not t:
        return "en"
    if len(DEVANAGARI_RE.findall(t)) / max(1, sum(c.isalpha() for c in t)) > 0.3:
        return "hi"
    words = re.findall(r"[a-zA-Z]+", t.lower())
    if words and sum(w in HINGLISH_MARKERS for w in words) / len(words) >= 0.15:
        return "hinglish"
    return "en"


def near_dup_key(text: str) -> str:
    """Key for near-duplicate detection: ignores case, links, emojis, digits and punctuation."""
    t = URL_RE.sub(" ", str(text).lower())
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(rf"[^a-z{DEVANAGARI}\s]", " ", t)
    return WS_RE.sub(" ", t).strip()
