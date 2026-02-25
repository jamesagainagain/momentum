"""
Clean, human-readable cluster labels.
Filters URL tracking tokens (8+ chars, <2 vowels) before TF-IDF labelling.
"""
import re

_VOWELS = set("aeiou")
_HEX_RE = re.compile(r"^[0-9a-f]{8,}$")
_MIXED_DIGIT_START = re.compile(r"^\d[a-z0-9]{7,}$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_noise_token(token: str) -> bool:
    if len(token) < 8:
        return False
    if _HEX_RE.match(token):
        return True
    if _MIXED_DIGIT_START.match(token):
        return True
    # Any mixed alphanumeric token (letters + digits) of length 8+ is a tracking ID
    has_digit = any(c.isdigit() for c in token)
    has_alpha = any(c.isalpha() for c in token)
    if has_digit and has_alpha:
        return True
    # Pure-alpha tokens: filter by vowel ratio — real words have >= 30% vowels
    alpha_chars = [c for c in token if c.isalpha()]
    if alpha_chars:
        vowel_ratio = sum(1 for c in alpha_chars if c in _VOWELS) / len(alpha_chars)
        if vowel_ratio < 0.30:
            return True
    return False


def strip_url_tokens(text: str) -> str:
    tokens = _TOKEN_RE.findall(text.lower())
    return " ".join(t for t in tokens if not _is_noise_token(t))


def clean_label_text(tokens: list, n_top: int = 3) -> str:
    clean = [t for t in tokens if not _is_noise_token(t)]
    return ", ".join(clean[:n_top]) if clean else "unlabelled"
