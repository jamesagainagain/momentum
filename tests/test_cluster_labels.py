import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cluster_label_utils import strip_url_tokens, clean_label_text

def test_strip_url_tokens_removes_hex_like_strings():
    text = "deepmind owq9fagevl 5tjwz1kzcf artificial intelligence london"
    cleaned = strip_url_tokens(text)
    assert "owq9fagevl" not in cleaned
    assert "5tjwz1kzcf" not in cleaned
    assert "deepmind" in cleaned
    assert "london" in cleaned

def test_strip_url_tokens_keeps_real_words():
    text = "alphafold protein structure deepmind"
    assert strip_url_tokens(text) == text

def test_clean_label_text_produces_readable_label():
    tokens = ["deepmind", "owq9fagevl", "protein", "lrvqaf3vww", "ai"]
    label = clean_label_text(tokens, n_top=3)
    assert "owq9fagevl" not in label
    assert "lrvqaf3vww" not in label
    assert label
