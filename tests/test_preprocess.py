import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_data import (
    deduplicate_exact,
    cap_author_dominance,
    load_and_clean,
)

def _make_df(rows):
    return pd.DataFrame(rows, columns=["text", "Author", "timestamp", "X Likes", "X Reposts"])

def test_deduplicate_exact_removes_same_author_same_text():
    df = _make_df([
        ("hello world", "alice", "2024-01-01", 5, 1),
        ("hello world", "alice", "2024-01-02", 2, 0),  # duplicate — remove
        ("hello world", "bob",   "2024-01-01", 3, 0),  # different author — keep
        ("different",   "alice", "2024-01-01", 1, 0),
    ])
    out = deduplicate_exact(df, text_col="text", author_col="Author")
    assert len(out) == 3
    alice_rows = out[out["Author"] == "alice"]
    # alice has 2 distinct texts: "hello world" (deduped, highest kept) and "different"
    assert len(alice_rows) == 2
    hello_row = alice_rows[alice_rows["text"] == "hello world"]
    assert len(hello_row) == 1
    assert int(hello_row["X Likes"].iloc[0]) == 5

def test_cap_author_dominance():
    # 100 posts, one author has 20 (20%) → capped to 5%=5 posts
    rows = [("post", "spammer", f"2024-01-{i:02d}", i, 0) for i in range(1, 21)]
    rows += [("post", f"user{i}", "2024-01-01", 1, 0) for i in range(80)]
    df = _make_df(rows)
    out = cap_author_dominance(df, author_col="Author", max_share=0.05,
                               engagement_cols=["X Likes", "X Reposts"])
    spammer_count = (out["Author"] == "spammer").sum()
    assert spammer_count == 5
    spammer_likes = sorted(out[out["Author"] == "spammer"]["X Likes"].tolist(), reverse=True)
    assert spammer_likes == [20, 19, 18, 17, 16]

def test_load_and_clean_returns_report():
    df = _make_df([
        ("same text", "bot", "2024-01-01", 1, 0),
        ("same text", "bot", "2024-01-01", 1, 0),
        ("other",     "human", "2024-01-01", 5, 2),
    ])
    cleaned, report = load_and_clean(df, text_col="text", author_col="Author",
                                     timestamp_col="timestamp")
    assert report["rows_input"] == 3
    assert report["rows_output"] == 2
    assert report["exact_duplicates_removed"] == 1
