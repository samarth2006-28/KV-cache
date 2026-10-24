import pytest

from tieredkv.gsm8k import (build_prompt, extract, is_correct, mcnemar, target_of,
                            truncate_at_stop, wilson)


def test_target_and_prompt():
    assert target_of("She sold 3+4=7 clips.\n#### 1,072") == "1,072"
    shots = [{"question": "Q1?", "answer": "A1\n#### 1"}]
    assert build_prompt("Q2?", shots) == "Question: Q1?\nAnswer: A1\n#### 1\n\nQuestion: Q2?\nAnswer:"


def test_truncate_at_first_stop():
    assert truncate_at_stop(" 5 apples.\n#### 5\n\nQuestion: next one #### 9") == " 5 apples.\n#### 5\n\n"
    assert truncate_at_stop("no stop here") == "no stop here"


@pytest.mark.parametrize("text,target,strict,flex", [
    ("He has 2+3=5.\n#### 5", "5", True, True),
    ("Total is $1,072.", "1072", False, True),
    ("#### 18.\n", "18", True, True),
    ("It costs 7 dollars, then 12.50", "12.5", False, True),
    ("#### -3", "-3", True, True),
    ("#### 4\nQuestion: #### 9", "9", False, False),     # answer after the stop is ignored
    ("I don't know", "4", False, False),
])
def test_is_correct(text, target, strict, flex):
    r = is_correct(text, target)
    assert (r["strict"], r["flexible"]) == (strict, flex)


def test_extract_none():
    assert extract("nothing numeric") == (None, None)


def test_wilson():
    lo, hi = wilson(100, 200)
    assert lo < 0.5 < hi and 0.13 < hi - lo < 0.14     # +-7 points at n=200
    assert wilson(0, 0) == (0.0, 1.0)


def test_mcnemar():
    a = [1] * 30 + [0] * 10 + [1] * 60
    b = [0] * 30 + [1] * 10 + [1] * 60
    r = mcnemar(a, b)
    assert (r["a_only"], r["b_only"]) == (30, 10) and r["p"] < 0.01
    assert mcnemar([1, 0], [1, 0])["p"] == 1.0
    assert mcnemar([1, 0, 1, 0], [0, 1, 0, 1])["p"] == 1.0
    with pytest.raises(ValueError):
        mcnemar([1], [1, 0])
