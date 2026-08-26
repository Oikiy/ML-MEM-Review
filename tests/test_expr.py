import pandas as pd
import pytest

from utl_expr import evaluate_arithmetic, evaluate_arithmetic_frame


def test_evaluate_arithmetic_add():
    assert evaluate_arithmetic("a + b", {"a": 1.0, "b": 3.0}) == 4.0


def test_evaluate_arithmetic_ops_and_parens():
    assert evaluate_arithmetic("1.0 - a * (b + 2)", {"a": 2.0, "b": 3.0}) == -9.0


def test_evaluate_arithmetic_rejects_import():
    with pytest.raises(ValueError, match="restricted arithmetic"):
        evaluate_arithmetic("__import__('os')", {"a": 1.0})


def test_evaluate_arithmetic_rejects_attribute_access():
    with pytest.raises(ValueError, match="restricted arithmetic"):
        evaluate_arithmetic("a.__class__", {"a": 1.0})


def test_evaluate_arithmetic_unknown_name():
    with pytest.raises(ValueError, match="Unknown name"):
        evaluate_arithmetic("a + missing", {"a": 1.0})


def test_evaluate_arithmetic_frame_vectorized():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = evaluate_arithmetic_frame(df, "a + b")
    assert list(result) == [4.0, 6.0]
