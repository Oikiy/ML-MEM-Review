import pandas as pd

from utl_eda import _target_is_classification


def test_numeric_binary_target_matches_model_task_auto():
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    assert _target_is_classification(labels, {"modeling": {"task": "auto"}})
    assert _target_is_classification(labels, {"modeling": {"task": "classification"}})
    assert not _target_is_classification(labels, {"modeling": {"task": "regression"}})


def test_high_cardinality_numeric_is_regression():
    values = pd.Series([float(i) for i in range(20)])
    assert not _target_is_classification(values, {"modeling": {"task": "auto"}})


def test_string_labels_are_classification():
    labels = pd.Series(["pass", "fail", "pass", "fail"])
    assert _target_is_classification(labels, {"modeling": {"task": "auto"}})
