import pandas as pd

from utl_feateng import feature_engineering, get_custom_synthetic_features


def make_config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  dataset: x.xlsx\n  targets: [kpi_1]\n" + extra, encoding="utf-8"
    )
    return str(path)


def test_no_lookups_is_noop(tmp_path):
    import yaml

    from utl_config_loader import load_config

    config = load_config(config_path=make_config(tmp_path))
    df = pd.DataFrame({"a": [1, 2, 3], "kpi_1": [0.1, 0.2, 0.3]})
    out = feature_engineering(df, verbose=False, config=config)
    assert list(out.columns) == list(df.columns)
    assert yaml.safe_load is not None  # sanity: keep yaml imported locally


def test_lookup_with_missing_align_column_is_skipped(tmp_path):
    from utl_config_loader import load_config

    lookup_cfg = (
        "feature_engineering:\n"
        "  lookups:\n"
        "    - align_column: not_in_main_df\n"
        "      lookup_path: missing_lookup.xlsx\n"
        "      lookup_prefix: prop_\n"
    )
    config = load_config(config_path=make_config(tmp_path, lookup_cfg))
    df = pd.DataFrame({"a": [1, 2], "kpi_1": [0.1, 0.2]})
    out = feature_engineering(df, verbose=False, config=config)
    assert list(out.columns) == list(df.columns)


def test_synthetic_features_from_equation(tmp_path):
    from utl_config_loader import load_config

    synth_cfg = (
        "feature_engineering:\n"
        "  synthetic_feature:\n"
        '    - name: combined\n'
        '      equation: "a + b"\n'
    )
    config = load_config(config_path=make_config(tmp_path, synth_cfg))
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "kpi_1": [0.1, 0.2]})
    synthetic = get_custom_synthetic_features(df, config=config)
    assert list(synthetic["combined"]) == [4.0, 6.0]