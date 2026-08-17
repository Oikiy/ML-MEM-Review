import pandas as pd
import os


def check_and_merge(df, merge_configs, verbose=False):
    """
    Safely merge multiple property tables into the main dataframe.

    For each merge configuration, this function:
    - checks whether the merge key exists in both dataframes,
    - verifies the lookup table key is unique (to prevent row multiplication),
    - performs a left merge if it does,
    - logs warnings if columns are missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe to enrich.
    merge_configs : list of (str, pandas.DataFrame, str)
        List of merge instructions. Each entry is:
        (merge_key, property_df, property_df_name)
        where
        - merge_key is the column name to join on,
        - property_df is the lookup table with properties,
        - property_df_name is a label used in log messages.
    verbose : bool, optional
        If True, prints detailed status information for each merge.

    Returns
    -------
    pandas.DataFrame
        Dataframe with property tables merged in where possible.
    """
    if verbose:
        print("\n=== Checking merge columns for feature engineering ===")

    for col_name, prop_df, prop_df_name in merge_configs:
        in_main_df = col_name in df.columns
        in_prop_df = col_name in prop_df.columns

        if verbose:
            print(f"\nColumn '{col_name}':")
            print(f"  - Present in main df: {in_main_df}")
            print(f"  - Present in {prop_df_name}: {in_prop_df}")

        if not in_main_df:
            if verbose:
                print(f"  Skipping: '{col_name}' is missing from main dataframe.")
        elif not in_prop_df:
            if verbose:
                print(f"  Skipping: '{col_name}' is missing from {prop_df_name}.")
        else:
            # Guard against duplicate lookup keys that would duplicate rows in the main dataframe.
            if prop_df[col_name].duplicated().any():
                duplicate_keys = prop_df.loc[prop_df[col_name].duplicated(), col_name].dropna().unique().tolist()
                raise ValueError(
                    f"Duplicate merge keys detected in {prop_df_name} for '{col_name}': {duplicate_keys}. "
                    "Lookup keys must be unique before merging."
                )

            df = df.merge(prop_df, on=col_name, how="left")
            if verbose:
                print(f"  Successfully merged {prop_df_name} on '{col_name}'")

    if verbose:
        print("=" * 50)

    return df


def _load_lookup_table(lookup_source, source_name):
    """
    Load a lookup table from either a DataFrame or a file path.

    Parameters
    ----------
    lookup_source : pandas.DataFrame or str
        Source lookup table. Can be an in-memory DataFrame or a file path.
    source_name : str
        Human-readable source label for error messages.

    Returns
    -------
    pandas.DataFrame
        Loaded lookup table.
    """
    if isinstance(lookup_source, pd.DataFrame):
        return lookup_source.copy()

    if isinstance(lookup_source, str):
        try:
            lower_path = lookup_source.lower()
            if lower_path.endswith(".csv"):
                return pd.read_csv(lookup_source)
            if lower_path.endswith((".xls", ".xlsx")):
                return pd.read_excel(lookup_source)
            raise ValueError(
                f"Unsupported lookup file format for {source_name}: {lookup_source}. "
                "Only .csv, .xls, and .xlsx are supported."
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Feature-engineering property file missing for {source_name}: {exc}"
            ) from exc

    raise TypeError(
        f"Unsupported lookup source type for {source_name}: {type(lookup_source).__name__}. "
        "Expected pandas.DataFrame or file path string."
    )


def feature_engineering_from_lookups(df, lookup_specs, verbose=False):
    """
    Generic feature engineering via configurable lookup-table merges.

    This function is domain-agnostic and can be reused across projects.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to enrich.
    lookup_specs : list of dict
        Lookup merge specifications. Each dict supports:
        - merge_key (str): column name in df to merge on.
        - lookup_source (pandas.DataFrame or str): lookup table or path to Excel.
        - lookup_key (str): join key column in the lookup table.
        - prefix (str, optional): prefix to add to lookup property columns.
        - source_name (str, optional): label for logging/errors.
    verbose : bool, optional
        If True, prints detailed merge logs.

    Returns
    -------
    pandas.DataFrame
        Enriched dataframe.
    """
    if df.empty:
        raise ValueError("Input dataframe for feature engineering is empty.")

    merge_configs = []
    missing_merge_keys = []

    for spec in lookup_specs:
        merge_key = spec["merge_key"]
        lookup_key = spec["lookup_key"]
        source_name = spec.get("source_name", merge_key)
        prefix = spec.get("prefix", "")

        if merge_key not in df.columns:
            missing_merge_keys.append(merge_key)
            continue

        lookup_df = _load_lookup_table(spec["lookup_source"], source_name)

        if lookup_key not in lookup_df.columns:
            raise KeyError(
                f"Lookup key '{lookup_key}' is missing from {source_name}."
            )

        prepared_lookup = (
            lookup_df.set_index(lookup_key)
            .add_prefix(prefix)
            .reset_index()
            .rename(columns={lookup_key: merge_key})
        )

        merge_configs.append((merge_key, prepared_lookup, source_name))

    if missing_merge_keys:
        print(
            "Warning: the following feature-engineering key columns are missing "
            f"and will be skipped: {missing_merge_keys}"
        )

    if not merge_configs:
        return df

    return check_and_merge(df, merge_configs, verbose=verbose)


def feature_engineering(
    df,
    verbose=False,
    config=None,
):
    """
        Membrane-specific feature engineering wrapper.

        This function keeps membrane domain context (ion property lookups) and
        delegates the merge logic to the reusable generic function
        `feature_engineering_from_lookups`.

    Parameters
    ----------
    df : pandas.DataFrame
        Main dataframe with columns to enrich.
    verbose : bool, optional
        If True, prints detailed logging about missing columns and merges.
    config : dict
        Configuration dictionary (from utl_config_loader.load_config()).
        Preferred format is feature_engineering.lookups, where each entry is
        a dict with:
        - align_column (required): column in df to merge on.
        - lookup_path (required): lookup table file path.
        - lookup_key (optional): join key in lookup table. Defaults to align_column.
        - lookup_prefix (optional): prefix for merged lookup columns.
        Optionally supports feature_engineering.synthetic_feature for
        config-driven synthetic columns using full equations.

    Returns
    -------
    pandas.DataFrame
        Dataframe with aligned lookup properties merged in.
    """
    if config is None:
        raise ValueError(
            "config must be provided. "
            "Expected feature_engineering.lookups."
        )

    fe_cfg = config.get("feature_engineering", {})
    grouped_lookups = fe_cfg.get("lookups", [])

    normalized_lookups = []

    if not grouped_lookups:
        print("No feature_engineering.lookups configured; skipping enrichment lookups.")
    elif not isinstance(grouped_lookups, list):
        raise ValueError(
            "feature_engineering.lookups must be a list of lookup definitions."
        )

    for idx, lookup in enumerate(grouped_lookups):
        if not isinstance(lookup, dict):
            raise ValueError(
                f"feature_engineering.lookups[{idx}] must be a dictionary."
            )

        align_column = lookup.get("align_column")
        lookup_path = lookup.get("lookup_path")
        lookup_key = lookup.get("lookup_key", align_column)
        lookup_prefix = lookup.get("lookup_prefix", "")

        if not align_column or not lookup_path:
            raise ValueError(
                "Each feature_engineering.lookups entry must include "
                "non-empty align_column and lookup_path."
            )

        normalized_lookups.append(
            {
                "align_column": align_column,
                "lookup_path": lookup_path,
                "lookup_key": lookup_key,
                "lookup_prefix": lookup_prefix,
            }
        )

    project_root = config.get("_project_root", ".")
    resolved_lookup_specs = []
    for lookup in normalized_lookups:
        align_column = lookup["align_column"]
        resolved_path = lookup["lookup_path"]
        if not os.path.isabs(resolved_path) and not resolved_path.startswith(("$", "~")):
            resolved_path = os.path.join(project_root, resolved_path)
        else:
            resolved_path = os.path.expandvars(os.path.expanduser(resolved_path))

        resolved_lookup_specs.append(
            {
                "merge_key": align_column,
                "lookup_source": resolved_path,
                "lookup_key": lookup["lookup_key"],
                "prefix": lookup["lookup_prefix"],
                "source_name": f"lookup_{align_column}",
            }
        )

    if verbose:
        configured_columns = [item["align_column"] for item in normalized_lookups]
        print(
            "Using config-driven feature engineering lookup specs for columns: "
            f"{configured_columns}"
        )

    enriched_df = feature_engineering_from_lookups(df, resolved_lookup_specs, verbose=verbose)

    synthetic_df = get_custom_synthetic_features(
        enriched_df,
        config=config,
        verbose=verbose,
    )
    if not synthetic_df.empty:
        enriched_df = pd.concat([enriched_df, synthetic_df], axis=1)

    return enriched_df


def get_custom_synthetic_features(df, feature_specs=None, config=None, verbose=False):
    """
    Build custom synthetic feature columns from arithmetic equations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing source columns.
    feature_specs : list[dict] or dict, optional
        Synthetic feature definitions supplied directly.
        Supported formats:
        1) List of dictionaries with name and equation:
           [{"name": "total_dope_mass", "equation": "Dope1_Monomer1_Mass + Dope1_Add1_Mass"}]
        2) Dictionary mapping feature_name -> equation:
           {"total_dope_mass": "Dope1_Monomer1_Mass + Dope1_Add1_Mass"}
    config : dict, optional
        If feature_specs is None, definitions are read from
        config["feature_engineering"]["synthetic_feature"].
        The same two formats above are supported.
    verbose : bool, optional
        If True, prints created feature names.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only the newly created synthetic feature columns.

    Raises
    ------
    ValueError
        If definitions are malformed or an equation cannot be evaluated.
    TypeError
        If a computed synthetic feature is not numeric.
    """
    specs_obj = feature_specs
    if specs_obj is None and config is not None:
        specs_obj = config.get("feature_engineering", {}).get("synthetic_feature", [])

    if specs_obj is None or specs_obj == [] or specs_obj == {}:
        return pd.DataFrame(index=df.index)

    normalized_specs = []

    if isinstance(specs_obj, dict):
        for feature_name, equation in specs_obj.items():
            normalized_specs.append({"name": feature_name, "equation": equation})
    elif isinstance(specs_obj, list):
        for idx, spec in enumerate(specs_obj):
            if not isinstance(spec, dict):
                raise ValueError(f"synthetic_feature[{idx}] must be a dictionary.")
            normalized_specs.append(spec)
    else:
        raise ValueError(
            "synthetic_feature must be either a list of dictionaries or a dictionary "
            "of feature_name -> equation mappings."
        )

    synthetic_df = pd.DataFrame(index=df.index)
    for idx, spec in enumerate(normalized_specs):
        feature_name = spec.get("name")
        equation = spec.get("equation")

        if not feature_name or not isinstance(feature_name, str):
            raise ValueError(f"synthetic_feature[{idx}] must include a non-empty string 'name'.")
        if equation is None or not isinstance(equation, str) or not equation.strip():
            raise ValueError(
                f"synthetic feature '{feature_name}' must include a non-empty string 'equation'."
            )

        try:
            # Use DataFrame.eval so equations can reference columns directly.
            result = df.eval(equation, engine="python")
        except Exception as exc:
            raise ValueError(
                f"Failed to evaluate equation for synthetic feature '{feature_name}': {equation}. "
                f"Original error: {exc}"
            ) from exc

        if not pd.api.types.is_numeric_dtype(result):
            raise TypeError(
                f"Synthetic feature '{feature_name}' produced non-numeric values. "
                "Please use arithmetic equations with numeric source columns."
            )

        synthetic_df[feature_name] = result

    if verbose:
        print("Created synthetic feature columns:", synthetic_df.columns.tolist())

    return synthetic_df


def remove_low_variance_columns(df, exclude_columns=None, variance_threshold=0.0):
    """
    Remove columns that have variance below a specified threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    exclude_columns : list, optional
        List of column names to exclude from removal (e.g., target columns)
    variance_threshold : float, default=0.0
        Variance threshold below which columns are removed. 0.0 removes only zero variance columns.
        
    Returns:
    --------
    tuple
        (cleaned_df, removed_columns) - DataFrame with low variance columns removed 
        and list of removed column names
    """
    if exclude_columns is None:
        exclude_columns = []

    
    # Find columns with variance below threshold
    low_var_cols = []
    for col in df.columns:
        if col not in exclude_columns:
            # For numerical columns, check variance
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null_values = df[col].dropna()
                if len(non_null_values) > 1:
                    variance = non_null_values.var()
                    if variance <= variance_threshold:
                        low_var_cols.append(col)
                elif len(non_null_values.unique()) <= 1:
                    low_var_cols.append(col)
            # For categorical columns, check unique values (treated as zero variance if <= 1 unique)
            else:
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 1:
                    low_var_cols.append(col)
    
    # Remove these columns
    df_cleaned = df.drop(columns=low_var_cols)
    
    if low_var_cols:
        print(f"Removed {len(low_var_cols)} low variance columns (threshold: {variance_threshold}):")
        for col in low_var_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                non_null_values = df[col].dropna()
                if len(non_null_values) > 1:
                    variance = non_null_values.var()
                    print(f"  - {col}: variance={variance:.6f} (dtype: {df[col].dtype})")
                else:
                    unique_val = df[col].dropna().unique()
                    if len(unique_val) > 0:
                        print(f"  - {col}: constant value {unique_val[0]} (dtype: {df[col].dtype})")
                    else:
                        print(f"  - {col}: (all NaN) (dtype: {df[col].dtype})")
            else:
                unique_val = df[col].dropna().unique()
                if len(unique_val) > 0:
                    print(f"  - {col}: '{unique_val[0]}' (dtype: {df[col].dtype})")
                else:
                    print(f"  - {col}: (all NaN) (dtype: {df[col].dtype})")
    else:
        print(f"No columns with variance <= {variance_threshold} found.")
    
    return df_cleaned, low_var_cols