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
            return pd.read_excel(lookup_source)
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
    CATION_PROP_PATH=None,
    ANION_PROP_PATH=None,
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
        Main membrane dataset, containing at least the key columns
        for monomers and ions, such as:
        'Dope1_Monomer1_Name', 
        'Dope2_Monomer1_Name', 
        'S1_Cation', 'S1_Anion', 'S2_Cation', 'S2_Anion'.
    CATION_PROP_PATH : str, optional
        Path to Excel file with cation properties.
        Must contain a column 'Cation' as key.
        If not provided and config is present, loaded from config.
    ANION_PROP_PATH : str, optional
        Path to Excel file with anion properties.
        Must contain a column 'Anion' as key.
        If not provided and config is present, loaded from config.
    verbose : bool, optional
        If True, prints detailed logging about missing columns and merges.
    config : dict, optional
        Configuration dictionary (from config_loader.load_config()).
        If provided, paths are extracted from config when CATION_PROP_PATH 
        or ANION_PROP_PATH are None.

    Returns
    -------
    pandas.DataFrame
        Dataframe with additional descriptor columns for:
        - Dope1/Dope2 monomer 1 and 2 (e.g., MW, atom counts, functional groups),
        - solute 1 and solute 2 cations and anions (e.g., radius, charge).
    """
    
    # Load paths from config if not explicitly provided
    if config is not None:
        if CATION_PROP_PATH is None:
            project_root = config.get("_project_root", ".")
            data_raw = config["project"]["paths"]["data_raw"]
            CATION_PROP_PATH = os.path.join(
                project_root, data_raw, config["data"]["cation_properties"]
            )
        if ANION_PROP_PATH is None:
            project_root = config.get("_project_root", ".")
            data_raw = config["project"]["paths"]["data_raw"]
            ANION_PROP_PATH = os.path.join(
                project_root, data_raw, config["data"]["anion_properties"]
            )

    # Optional config-driven lookup specifications for membrane context.
    # This allows the membrane lookup mapping to be maintained in config.yaml.
    if config is not None:
        fe_cfg = config.get("feature_engineering", {})
        cfg_lookup_specs = fe_cfg.get("lookup_specs")

        if cfg_lookup_specs:
            project_root = config.get("_project_root", ".")
            data_raw = config["project"]["paths"]["data_raw"]

            resolved_lookup_specs = []
            for spec in cfg_lookup_specs:
                resolved_spec = {
                    "merge_key": spec["merge_key"],
                    "lookup_key": spec["lookup_key"],
                    "prefix": spec.get("prefix", ""),
                    "source_name": spec.get("source_name", spec["merge_key"]),
                }

                # Prefer explicit lookup_source if provided, otherwise resolve from data_key.
                if "lookup_source" in spec and spec["lookup_source"] is not None:
                    resolved_spec["lookup_source"] = spec["lookup_source"]
                elif "data_key" in spec:
                    data_file = config["data"][spec["data_key"]]
                    resolved_spec["lookup_source"] = os.path.join(
                        project_root, data_raw, data_file
                    )
                else:
                    raise ValueError(
                        "Each lookup spec must define either 'lookup_source' or 'data_key'."
                    )

                resolved_lookup_specs.append(resolved_spec)

            return feature_engineering_from_lookups(
                df,
                resolved_lookup_specs,
                verbose=verbose,
            )
    
    # Validate that paths are provided
    if CATION_PROP_PATH is None or ANION_PROP_PATH is None:
        raise ValueError(
            "CATION_PROP_PATH and ANION_PROP_PATH must be provided either as "
            "explicit arguments or via config parameter."
        )
    # Membrane-context lookup specifications
    lookup_specs = [
        {
            "merge_key": "S1_Cation",
            "lookup_source": CATION_PROP_PATH,
            "lookup_key": "Cation",
            "prefix": "S1_",
            "source_name": "df_cation_S1",
        },
        {
            "merge_key": "S1_Anion",
            "lookup_source": ANION_PROP_PATH,
            "lookup_key": "Anion",
            "prefix": "S1_",
            "source_name": "df_anion_S1",
        },
    ]

    return feature_engineering_from_lookups(df, lookup_specs, verbose=verbose)


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