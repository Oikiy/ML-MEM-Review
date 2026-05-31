import pandas as pd


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
        Main dataframe containing membrane recipes / experiments.
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
def feature_engineering(
    df,
    CATION_PROP_PATH,
    ANION_PROP_PATH,
    verbose=False,
):
    """
    Enrich the membrane dataset with domain-informed features.

    This function:
    - loads property tables for aqueous monomers, organic monomers, solvents,
      cations, and anions;
    - prefixes their columns to indicate role (e.g., dope1_mon1_*, S1_*);
    - merges the corresponding properties into the main dataframe based on
      monomer and ion names.

    Parameters
    ----------
    df : pandas.DataFrame
        Main membrane dataset, containing at least the key columns
        for monomers and ions, such as:
        'Dope1_Monomer1_Name', 
        'Dope2_Monomer1_Name', 
        'S1_Cation', 'S1_Anion', 'S2_Cation', 'S2_Anion'.
    CATION_PROP_PATH : str
        Path to Excel file with cation properties.
        Must contain a column 'Cation' as key.
    ANION_PROP_PATH : str
        Path to Excel file with anion properties.
        Must contain a column 'Anion' as key.
    verbose : bool, optional
        If True, prints detailed logging about missing columns and merges.

    Returns
    -------
    pandas.DataFrame
        Dataframe with additional descriptor columns for:
        - Dope1/Dope2 monomer 1 and 2 (e.g., MW, atom counts, functional groups),
        - solute 1 and solute 2 cations and anions (e.g., radius, charge).
    """
    # Columns we ideally want for merging ion properties
    required_columns = [

        "S1_Cation",
        "S1_Anion"
    ]

    available_cols = [col for col in required_columns if col in df.columns]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        print(
            "Warning: the following feature-engineering key columns are missing "
            f"and will be skipped: {missing_cols}"
        )
        print(f"Columns available for merging: {available_cols}")

    if df.empty:
        raise ValueError("Input dataframe for feature_engineering is empty.")

    # Load property tables
    try:
        df_cation = pd.read_excel(CATION_PROP_PATH)
        df_anion = pd.read_excel(ANION_PROP_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature-engineering property file missing: {e}") from e

    # Set indices for easier prefixing and joining
    cation_props = df_cation.set_index("Cation")
    anion_props = df_anion.set_index("Anion")

    # Cation/anion properties for solute 1 and solute 2
    df_cation_S1 = (
        cation_props.add_prefix("S1_").reset_index().rename(columns={"Cation": "S1_Cation"})
    )

    df_anion_S1 = (
        anion_props.add_prefix("S1_").reset_index().rename(columns={"Anion": "S1_Anion"})
    )

    # Define all merge steps to perform
    merge_configs = [
        ("S1_Cation", df_cation_S1, "df_cation_S1"),
        ("S1_Anion", df_anion_S1, "df_anion_S1")
    ]

    df = check_and_merge(df, merge_configs, verbose=verbose)

    return df


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
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
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
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
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