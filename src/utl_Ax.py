
import os
import sys
import joblib
import pandas as pd
import warnings
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
CODE_PATH = os.path.join(PROJECT_ROOT, "src")
CATION_PROP_PATH = os.path.join(DATA_PATH, "cation_prop.xlsx")
ANION_PROP_PATH = os.path.join(DATA_PATH, "anion_prop.xlsx")

if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

from utl_feateng import feature_engineering

def normalize(value: float, norm_dict: dict, key: str) -> float:
    """
    Normalize a metric into [0, 1] based on fixed min/max.

    Parameters
    ----------
    value : float
        Raw metric value.
    key : str
        Metric name ('PureWaterPermeability' or 'solute1Rejection').

    Returns
    -------
    float
        Normalized value between 0 and 1.
    """
    vmin, vmax = norm_dict[key]
    return (float(value) - vmin) / (vmax - vmin)


def scalar_objective(pwp_norm: float, rej_norm: float, weight_dict: dict) -> float:
    """
    Combine normalized permeability and rejection into a single scalar objective.

    Parameters
    ----------
    pwp_norm : float
        Normalized pure water permeability.
    rej_norm : float
        Normalized solute1 rejection.

    Returns
    -------
    float
        Scalarized objective in [0, 1], larger is better.
    """
    return (
        weight_dict["PureWaterPermeability"] * pwp_norm
        + weight_dict["solute1Rejection"] * rej_norm
    )

def oracle_from_params(
    params: dict,
    constant_params:dict,
    norm_dict: dict,
    # model_path_dict:dict,
    weight_dict: dict,
    cation_path:str,
    anion_path:str,
    
) -> dict:
    # 1) Defaults for missing recipe/test variables (Constant or not included in search space).
    # Merge Ax parameters with defaults (Ax parameters override defaults).
    combined = {**constant_params, **params}

    # 2) Compute implied solvent mass fractions so each dope is approximately mass-balanced.
    # Updated column names to match the rest of the codebase:
    combined["Dope1_Solvent1_Mass"] = max(
        0.0,
        1.0
        - combined["Dope1_Monomer1_Mass"]
        - combined["Dope1_Add1_Mass"]
        - combined.get("Dope1_Monomer2_Mass", 0.0)
        - combined.get("Dope1_Solvent2_Mass", 0.0)
        - combined.get("Dope1_Add2_Mass", 0.0),
    )
    combined["Dope2_Solvent1_Mass"] = max(
        0.0,
        1.0
        - combined["Dope2_Monomer1_Mass"]
        - combined.get("Dope2_Monomer2_Mass", 0.0)
        - combined.get("Dope2_Solvent2_Mass", 0.0)
        - combined.get("Dope2_Add_Mass", 0.0),
    )

    # 3) Build one-row DataFrame for model inference and apply feature engineering to enrich with descriptors.
    X_raw = pd.DataFrame([combined])
    x_raw_enriched = feature_engineering(
        X_raw,
        cation_path,
        anion_path,
        verbose=False,
    )


    # 4) Predict with pre-trained models.
    # Resolve from project-level MODEL_PATH so loading is independent of current working directory.
    pwp_model_path = os.path.join(
        MODEL_PATH,
        "Pure_Water_Permeability_{Lm^{-2}^h{-1}bar^{-1}}_extratrees.pkl",
    )
    rej_model_path = os.path.join(
        MODEL_PATH,
        "solute1_Rejection_extratrees.pkl",
    )

    pipe_pwp = joblib.load(pwp_model_path)
    pipe_rej = joblib.load(rej_model_path)

    # Infer expected input columns directly from fitted preprocessors.
    # This removes the need for manually maintained feature_cols_* lists.
    def _extract_expected_input_columns(pipe, model_name: str):
        preprocess = pipe.named_steps.get("preprocess", None)
        if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
            return list(preprocess.feature_names_in_)
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
        raise ValueError(
            f"Could not infer input feature columns from pipeline for {model_name}. "
            "Expected a fitted pipeline with feature_names_in_."
        )

    feature_cols_pwp = _extract_expected_input_columns(pipe_pwp, "PureWaterPermeability")
    feature_cols_rej = _extract_expected_input_columns(pipe_rej, "solute1Rejection")

    # Validate required columns exist in current candidate row before prediction.
    missing_pwp = [c for c in feature_cols_pwp if c not in x_raw_enriched.columns]
    missing_rej = [c for c in feature_cols_rej if c not in x_raw_enriched.columns]
    if missing_pwp:
        raise KeyError(f"Missing columns for PWP model input: {missing_pwp}")
    if missing_rej:
        raise KeyError(f"Missing columns for rejection model input: {missing_rej}")

    # Keep exact training column order for stable inference.
    X_pwp = x_raw_enriched.reindex(columns=feature_cols_pwp)
    X_rej = x_raw_enriched.reindex(columns=feature_cols_rej)

    pwp_pred = pipe_pwp.predict(X_pwp)[0]
    rej_pred = pipe_rej.predict(X_rej)[0]

    # 5) Normalize and scalarize.
    pwp_norm = normalize(pwp_pred, norm_dict, "PureWaterPermeability")
    rej_norm = normalize(rej_pred, norm_dict, "solute1Rejection")
    scalar = scalar_objective(pwp_norm, rej_norm, weight_dict)

    return {
        "normPureWaterPermeability": pwp_norm,
        "solute1Rejection": rej_norm,
        "scalarizedobjective": scalar,
    }

def run_ax_optimization(
    search_space_parameters,
    oracle_from_params,
    constant_params,
    norm_dict,
    weight_dict,
    cation_path,
    anion_path,
    experiment_name="NF_Inverse_Design_Tutorial",
    experiment_description="Toy inverse design of NF IP membrane recipe",
    objective_name="scalarizedobjective",
    sobol_init=5,
    n_trials=20,
    trial_error=0.01,
    verbose=True,
):
    client = Client()

    client.configure_experiment(
        parameters=search_space_parameters,
        name=experiment_name,
        description=experiment_description,
    )

    client.configure_optimization(
        objective=objective_name,
    )

    client.configure_generation_strategy(
        method="fast",
        initialization_budget=sobol_init,
    )

    results = []

    with pd.option_context("future.no_silent_downcasting", False), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Downcasting behavior in `replace` is deprecated.*",
            category=FutureWarning,
            module=r"ax\.adapter\.transforms\.choice_encode",
        )

        for i in range(n_trials):
            trials = client.get_next_trials(max_trials=1)

            for trial_index, trial_params in trials.items():
                if verbose:
                    print(f"\nTrial {trial_index}: parameters = {trial_params}")

                metrics_raw = oracle_from_params(
                    trial_params,
                    constant_params=constant_params,
                    norm_dict=norm_dict,
                    weight_dict=weight_dict,
                    cation_path=cation_path,
                    anion_path=anion_path,
                )

                metrics = {k: (float(v), trial_error) for k, v in metrics_raw.items()}

                if verbose:
                    print("  metrics:", metrics)

                client.complete_trial(
                    trial_index=trial_index,
                    raw_data=metrics,
                )

                results.append(
                    {
                        "trial_index": trial_index,
                        "params": trial_params,
                        "metrics_raw": metrics_raw,
                        "metrics": metrics,
                    }
                )

    return client, results