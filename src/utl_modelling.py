import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target

from utl_config_loader import (
    get_model_dir,
    load_config,
    resolve_features_to_model,
)
from utl_model_arch import make_estimator, normalize_model_key, resolve_param_dist

try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - optional dependency
    BorderlineSMOTE = None
    ImbPipeline = None

try:
    from skopt import BayesSearchCV
except Exception:  # pragma: no cover - optional dependency
    BayesSearchCV = None


RANDOM_STATE = 42
FONT_SIZE = 20


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_feature_table(
    df_raw: pd.DataFrame,
    features_to_model: list,
    target: str,
    missing_value_threshold: float = 0.3,
):
    """Build an ML feature table: subset to configured features and drop target-missing rows.

    Column-quality filtering (high-missingness, constant columns) is intentionally NOT
    performed here: it is done after the train/test split in :func:`train_model` using
    train-only statistics to prevent data leakage.
    """
    df = df_raw.copy()
    df = df.dropna(subset=[target]).reset_index(drop=True)
    features_to_model = [c for c in features_to_model if c in df.columns]

    X = df[features_to_model].copy()
    y = df[target].copy()

    print(f"Initial feature count: {X.shape[1]}")
    print(f"Initial sample count: {X.shape[0]}")
    return X, y


def _drop_low_information_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    missing_value_threshold: float = 0.3,
):
    """Drop high-missingness/constant columns using TRAIN-ONLY statistics.

    Computing these statistics on the training split only avoids leaking information
    from the test set into preprocessing decisions.

    Returns
    -------
    tuple
        (X_train, X_test, list_of_dropped_columns)
    """
    if X_train.shape[1] == 0:
        return X_train, X_test, []

    missing_frac = X_train.isna().mean()
    high_missing_cols = missing_frac[missing_frac > missing_value_threshold].index.tolist()

    nunique = X_train.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()

    dropped_cols = list(dict.fromkeys(high_missing_cols + constant_cols))
    if high_missing_cols:
        print("Dropping high-missing columns (train statistics):", high_missing_cols)
    if constant_cols:
        print("Dropping constant columns (train statistics):", constant_cols)

    if dropped_cols:
        X_train = X_train.drop(columns=dropped_cols)
        X_test = X_test.drop(columns=dropped_cols)

    print(f"Final feature count: {X_train.shape[1]}")
    return X_train, X_test, dropped_cols


def _split_feature_types(df_frame: pd.DataFrame, feature_list: list):
    model_features = [feature for feature in feature_list if feature in df_frame.columns]
    categorical_cols = []
    numerical_cols = []
    for feature in model_features:
        series = df_frame[feature]
        # Non-numeric (object, string[python], category, bool, etc.) -> categorical.
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            categorical_cols.append(feature)
        else:
            numerical_cols.append(feature)
    return model_features, categorical_cols, numerical_cols


def _final_estimator(estimator):
    if hasattr(estimator, "named_steps") and getattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]
    if hasattr(estimator, "steps") and getattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def _infer_task(task: Optional[str], pipeline, y):
    requested_task = (task or "auto").lower()
    if requested_task != "auto":
        if requested_task not in {"classification", "regression"}:
            raise ValueError("task must be 'auto', 'classification', or 'regression'.")
        return requested_task

    if pipeline is not None:
        estimator = _final_estimator(pipeline)
        estimator_type = getattr(estimator, "_estimator_type", None)
        if estimator_type in {"classifier", "regressor"}:
            return estimator_type

    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    y_clean = y_series.dropna()
    if type_of_target(y_clean) in {"binary", "multiclass"}:
        # scikit-learn treats any integer-encoded target as class labels, so a
        # continuous integer target (e.g. counts, scores) would be misdetected
        # as classification. Treat high-cardinality numeric targets as
        # regression and low-cardinality ones (0/1, small class sets) as
        # classification.
        if pd.api.types.is_numeric_dtype(y_clean):
            n_unique = int(y_clean.nunique())
            n_rows = max(int(len(y_clean)), 1)
            if n_unique >= 10 or n_unique / n_rows > 0.25:
                return "regression"
        return "classification"
    return "regression"


def _default_param_dist(task: str, model_selection: str = "extratrees"):
    """Backward-compatible wrapper; prefer :func:`resolve_param_dist`."""
    return resolve_param_dist(model_selection, task)


def _default_model(
    task: str,
    random_state: int,
    n_estimators: int,
    model_selection: str = "extratrees",
):
    """Instantiate the configured model architecture for the given task."""
    return make_estimator(
        model_selection,
        task=task,
        random_state=random_state,
        n_estimators=n_estimators,
    )


def _task_to_filename_label(task: str) -> str:
    task_lower = str(task).lower()
    if task_lower == "regression":
        return "regressor"
    if task_lower == "classification":
        return "classifier"
    return task_lower


def _resolve_model_path(
    model_dir: str,
    model_filename_pattern: str,
    target: str,
    task: str,
    model: str,
) -> str:
    task_label = _task_to_filename_label(task)
    filename = model_filename_pattern.format(target=target, task=task_label, model=model)
    if "{task}" not in model_filename_pattern:
        base_name, extension = os.path.splitext(filename)
        filename = f"{base_name}_{task_label}{extension}"
    return os.path.join(model_dir, filename)


def _build_default_pipeline(
    X_train: pd.DataFrame,
    task: str,
    random_state: int,
    n_estimators: int,
    use_smote: bool = False,
    estimator=None,
    model_selection: str = "extratrees",
):
    _, categorical_cols, numerical_cols = _split_feature_types(X_train, list(X_train.columns))

    transformers = []
    if numerical_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numerical_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable features remain after preprocessing.")

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    model = (
        estimator
        if estimator is not None
        else _default_model(task, random_state, n_estimators, model_selection=model_selection)
    )

    if task == "classification" and use_smote:
        if ImbPipeline is None or BorderlineSMOTE is None:
            raise ImportError("imblearn is required when use_smote=True for classification training.")
        return ImbPipeline(
            steps=[
                ("preprocess", preprocess),
                ("smote", BorderlineSMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def regression_metrics(y_true, y_pred, label="", verbose: bool = True):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    if verbose:
        print(f"{label} R2    = {r2:.4f}")
        print(f"{label} RMSE  = {rmse:.4f}")
        print(f"{label} MAE   = {mae:.4f}")
    return r2, rmse, mae


def classification_metrics(y_true, y_pred, label="", verbose: bool = True):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if verbose:
        print(f"{label} Accuracy         = {accuracy:.4f}")
        print(f"{label} Balanced accuracy = {balanced_accuracy:.4f}")
        print(f"{label} Precision        = {precision:.4f}")
        print(f"{label} Recall           = {recall:.4f}")
        print(f"{label} F1               = {f1:.4f}")
    return accuracy, balanced_accuracy, precision, recall, f1


def fit_pipeline(
    pipe,
    X_train,
    y_train,
    use_hyperparam_tuning: bool,
    use_kfold: bool,
    n_splits: int,
    n_iter_search: int,
    param_dist: dict,
    random_state: int,
    task: str,
    scoring: Optional[str] = None,
    search_cv: str = "randomizedsearchcv",
):
    if not use_hyperparam_tuning:
        print("Hyperparameter tuning: OFF.")
        pipe.fit(X_train, y_train)
        return pipe

    print("Hyperparameter tuning: ON")
    if use_kfold:
        if task == "classification":
            print(f"Using StratifiedKFold with n_splits={n_splits}.")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            print(f"Using KFold with n_splits={n_splits}.")
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        print("Using default CV strategy from RandomizedSearchCV.")
        cv = None

    if scoring is None:
        scoring = "accuracy" if task == "classification" else "r2"

    search_cv_key = (search_cv or "randomizedsearchcv").strip().lower()

    if search_cv_key in {"randomized", "randomizedsearch", "randomizedsearchcv"}:
        print("Search backend: RandomizedSearchCV")
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
    elif search_cv_key in {"grid", "gridsearch", "gridsearchcv"}:
        print("Search backend: GridSearchCV")
        search = GridSearchCV(
            pipe,
            param_grid=param_dist,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )
    elif search_cv_key in {"bayes", "bayesian", "bayesiansearch", "bayesiansearchcv", "bayessearchcv"}:
        if BayesSearchCV is None:
            raise ImportError(
                "BayesSearchCV requested but scikit-optimize is not available. "
                "Install scikit-optimize (skopt) or choose another search_cv backend."
            )
        print("Search backend: BayesSearchCV")
        search = BayesSearchCV(
            pipe,
            search_spaces=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
    else:
        raise ValueError(
            "Unsupported search_cv value. Use one of: "
            "randomizedsearchcv, gridsearchcv, bayessearchcv. "
            f"Received: {search_cv}"
        )

    search.fit(X_train, y_train)

    print(f"Best CV score: {search.best_score_:.4f}")
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    return search.best_estimator_


def _print_train_test_metrics(task: str, y_train, y_pred_train, y_test, y_pred_test):
    print("\nFinal performance on train/test split:")
    if task == "classification":
        classification_metrics(y_train, y_pred_train, label="Train")
        classification_metrics(y_test, y_pred_test, label="Test")
        return

    regression_metrics(y_train, y_pred_train, label="Train")
    regression_metrics(y_test, y_pred_test, label="Test")


def train_model(
    df,
    target,
    pipeline=None,
    config_path=None,
    force_retrain=False,
    param_dist_override=None,
    task="auto",
    use_smote: Optional[bool] = None,
    scoring: Optional[str] = None,
):
    """Train a regression or classification model using a single entry point."""
    config = load_config(config_path=config_path)
    modeling_cfg = config.get("modeling", {})
    feature_engineering_cfg = config.get("feature_engineering", {})

    # Feature columns come from the config resolver (auto mode or explicit list).
    features_to_model = resolve_features_to_model(config, df)

    requested_task = task
    if (requested_task or "auto").lower() == "auto":
        requested_task = modeling_cfg.get("task", "auto")
    resolved_task = _infer_task(requested_task, pipeline, df[target].dropna())

    model_dir = get_model_dir(config)
    selected_model_name = normalize_model_key(
        modeling_cfg.get("model_selection", "extratrees")
    )
    model_filename_pattern = modeling_cfg.get("model_filename_pattern", "{target}_{model}_{task}.pkl")
    model_path = _resolve_model_path(
        model_dir=model_dir,
        model_filename_pattern=model_filename_pattern,
        target=target,
        task=resolved_task,
        model=selected_model_name,
    )

    train_test_ratio = modeling_cfg.get("train_test_ratio", 0.2)
    random_state = modeling_cfg.get("random_state", RANDOM_STATE)
    n_estimators = modeling_cfg.get("n_estimators", 300)
    use_kfold = modeling_cfg.get("use_kfold", False)
    n_splits = int(modeling_cfg.get("cv_folds", modeling_cfg.get("n_splits", 5)))
    use_hyperparam_tuning = modeling_cfg.get("use_hyperparam_tuning", False)
    n_iter_search = modeling_cfg.get("n_iter_search", 20)
    search_cv = modeling_cfg.get("search_cv", "randomizedsearchcv")
    use_smote = modeling_cfg.get("use_smote", False) if use_smote is None else use_smote
    user_param_dist = param_dist_override if param_dist_override is not None else modeling_cfg.get("param_dist")
    param_dist = resolve_param_dist(
        selected_model_name, resolved_task, user_param_dist=user_param_dist
    )
    if use_hyperparam_tuning and not param_dist:
        print(
            f"No hyperparameter search space for model '{selected_model_name}' "
            f"({resolved_task}); training with defaults."
        )
        use_hyperparam_tuning = False
    missing_value_threshold = feature_engineering_cfg.get("missing_value_threshold", 0.3)

    X, y = _build_feature_table(
        df,
        features_to_model=features_to_model,
        target=target,
        missing_value_threshold=missing_value_threshold,
    )

    if resolved_task == "classification" and y.nunique(dropna=True) < 2:
        raise ValueError("Classification training requires at least two target classes.")

    stratify_target = None
    if resolved_task == "classification":
        class_counts = y.value_counts(dropna=True)
        if not class_counts.empty and class_counts.min() >= 2:
            stratify_target = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=train_test_ratio,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError:
        print("Warning: stratified split failed, falling back to an unstratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=train_test_ratio,
            random_state=random_state,
        )

    X_train, X_test, _ = _drop_low_information_columns(
        X_train,
        X_test,
        missing_value_threshold=missing_value_threshold,
    )

    # Cap CV folds so StratifiedKFold / KFold never exceeds available samples.
    if use_kfold:
        max_splits = int(len(y_train))
        if resolved_task == "classification":
            max_splits = int(y_train.value_counts(dropna=True).min())
        if max_splits < 2:
            print("Warning: too few samples per class/fold for CV; disabling k-fold.")
            use_kfold = False
        elif n_splits > max_splits:
            print(
                f"Warning: reducing cv_folds from {n_splits} to {max_splits} "
                "(limited by training samples / smallest class)."
            )
            n_splits = max_splits

    if os.path.exists(model_path) and not force_retrain:
        print(f"\n=== Reusing existing model for target: {target} ===")
        print(f"Loading model from: {model_path}")
        pipe = joblib.load(model_path)
    else:
        print(
            f"\n=== Training {resolved_task} model ({selected_model_name}) "
            f"for target: {target} ==="
        )
        print("Training samples:", X_train.shape[0], "using features:", X_train.columns.tolist())
        print("Test samples:", X_test.shape[0])

        pipe = pipeline
        if pipe is None:
            pipe = _build_default_pipeline(
                X_train=X_train,
                task=resolved_task,
                random_state=random_state,
                n_estimators=n_estimators,
                use_smote=use_smote,
                model_selection=selected_model_name,
            )
        elif not hasattr(pipe, "named_steps") and not hasattr(pipe, "steps"):
            pipe = _build_default_pipeline(
                X_train=X_train,
                task=resolved_task,
                random_state=random_state,
                n_estimators=n_estimators,
                use_smote=use_smote,
                estimator=pipe,
                model_selection=selected_model_name,
            )

        print(
            "Pipeline received:",
            type(pipe).__name__,
            "| estimator:",
            type(pipe.named_steps.get("model", pipe)).__name__,
        )

        pipe = fit_pipeline(
            pipe=pipe,
            X_train=X_train,
            y_train=y_train,
            use_hyperparam_tuning=use_hyperparam_tuning,
            use_kfold=use_kfold,
            n_splits=n_splits,
            n_iter_search=n_iter_search,
            param_dist=param_dist,
            random_state=random_state,
            task=resolved_task,
            scoring=scoring,
            search_cv=search_cv,
        )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipe, model_path)
        print(f"Model saved to: {model_path}")

    # Metrics are printed once by evaluate_and_plot_results / evaluate_and_plot_parity.
    # Keep the historical 5-tuple return for the teaching notebook.
    return pipe, X_train, X_test, y_train, y_test


def evaluate_and_plot_results(
    target,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    task="auto",
    fig_size=(10, 8),
    config=None,
):
    """Task-aware evaluation plot (parity for regression, confusion for classification).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (caller should save it; do not rely on plt.gcf()).
    str
        Resolved task name (``regression`` or ``classification``).
    """
    if config is None:
        config = load_config()

    requested = task
    if (requested or "auto").lower() == "auto":
        requested = config.get("modeling", {}).get("task", "auto")
    resolved_task = _infer_task(requested, None, y_train)

    print(f"\nPerformance for target: {target} ({resolved_task})")
    if resolved_task == "classification":
        classification_metrics(y_train, y_pred_train, label="Train")
        classification_metrics(y_test, y_pred_test, label="Test")
        fig = _plot_confusion_matrices(
            target, y_train, y_pred_train, y_test, y_pred_test, fig_size=fig_size, config=config
        )
    else:
        regression_metrics(y_train, y_pred_train, label="Train")
        regression_metrics(y_test, y_pred_test, label="Test")
        fig = _plot_parity(
            target, y_train, y_pred_train, y_test, y_pred_test, fig_size=fig_size, config=config
        )
    return fig, resolved_task


def evaluate_and_plot_parity(
    target,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    fig_size=(10, 8),
    metrics_fn=None,
    config=None,
    task="auto",
):
    """Backward-compatible entry point; dispatches on task and returns the figure."""
    if metrics_fn is classification_metrics:
        task = "classification"
    elif metrics_fn is regression_metrics:
        task = "regression"
    fig, _ = evaluate_and_plot_results(
        target=target,
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        task=task,
        fig_size=fig_size,
        config=config,
    )
    return fig


def _plot_parity(
    target,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    fig_size=(10, 8),
    config=None,
):
    plotting_cfg = config.get("plotting", {}) if config is not None else {}
    font_size = plotting_cfg.get("font_size", FONT_SIZE)
    colors = plotting_cfg.get("custom_colormap", {})
    train_color = colors.get("colors", ["#8fbce3"])[0]
    test_color = colors.get("colors", ["#e38f8f"])[-1]
    annotation_color = colors.get("scatter", "#7FAFD9")

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(y_train, y_pred_train, label="Train Data", color=train_color, alpha=0.6)
    ax.scatter(y_test, y_pred_test, label="Test Data", color=test_color, marker="D")
    lo = min(min(y_test), min(y_train))
    hi = max(max(y_test), max(y_train))
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("True Values", fontsize=font_size)
    ax.set_ylabel("Predictions", fontsize=font_size)
    ax.set_title(f"Parity Plot for {target}", fontsize=font_size, fontweight="bold")

    r2_train, rmse_train, _ = regression_metrics(y_train, y_pred_train, label="Train", verbose=False)
    r2_test, rmse_test, _ = regression_metrics(y_test, y_pred_test, label="Test", verbose=False)
    ax.annotate(
        f"R$^2$ (Train)={r2_train:.2f}\nRMSE (Train)={rmse_train:.2f}",
        xy=(0.95, 0.12),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=font_size,
        color=annotation_color,
    )
    ax.annotate(
        f"R$^2$ (Test)={r2_test:.2f}\nRMSE (Test)={rmse_test:.2f}",
        xy=(0.95, 0.12),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=font_size,
        color=test_color,
    )
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1.5,
        labelsize=font_size,
        colors="black",
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_confusion_matrices(
    target,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    fig_size=(12, 5),
    config=None,
):
    plotting_cfg = config.get("plotting", {}) if config is not None else {}
    font_size = plotting_cfg.get("font_size", FONT_SIZE)

    labels = list(pd.unique(pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)))
    # Stable order: sorted when comparable, else first-seen order.
    try:
        labels = sorted(labels)
    except TypeError:
        pass

    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    for ax, y_true, y_pred, title in (
        (axes[0], y_train, y_pred_train, "Train"),
        (axes[1], y_test, y_pred_test, "Test"),
    ):
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{title} confusion — {target}", fontsize=font_size, fontweight="bold")
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([str(label) for label in labels], fontsize=max(8, font_size - 6))
        ax.set_yticklabels([str(label) for label in labels], fontsize=max(8, font_size - 6))
        ax.set_xlabel("Predicted", fontsize=font_size)
        ax.set_ylabel("True", fontsize=font_size)
        thresh = matrix.max() / 2.0 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=max(8, font_size - 4),
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig