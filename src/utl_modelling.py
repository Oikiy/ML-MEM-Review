import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
FONT_SIZE = 20

def _build_feature_table(df_raw: pd.DataFrame, base_features: list, target: str, threshold = 0.3):
    """
    An internal function that builds an ML-ready feature table from the raw NF membrane dataset.

    Tutorial notes
    --------------
    1) Keep only rows where the chosen target exists (supervised learning needs labels).
    2) Keep only features that are actually present in this dataset version.
    3) Remove high-missing and constant columns to reduce noise and instability.
    """
    df = df_raw.copy()

    # Keep only rows with a valid target value.
    # This ensures X and y are aligned and avoids NaN labels during fitting.
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Keep only columns that exist in this dataset
    # (important when moving the notebook between batches with slightly different schemas).
    base_features = [c for c in base_features if c in df.columns]

    # Split into features and target
    X = df[base_features].copy()
    y = df[target].copy()

    print(f"Initial feature count: {X.shape[1]}")
    print(f"Initial sample count: {X.shape[0]}")

    # Drop columns with too many missing values
    missing_frac = X.isna().mean()
    high_missing_cols = missing_frac[missing_frac > threshold].index.tolist()
    if high_missing_cols:
        print("Dropping high-missing columns:", high_missing_cols)
        X = X.drop(columns=high_missing_cols)

    # Drop columns that are constant (no variance)
    # Constant columns cannot help tree splits and only increase model complexity.
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)

    print(f"Final feature count: {X.shape[1]}")
    return X, y

def regression_metrics(y_true, y_pred, label=""):
    """
    Compute and print standard regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values.
    label : str, optional
        Label indicating which split ('Train' or 'Test') the metrics belong to.
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label} R²   = {r2:.4f}")
    print(f"{label} RMSE = {rmse:.4f}")
    print(f"{label} MAE  = {mae:.4f}")
    return r2, rmse, mae

def train_and_save_extratrees_model(
    df,
    base_features,
    target,
    model_path,
    test_size=0.2,
    random_state=RANDOM_STATE,
    n_estimators=300,
    use_kfold=False,
    n_splits=5,
    use_hyperparam_tuning=False,
    n_iter_search=20,
):
    """
    Build features, train an ExtraTreesRegressor in a sklearn Pipeline,
    optionally use K-fold CV and hyperparameter tuning, and save the fitted
    pipeline to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe (already enriched by feature_engineering).
    base_features : list of str
        List of candidate feature column names to pass to build_feature_table.
    target : str
        Name of the target column to predict.
    model_path : str
        File path for saving the fitted pipeline (e.g. 'models/IPlayerThicknessnm.pkl').
    test_size : float, optional
        Fraction of data to use for the test split.
    random_state : int, optional
        Random seed for reproducible splits and models.
    n_estimators : int, optional
        Default number of trees for ExtraTreesRegressor when tuning is disabled.
    use_kfold : bool, optional
        If True, K-fold CV is used for tuning/assessment inside RandomizedSearchCV.
    n_splits : int, optional
        Number of folds for KFold when use_kfold is True.
    use_hyperparam_tuning : bool, optional
        If True, perform RandomizedSearchCV over ExtraTrees hyperparameters.
        If False, fit a single ExtraTrees model with fixed n_estimators.
    n_iter_search : int, optional
        Number of parameter settings sampled in RandomizedSearchCV
        when use_hyperparam_tuning is True.

    Returns
    -------
    pipe : sklearn.Pipeline
        Fitted pipeline (preprocessing + model).
    X_train, X_test, y_train, y_test : arrays / DataFrames
        Train/test splits for further analysis (e.g. SHAP).
    """
    # Build ML-ready feature table
    X, y = _build_feature_table(df, base_features=base_features, target=target)

    # Train-test split kept outside CV so we still have an untouched holdout set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n=== Training ExtraTrees model for target: {target} ===")
    print("Training samples:", X_train.shape[0], "using features:", X_train.columns.tolist())
    print("Test samples:", X_test.shape[0])

    # Identify categorical vs numerical features
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical features:", len(categorical_cols))
    print("Numerical features:", len(numerical_cols))

    # Preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    # Base model
    base_model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    # Assemble pipeline
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", base_model),
        ]
    )

    # Option 1: simple fit (no tuning)
    if not use_hyperparam_tuning:
        print("Hyperparameter tuning: OFF (using default ExtraTrees settings).")
        pipe.fit(X_train, y_train)

    # Option 2: hyperparameter tuning (with optional K-fold CV)
    else:
        print("Hyperparameter tuning: ON")
        if use_kfold:
            print(f"Using K-fold CV with n_splits={n_splits}.")
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            print("Using default CV strategy from RandomizedSearchCV (e.g., 5-fold).")
            cv = None  # RandomizedSearchCV default

        param_dist = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        }

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        search.fit(X_train, y_train)

        print(f"Best CV R²: {search.best_score_:.4f}")
        print("Best parameters:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")

        # Replace pipe with best estimator
        pipe = search.best_estimator_

    # Evaluate on train and test
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    print("\nFinal performance on train/test split:")
    _,_,_, = regression_metrics(y_train, y_pred_train, label="Train")
    _,_,_, = regression_metrics(y_test, y_pred_test, label="Test")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Model saved to: {model_path}")

    return pipe, X_train, X_test, y_train, y_test

def evaluate_and_plot_parity(
    target,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    metrics_fn=regression_metrics,
):
    """
    Print regression metrics for train and test sets and plot a parity
    (y_true vs. y_pred) scatter plot for the test set.

    Parameters
    ----------
    target : str
        Name of the target variable (used in printed messages and plot title).
    y_train, y_pred_train : array-like
        True and predicted values for the training set.
    y_test, y_pred_test : array-like
        True and predicted values for the test set.
    metrics_fn : callable, optional
        Function to compute and print metrics, with signature
        metrics_fn(y_true, y_pred, label="").
    """
    print(f"\nPerformance for target: {target}")
    metrics_fn(y_train, y_pred_train, label="Train")
    metrics_fn(y_test, y_pred_test, label="Test")

    plt.figure(figsize=(10,8))
    plt.scatter(y_train, y_pred_train, label='Train Data', color='#8fbce3', alpha=0.6)
    plt.scatter(y_test, y_pred_test, label='Test Data', color='#e38f8f', marker='D')
    plt.plot([min(min(y_test),min(y_train)), max(max(y_test),max(y_train))], [min(min(y_test),min(y_train)), max(max(y_test),max(y_train))], '--', color='gray')
    plt.xlabel(f'True Values', fontsize=FONT_SIZE)  
    plt.ylabel(f'Predictions', fontsize=FONT_SIZE) 
    plt.title(f'Parity Plot for {target}', fontsize=FONT_SIZE, fontweight='bold')
    
    r2_train,rmse_train,_, = regression_metrics(y_train, y_pred_train, label="Train")  
    r2_test,rmse_test,_, = regression_metrics(y_test, y_pred_test, label="Test")
    plt.annotate(f'R$^2$ (Train)={r2_train:.2f}\nRMSE (Train)={rmse_train:.2f}',
                xy=(0.95, 0.12), xycoords='axes fraction', ha='right', va='top',
                fontsize=FONT_SIZE, color="#7FAFD9")
    plt.annotate(f'R$^2$ (Test)={r2_test:.2f}\nRMSE (Test)={rmse_test:.2f}',
                xy=(0.95, 0.12), xycoords='axes fraction', ha='right', va='bottom',
                fontsize=FONT_SIZE, color='#e38f8f')

    ax = plt.gca()
    ax.grid(False)
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=6,
        width=1.5,
        labelsize=FONT_SIZE,
        colors='black',
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    plt.legend()
    plt.show()