"""
Shared functions used by every experiment notebook (01_model_expression,
02_model_concatenated, 03_model_mofa).
the only thing that should differ between experiments is the input
feature matrix itself.

Usage in each experiment notebook:
    import common_utils as cu

    cu.CONFIG["debug_mode"] = True  # or False for the real run
    X, y, folds, samples, feature_names = cu.load_reference_cohort(feature_file="expression_filtered.csv")
    results = cu.run_cross_validation(X, y, folds)
    cu.plot_comparison(results)
    cu.save_results("expression", results)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")



# CONFIG
CONFIG = {
    "debug_mode": False,

    "reference_cohort_dir": "reference_cohort", #from 00_build_reference_cohort
    "results_dir": "results",

    "endpoint_event": "OS",
    "endpoint_time": "OS.time",

    "random_state": 42,

    # Elastic-Net Cox settings
    "l1_ratio": 0.9,
    "alpha_min_ratio": 0.01,
    "max_iter": 3000,

    # Evaluation time grid
    "eval_time_quantiles": (0.10, 0.90),
    "n_eval_times": 15,

    "n_top_features_to_plot": 20,
}


def _debug_scaled_config():
    #settings that should shrink in debug mode
    if CONFIG["debug_mode"]:
        return {
            "n_samples_subset": 60,
            "n_top_features": 200,
            "rsf_n_estimators": 20,
            "alpha_subsample": 30,
            "inner_cv_folds": 2,
            "n_folds": 3,
            "permutation_n_features": 50,
            "permutation_n_repeats": 3,
        }
    else:
        return {
            "n_samples_subset": None,
            "n_top_features": 5000,
            "rsf_n_estimators": 200,
            "alpha_subsample": 5,
            "inner_cv_folds": 5,
            "n_folds": 5,
            "permutation_n_features": 300,
            "permutation_n_repeats": 10,
        }


# DATA LOADING
def load_reference_cohort(feature_file=None, feature_files=None):
    """Load the reference cohort's clinical outcome, CV folds, and features."""

    ref_dir = CONFIG["reference_cohort_dir"]
    clin = pd.read_csv(os.path.join(ref_dir, "clinical_outcome.csv"), index_col=0)
    folds = pd.read_csv(os.path.join(ref_dir, "cv_folds.csv"), index_col=0)

    debug_cfg = _debug_scaled_config()

    if feature_files is not None:
        layers = {}
        common = set(clin.index) & set(folds.index)
        loaded = {}
        for name, path in feature_files.items():
            df = pd.read_csv(path, index_col=0)
            common &= set(df.index)
            loaded[name] = df
        common = sorted(common)

        if debug_cfg["n_samples_subset"]:
            common = common[: debug_cfg["n_samples_subset"]]

        for name, df in loaded.items():
            df = df.loc[common]
            if debug_cfg["n_top_features"] and debug_cfg["n_top_features"] < df.shape[1]:
                top_cols = df.var(axis=0).sort_values(ascending=False).head(debug_cfg["n_top_features"]).index
                df = df[top_cols]
            layers[name] = df.values

        clin = clin.loc[common]
        folds = folds.loc[common]
        y = Surv.from_dataframe(CONFIG["endpoint_event"], CONFIG["endpoint_time"], clin)
        fold_arr = folds["fold"].values.astype(int)

        print(f"Loaded {len(layers)} layers, {len(common)} common samples, "
              f"{fold_arr.max() + 1} folds.")
        if CONFIG["debug_mode"]:
            print(f"  (DEBUG_MODE: subsetting each layer to top "
                  f"{debug_cfg['n_top_features']} variance features)")
        return layers, y, fold_arr, common

    else:
        df = pd.read_csv(feature_file, index_col=0)
        common = sorted(set(clin.index) & set(folds.index) & set(df.index))

        if debug_cfg["n_samples_subset"]:
            common = common[: debug_cfg["n_samples_subset"]]

        df = df.loc[common]

        # debug mode: subset to top variance features (to make small debug runs faster)
        if debug_cfg["n_top_features"] and debug_cfg["n_top_features"] < df.shape[1]:
            top_cols = df.var(axis=0).sort_values(ascending=False).head(debug_cfg["n_top_features"]).index
            df = df[top_cols]
        clin = clin.loc[common]
        folds = folds.loc[common]

        y = Surv.from_dataframe(CONFIG["endpoint_event"], CONFIG["endpoint_time"], clin)
        fold_arr = folds["fold"].values.astype(int)

        print(f"Loaded {df.shape[0]} samples x {df.shape[1]} features, "
              f"{fold_arr.max() + 1} folds.")
        if CONFIG["debug_mode"]:
            print(f"  (DEBUG_MODE: subsetting to top {debug_cfg['n_top_features']} variance features)")
        return df.values, y, fold_arr, common, df.columns.values

#build the evaluation time grid directly from a structured survival array
def _eval_times_from_y(y):
    event_field, time_field = y.dtype.names
    event_times = y[time_field][y[event_field]]
    lo = np.quantile(event_times, CONFIG["eval_time_quantiles"][0])
    hi = np.quantile(event_times, CONFIG["eval_time_quantiles"][1])
    return np.linspace(lo, hi, CONFIG["n_eval_times"])


# MODEL FITTING
def fit_coxnet_with_inner_cv(X_train, y_train):
    """Fit the Elastic-Net Cox alpha path once, then select the best
    alpha via inner cross-validation on the training data only."""
    debug_cfg = _debug_scaled_config()

    base = CoxnetSurvivalAnalysis(
        l1_ratio=CONFIG["l1_ratio"], alpha_min_ratio=CONFIG["alpha_min_ratio"],
        max_iter=CONFIG["max_iter"], fit_baseline_model=True,
    )
    base.fit(X_train, y_train)
    alphas_sub = base.alphas_[:: debug_cfg["alpha_subsample"]]

    gcv = GridSearchCV(
        CoxnetSurvivalAnalysis(l1_ratio=CONFIG["l1_ratio"], max_iter=CONFIG["max_iter"],
                                fit_baseline_model=True),
        param_grid={"alphas": [[a] for a in alphas_sub]},
        cv=debug_cfg["inner_cv_folds"], error_score=0.5, n_jobs=1,
    )
    gcv.fit(X_train, y_train)
    return gcv.best_estimator_, gcv.best_params_["alphas"][0]


def fit_rsf(X_train, y_train):
    debug_cfg = _debug_scaled_config()
    rsf = RandomSurvivalForest(
        n_estimators=debug_cfg["rsf_n_estimators"],
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=1,
        random_state=CONFIG["random_state"],
    )
    rsf.fit(X_train, y_train)
    return rsf


# name -> (fit_function, needs_alpha_report)
MODEL_REGISTRY = {
    "coxnet": (fit_coxnet_with_inner_cv, True),
    "rsf": (fit_rsf, False),
}


# EVALUATION
def evaluate_fold(model, X_train, y_train, X_test, y_test, eval_times):
    """Return risk scores, C-index, integrated Brier score, and the
    dynamic AUC curve for one fitted model on one held-out fold."""
    risk = model.predict(X_test)
    event_field, time_field = y_test.dtype.names
    c_index = concordance_index_censored(y_test[event_field], y_test[time_field], risk)[0]
    
    #countering the fact that IBS can fail on small folds with few training samples relative to features
    #so we catch the exception and record NaN for that fold, rather than crashing the whole run
    try:
        surv_funcs = model.predict_survival_function(X_test)
        surv_probs = np.row_stack([f(eval_times) for f in surv_funcs])
        if np.isnan(surv_probs).any():
            raise ValueError("predicted survival function contains NaN "
                              "(likely numerical instability from too few "
                              "training samples relative to features)")
        ibs = integrated_brier_score(y_train, y_test, surv_probs, eval_times)
    except Exception as e:
        print(f"    WARNING: IBS computation failed for this fold ({e}). "
              f"Recording IBS=NaN and continuing.")
        ibs = np.nan

    # cumulative_dynamic_auc can also fail, isolating this the same way as the IBS guard
    try:
        auc_curve, _ = cumulative_dynamic_auc(y_train, y_test, risk, eval_times)
    except Exception as e:
        print(f"    WARNING: dynamic AUC computation failed for this fold ({e}). "
              f"Recording AUC=NaN for this fold and continuing.")
        auc_curve = np.full(len(eval_times), np.nan)

    return risk, c_index, ibs, auc_curve


def run_cross_validation(X, y, fold_arr, eval_times=None, models=("coxnet", "rsf"),
                          feature_builder=None):
    """Run the fixed K-fold CV loop for one or more models."""

    n_folds = fold_arr.max() + 1
    is_multilayer = isinstance(X, dict)
    n_samples = len(fold_arr)

    if eval_times is None:
        eval_times = _eval_times_from_y(y)

# Every test time must be within a range that ALL folds' data can support,
# or the survival/AUC math breaks (NaN or errors). So we compute one safe
# time window that works everywhere before running anything.

    event_field, time_field = y.dtype.names
    train_max_times, test_min_times, test_max_times = [], [], []
    for k in range(n_folds):
        train_idx = np.where(fold_arr != k)[0]
        test_idx = np.where(fold_arr == k)[0]
        train_max_times.append(y[time_field][train_idx].max())
        test_min_times.append(y[time_field][test_idx].min())
        test_max_times.append(y[time_field][test_idx].max())

    safe_lower = max(test_min_times) * 1.001
    safe_upper = min(min(train_max_times), min(test_max_times)) * 0.999

    eval_times = eval_times[(eval_times > safe_lower) & (eval_times < safe_upper)]
    if len(eval_times) < 2:
        # fallback: too few valid points left in the original grid, rebuild
        # a small safe grid directly from the bounds
        if safe_upper <= safe_lower:
            raise ValueError(
                "Could not find a valid evaluation time window across all "
                "folds (safe_lower >= safe_upper)."
            )
        eval_times = np.linspace(safe_lower, safe_upper, 5)

    results = {m: {"c_index": [], "ibs": [], "auc_curves": [], "risk_oof": np.zeros(n_samples)}
               for m in models}

    for k in range(n_folds):
        train_idx = np.where(fold_arr != k)[0]
        test_idx = np.where(fold_arr == k)[0]
        y_train, y_test = y[train_idx], y[test_idx]

        if is_multilayer:
            if feature_builder is None:
                raise ValueError("X is a dict of layers -- provide a feature_builder function.")
            train_layers = {name: arr[train_idx] for name, arr in X.items()}
            test_layers = {name: arr[test_idx] for name, arr in X.items()}
            X_train, X_test = feature_builder(train_layers, test_layers)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

        for model_name in models:
            fit_fn, reports_alpha = MODEL_REGISTRY[model_name]
            if reports_alpha:
                model, best_alpha = fit_fn(X_train, y_train)
                extra = f"  (alpha={best_alpha:.4g})"
            else:
                model = fit_fn(X_train, y_train)
                extra = ""

            risk, c_idx, ibs, auc_curve = evaluate_fold(model, X_train, y_train, X_test, y_test, eval_times)
            results[model_name]["c_index"].append(c_idx)
            results[model_name]["ibs"].append(ibs)
            results[model_name]["auc_curves"].append(auc_curve)
            results[model_name]["risk_oof"][test_idx] = risk
            print(f"[fold {k}] {model_name:8s} C-index={c_idx:.3f}  IBS={ibs:.3f}{extra}")

    results["_eval_times"] = eval_times
    return results


def summarize_results(results):
    print("\n=== Cross-validated performance (mean +/- std across folds) ===")
    for model_name, res in results.items():
        if model_name.startswith("_"):
            continue
        c, b = res["c_index"], res["ibs"]
        n_nan_ibs = np.sum(np.isnan(b))
        nan_note = f"  ({n_nan_ibs} fold(s) had unstable IBS, excluded)" if n_nan_ibs else ""
        print(f"{model_name:8s}  C-index = {np.nanmean(c):.3f} +/- {np.nanstd(c):.3f}   "
              f"IBS = {np.nanmean(b):.3f} +/- {np.nanstd(b):.3f}{nan_note}")


# PLOTTING

def plot_comparison(results, title_suffix="", save_path=None):
    eval_times = results["_eval_times"]
    model_names = [m for m in results if not m.startswith("_")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].boxplot(
        [results[m]["c_index"] for m in model_names],
        tick_labels=model_names,
    )
    axes[0].set_ylabel("C-index (per fold)")
    axes[0].set_title(f"Concordance index across folds{title_suffix}")

    for m in model_names:
        mean_auc = np.nanmean(results[m]["auc_curves"], axis=0)
        axes[1].plot(eval_times, mean_auc, marker="o", label=m)
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Time-dependent AUC")
    axes[1].set_title(f"Mean dynamic AUC across folds{title_suffix}")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.show()


# FEATURE IMPORTANCE (RSF and Elastic Cox Net)
def feature_importance_coxnet(X, y, feature_names, save_prefix=None):
    #Refit Elastic-Net Cox on the full ds for a final set of nonzero coefficients.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model, best_alpha = fit_coxnet_with_inner_cv(X_scaled, y)
    coefs = pd.Series(model.coef_.ravel(), index=feature_names)
    nonzero = coefs[coefs != 0].sort_values(key=np.abs, ascending=False)

    print(f"Elastic-Net Cox (full-data refit): alpha={best_alpha:.4g}, "
          f"{len(nonzero)} nonzero features out of {len(coefs)}")

    if len(nonzero) == 0:
        print("WARNING: Elastic-Net selected ZERO nonzero features")
        return nonzero

    top = nonzero.head(CONFIG["n_top_features_to_plot"])
    plt.figure(figsize=(6, 6))
    top.sort_values().plot(kind="barh")
    plt.xlabel("Cox coefficient (log hazard ratio)")
    plt.title("Top features — Elastic-Net Cox")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_coxnet_importance.png", dpi=150)
        nonzero.to_csv(f"{save_prefix}_coxnet_importance.csv", header=["coefficient"])
    plt.show()
    return nonzero


def feature_importance_rsf(X, y, feature_names, save_prefix=None):
    """Refit RSF on the full dataset, then compute permutation importance
    restricted to a manageable top-variance feature subset (only a subset because otherwise the runtime is very long)"""
    debug_cfg = _debug_scaled_config()
    n_perm_features = min(debug_cfg["permutation_n_features"], X.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    variances = X_scaled.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:n_perm_features]
    X_scaled_subset = X_scaled[:, top_idx]

    # Fit RSF on the SAME subset used for permutation importance below
    rsf = RandomSurvivalForest(
        n_estimators=debug_cfg["rsf_n_estimators"], min_samples_leaf=15,
        max_features="sqrt", n_jobs=1, random_state=CONFIG["random_state"],
    )
    rsf.fit(X_scaled_subset, y)

    perm = permutation_importance(
        rsf, X_scaled_subset, y,
        n_repeats=debug_cfg["permutation_n_repeats"],
        random_state=CONFIG["random_state"], n_jobs=1,
    )
    importances = pd.Series(perm.importances_mean, index=np.array(feature_names)[top_idx]).sort_values(ascending=False)

    print(f"RSF permutation importance computed on top {n_perm_features} "
          f"(by variance) of {X.shape[1]} total features.")

    top = importances.head(CONFIG["n_top_features_to_plot"])
    plt.figure(figsize=(6, 6))
    top.sort_values().plot(kind="barh")
    plt.xlabel("Permutation importance (mean drop in C-index)")
    plt.title("Top features — Random Survival Forest")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_rsf_importance.png", dpi=150)
        importances.to_csv(f"{save_prefix}_rsf_importance.csv", header=["importance"])
    plt.show()
    return importances


# SAVE / LOAD RESULTS (for 04_compare_results.ipynb)

def save_results(experiment_name, results):
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    path = os.path.join(CONFIG["results_dir"], f"results_{experiment_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {path}")

def load_results(experiment_name):
    path = os.path.join(CONFIG["results_dir"], f"results_{experiment_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------
# XGBOOST-COX (optional third model)
# -----------------------------------------------------------------------


import xgboost as xgb
from sksurv.linear_model.coxph import BreslowEstimator


class XGBoostCoxWrapper:
    """Wraps an XGBoost 'survival:cox' booster to match the sksurv model
    interface (.predict() -> risk score, .predict_survival_function() ->
    step functions)"""

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05,
                 reg_lambda=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.booster = None
        self.breslow = None

    def fit(self, X, y):
        event_field, time_field = y.dtype.names
        event = y[event_field]
        time = y[time_field]

        #XGBoost's survival:cox convention: label = time if event occurred,
        # -time if censored (negative sign encodes censoring).
        label = np.where(event, time, -time)

        dtrain = xgb.DMatrix(X, label=label)
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "lambda": self.reg_lambda,
            "seed": self.random_state,
            "verbosity": 0,
        }
        self.booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators)

        #fit the Breslow baseline hazard estimator on the train data linear predictor (risk score)
        linear_predictor = self.booster.predict(dtrain, output_margin=True)
        self.breslow = BreslowEstimator().fit(linear_predictor, event, time)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.booster.predict(dtest, output_margin=True)

    def predict_survival_function(self, X):
        risk = self.predict(X)
        return self.breslow.get_survival_function(risk)

    def get_feature_importance(self, feature_names):
        score_dict = self.booster.get_score(importance_type="gain")
        #XGBoost names features f0, f1, f2,....
        importances = pd.Series(0.0, index=feature_names)
        for fname, score in score_dict.items():
            idx = int(fname[1:])
            importances.iloc[idx] = score
        return importances.sort_values(ascending=False)


def fit_xgboost_cox(X_train, y_train):
    debug_cfg = _debug_scaled_config()
    n_estimators = 50 if CONFIG["debug_mode"] else 200
    model = XGBoostCoxWrapper(n_estimators=n_estimators, random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)
    return model


# Register as a third option 
#   results = cu.run_cross_validation(X, y, fold_arr, models=("coxnet", "rsf", "xgboost"))
MODEL_REGISTRY["xgboost"] = (fit_xgboost_cox, False)


def feature_importance_xgboost(X, y, feature_names, save_prefix=None):
    """Refit XGBoost-Cox on the FULL dataset, use the gain-based feature importance from the booster to rank features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = fit_xgboost_cox(X_scaled, y)
    importances = model.get_feature_importance(feature_names)
    importances = importances[importances > 0]

    print(f"XGBoost-Cox (full-data refit): {len(importances)} features with "
          f"nonzero importance out of {len(feature_names)}")

    if len(importances) == 0:
        print("WARNING: XGBoost selected ZERO features with nonzero gain. ")
        return importances

    top = importances.head(CONFIG["n_top_features_to_plot"])
    plt.figure(figsize=(6, 6))
    top.sort_values().plot(kind="barh")
    plt.xlabel("Gain-based importance")
    plt.title("Top features — XGBoost-Cox")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_xgboost_importance.png", dpi=150)
        importances.to_csv(f"{save_prefix}_xgboost_importance.csv", header=["importance"])
    plt.show()
    return importances
