#This file is migrated from the notebook to here by IA and myself (as it required cleaning and avoiding bifurcations from the notebook).

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from get_data import save_finished_games_csv


CLASS_ORDER = ["win", "loss", "draw"]
CLASS_TO_INT = {"win": 0, "loss": 1, "draw": 2}
INT_TO_CLASS = {0: "win", 1: "loss", 2: "draw"}
LOG_LOSS_LABEL_ORDER = sorted(CLASS_ORDER)
XGB_OBJECTIVE = "multi:softprob"
DROP_COLUMNS = [
    "tournament_name",
    "white_result",
    "black_result",
    "white_points",
    "black_points",
    "black_outcome",
    "game_opening",
]
TARGET_COLUMN = "white_outcome"
LOW_VARIANCE_THRESHOLD = 0.01
FINISHED_DATA_PATH = Path("titled_tuesday_games_finished.csv")


class RuleBasedEloModel(BaseEstimator, ClassifierMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RuleBasedEloModel":
        _ = (X, y)
        self.classes_ = np.array(CLASS_ORDER)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rating_diff = X["rating_diff"].to_numpy()
        return np.select(
            [rating_diff > 200, rating_diff < -200],
            ["win", "loss"],
            default="draw",
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rating_diff = X["rating_diff"].to_numpy()
        proba = np.full((len(rating_diff), 3), 1e-6)

        win_mask = rating_diff > 200
        loss_mask = rating_diff < -200
        draw_mask = ~(win_mask | loss_mask)

        proba[win_mask] = [0.80, 0.10, 0.10]
        proba[loss_mask] = [0.10, 0.80, 0.10]
        proba[draw_mask] = [0.20, 0.20, 0.60]

        return proba


def fetch_and_prepare_data(output_path: Path = FINISHED_DATA_PATH) -> pd.DataFrame:
    print("=== FETCHING DATA ===")
    dataframe = save_finished_games_csv(output_path=output_path, verbose=True)
    if dataframe.empty:
        raise ValueError("No games were returned by get_data.py.")
    print(f"Data shape: {dataframe.shape}")
    print()
    return dataframe


def prepare_feature_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    feature_frame = dataframe.drop(columns=DROP_COLUMNS + [TARGET_COLUMN]).copy()
    numeric_feature_frame = feature_frame.select_dtypes(include=[np.number, "bool"]).copy()
    feature_names = numeric_feature_frame.columns.tolist()
    excluded_non_numeric = [
        column for column in feature_frame.columns if column not in numeric_feature_frame.columns
    ]

    vt = VarianceThreshold(threshold=LOW_VARIANCE_THRESHOLD)
    vt.fit(numeric_feature_frame)
    low_variance_features = [
        feature_names[index]
        for index, keep in enumerate(vt.get_support())
        if not keep
    ]

    numeric_feature_frame = numeric_feature_frame.drop(columns=low_variance_features)
    selected_features = numeric_feature_frame.columns.tolist()

    return numeric_feature_frame, selected_features, excluded_non_numeric, low_variance_features


def split_data(
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, pd.DataFrame | pd.Series | str | int]:
    test_tournament = dataframe.sort_values("tournament_index")["tournament_name"].iloc[-1]
    test = dataframe[dataframe["tournament_name"] == test_tournament].copy()
    train_validation = dataframe[dataframe["tournament_name"] != test_tournament].copy()

    if train_validation.empty:
        raise ValueError("At least two tournaments are required to create train/validation/test splits.")

    split_round = round(train_validation["round_index"].max() * 0.8)
    train_mask = train_validation["round_index"] < split_round
    val_mask = train_validation["round_index"] >= split_round
    test_mask = test["round_index"] >= 1

    x_train = train_validation.loc[train_mask, feature_names].copy()
    y_train = train_validation.loc[train_mask, TARGET_COLUMN].copy()
    x_val = train_validation.loc[val_mask, feature_names].copy()
    y_val = train_validation.loc[val_mask, TARGET_COLUMN].copy()
    x_test = test.loc[test_mask, feature_names].copy()
    y_test = test.loc[test_mask, TARGET_COLUMN].copy()

    if x_train.empty or x_val.empty or x_test.empty:
        raise ValueError("One of the train/validation/test splits is empty. Check tournament coverage.")

    return {
        "test_tournament": test_tournament,
        "split_round": split_round,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def build_model_configs(y_tr: pd.Series) -> dict[str, tuple[object, pd.Series, dict[str, np.ndarray], bool]]:
    sw_bal = compute_sample_weight(class_weight="balanced", y=y_tr)
    y_tr_xgb = y_tr.map(CLASS_TO_INT)

    return {
        "Rule-based Elo": (RuleBasedEloModel(), y_tr, {}, False),
        "LR (baseline)": (
            SkPipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
            ], memory=None),
            y_tr,
            {},
            False,
        ),
        "RF (baseline)": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
            ),
            y_tr,
            {},
            False,
        ),
        "GBM (baseline)": (
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            ),
            y_tr,
            {},
            False,
        ),
        "XGB (baseline)": (
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=XGB_OBJECTIVE,
                eval_metric="mlogloss",
                random_state=42,
            ),
            y_tr_xgb,
            {},
            True,
        ),
        "LR (balanced)": (
            SkPipeline([
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ], memory=None),
            y_tr,
            {},
            False,
        ),
        "RF (balanced)": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
            ),
            y_tr,
            {},
            False,
        ),
        "GBM (sample_weight)": (
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            ),
            y_tr,
            {"sample_weight": sw_bal},
            False,
        ),
        "XGB (sample_weight)": (
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=XGB_OBJECTIVE,
                eval_metric="mlogloss",
                random_state=42,
            ),
            y_tr_xgb,
            {"sample_weight": sw_bal},
            True,
        ),
        "LR + ADASYN": (
            ImbPipeline([
                ("adasyn", ADASYN(random_state=42, sampling_strategy="minority", n_neighbors=3)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
            ], memory=None),
            y_tr,
            {},
            False,
        ),
        "RF + ADASYN": (
            ImbPipeline([
                ("adasyn", ADASYN(random_state=42, sampling_strategy="minority", n_neighbors=3)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        random_state=42,
                    ),
                ),
            ]),
            y_tr,
            {},
            False,
        ),
        "GBM + ADASYN": (
            ImbPipeline([
                ("adasyn", ADASYN(random_state=42, sampling_strategy="minority", n_neighbors=3)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=42,
                    ),
                ),
            ]),
            y_tr,
            {},
            False,
        ),
        "XGB + ADASYN": (
            ImbPipeline([
                ("adasyn", ADASYN(random_state=42, sampling_strategy="minority", n_neighbors=3)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective=XGB_OBJECTIVE,
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]),
            y_tr_xgb,
            {},
            True,
        ),
    }


def evaluate_split(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    is_xgb: bool,
    split_name: str,
) -> dict[str, object]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    if is_xgb:
        y_pred = pd.Series(y_pred).map(INT_TO_CLASS).to_numpy()
        xgb_class_order = [INT_TO_CLASS[index] for index in sorted(INT_TO_CLASS)]
        y_prob = pd.DataFrame(y_prob, columns=xgb_class_order)
    else:
        model_class_order = list(getattr(model, "classes_", []))
        if not model_class_order and hasattr(model, "named_steps") and "model" in model.named_steps:
            model_class_order = list(model.named_steps["model"].classes_)
        y_prob = pd.DataFrame(y_prob, columns=model_class_order)

    y_prob = y_prob.reindex(columns=LOG_LOSS_LABEL_ORDER)

    report = classification_report(
        y,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    return {
        "split": split_name,
        "pred": y_pred,
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_prob.to_numpy(), labels=LOG_LOSS_LABEL_ORDER),
        "macro_f1": f1_score(y, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0),
        "win_rec": report["win"]["recall"],
        "loss_rec": report["loss"]["recall"],
        "draw_rec": report["draw"]["recall"],
    }


def train_and_benchmark(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, object], pd.DataFrame, pd.DataFrame]:
    val_results: dict[str, dict[str, object]] = {}
    test_results: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, object] = {}

    model_configs = build_model_configs(y_train)
    for name, (model, y_fit, fit_kwargs, is_xgb) in model_configs.items():
        model.fit(x_train, y_fit, **fit_kwargs)
        fitted_models[name] = model

        val_result = evaluate_split(model, x_val, y_val, is_xgb=is_xgb, split_name="validation")
        test_result = evaluate_split(model, x_test, y_test, is_xgb=is_xgb, split_name="test")

        val_result["name"] = name
        test_result["name"] = name
        val_results[name] = val_result
        test_results[name] = test_result

    metric_columns = ["name", "accuracy", "log_loss", "macro_f1", "weighted_f1", "win_rec", "loss_rec", "draw_rec"]
    val_table = pd.DataFrame(val_results.values())[metric_columns].sort_values(
        ["macro_f1", "log_loss"], ascending=[False, True]
    ).reset_index(drop=True)
    test_table = pd.DataFrame(test_results.values())[metric_columns].sort_values(
        ["macro_f1", "log_loss"], ascending=[False, True]
    ).reset_index(drop=True)

    selected_name = val_table.iloc[0]["name"]
    selected_artifact = {
        "name": selected_name,
        "model": fitted_models[selected_name],
        "is_xgb": model_configs[selected_name][3],
        "val_metrics": val_results[selected_name],
        "test_metrics": test_results[selected_name],
    }

    return val_results, test_results, selected_artifact, val_table, test_table


def compute_feature_importance(
    model: object,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    is_xgb: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    y_importance = y_val.map(CLASS_TO_INT) if is_xgb else y_val
    perm_result = permutation_importance(
        model,
        x_val,
        y_importance,
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_importance = pd.DataFrame(
        {
            "feature": x_val.columns,
            "importance_mean": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    native_model = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
    native_importance = None
    if hasattr(native_model, "feature_importances_"):
        native_importance = pd.DataFrame(
            {
                "feature": x_val.columns,
                "importance": native_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    return perm_importance, native_importance


def print_split_summary(
    dataframe: pd.DataFrame,
    split_info: dict[str, pd.DataFrame | pd.Series | str | int],
    excluded_non_numeric: list[str],
    low_variance_features: list[str],
) -> None:
    x_train = split_info["x_train"]
    x_val = split_info["x_val"]
    x_test = split_info["x_test"]
    y_train = split_info["y_train"]
    y_val = split_info["y_val"]
    y_test = split_info["y_test"]
    test_tournament = split_info["test_tournament"]
    split_round = split_info["split_round"]

    print("=== DATA SUMMARY ===")
    print(f"Rows: {len(dataframe)}")
    print(f"Train/validation tournaments: {sorted(dataframe.loc[dataframe['tournament_name'] != test_tournament, 'tournament_name'].unique())}")
    print(f"Held-out test tournament: {test_tournament}")
    print(f"Non-numeric features excluded: {len(excluded_non_numeric)}")
    print(f"Low-variance features dropped: {len(low_variance_features)}")
    if low_variance_features:
        print(f"Dropped low-variance columns: {low_variance_features}")
    print()

    print("=== SPLIT SUMMARY ===")
    print(f"Train:      {len(x_train)} games (round_index < {split_round})")
    print(f"Validation: {len(x_val)} games (round_index >= {split_round})")
    print(f"Test:       {len(x_test)} games ({test_tournament})")
    print(f"Train class distribution: {dict(y_train.value_counts())}")
    print(f"Validation class distribution: {dict(y_val.value_counts())}")
    print(f"Test class distribution: {dict(y_test.value_counts())}")
    print()


def print_metric_table(title: str, table: pd.DataFrame) -> None:
    print(title)
    printable = table.copy()
    numeric_columns = printable.select_dtypes(include="number").columns
    printable[numeric_columns] = printable[numeric_columns].round(4)
    print(printable.to_string(index=False))
    print()


def main() -> None:
    dataframe = fetch_and_prepare_data()
    _, feature_names, excluded_non_numeric, low_variance_features = prepare_feature_frame(dataframe)
    split_info = split_data(dataframe, feature_names)

    print_split_summary(dataframe, split_info, excluded_non_numeric, low_variance_features)

    _, _, selected_artifact, val_table, test_table = train_and_benchmark(
        split_info["x_train"],
        split_info["y_train"],
        split_info["x_val"],
        split_info["y_val"],
        split_info["x_test"],
        split_info["y_test"],
    )

    print_metric_table("=== VALIDATION RESULTS ===", val_table)
    print_metric_table("=== TEST RESULTS ===", test_table)

    selected_name = selected_artifact["name"]
    best_test_name = test_table.iloc[0]["name"]
    selected_val_metrics = selected_artifact["val_metrics"]
    selected_test_metrics = selected_artifact["test_metrics"]

    print("=== SELECTED MODEL ===")
    print(f"Validation-selected model: {selected_name}")
    print(
        "Validation macro F1 / log loss: "
        f"{selected_val_metrics['macro_f1']:.4f} / {selected_val_metrics['log_loss']:.4f}"
    )
    print(
        "Selected model on test macro F1 / log loss: "
        f"{selected_test_metrics['macro_f1']:.4f} / {selected_test_metrics['log_loss']:.4f}"
    )
    print(f"Best realized test model: {best_test_name}")
    print()

    perm_importance, native_importance = compute_feature_importance(
        selected_artifact["model"],
        split_info["x_val"],
        split_info["y_val"],
        is_xgb=selected_artifact["is_xgb"],
    )

    print("=== TOP PERMUTATION IMPORTANCES ===")
    print(perm_importance.head(10).round(4).to_string(index=False))
    print()

    if native_importance is not None:
        print("=== TOP NATIVE FEATURE IMPORTANCES ===")
        print(native_importance.head(10).round(4).to_string(index=False))
        print()


if __name__ == "__main__":
    main()