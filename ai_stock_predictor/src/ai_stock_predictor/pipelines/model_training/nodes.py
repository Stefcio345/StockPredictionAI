from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    return np.mean(diff) * 100

def split_data(data: pd.DataFrame, target_columns: list, test_days: int = 36):
    data = data.copy()

    # Sort BEFORE shifting
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Create next-day targets
    for col in target_columns:
        data[f"{col}_target"] = data.groupby("Symbol")[col].shift(-1)

    target_label_cols = [f"{c}_target" for c in target_columns]

    # Drop rows without future targets (last row per symbol)
    data = data.dropna(subset=target_label_cols)

    train_parts, test_parts = [], []

    for symbol, group in data.groupby("Symbol", sort=False):
        group = group.sort_values("Date")
        cutoff_date = group["Date"].max() - pd.Timedelta(days=test_days)

        test_mask = group["Date"] > cutoff_date
        test_parts.append(group[test_mask])
        train_parts.append(group[~test_mask])

    train_data = pd.concat(train_parts).reset_index(drop=True)
    test_data = pd.concat(test_parts).reset_index(drop=True)

    return train_data, test_data


def train_multi_target_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_columns: list,
    time_limit: int = 120,
):
    train_data = train_data.copy()
    test_data = test_data.copy()

    predictors = {}

    target_label_cols = [f"{c}_target" for c in target_columns]

    # everything except *_target is a candidate feature
    base_feature_cols = [c for c in train_data.columns if c not in target_label_cols]

    for target in target_columns:
        label_col = f"{target}_target"
        print(f"\n=== Training model for target: {label_col} ===")

        cols_for_this_model = base_feature_cols + [label_col]

        train_df = train_data[cols_for_this_model].dropna(subset=[label_col])
        test_df = test_data[cols_for_this_model].dropna(subset=[label_col])

        predictor = MultiModalPredictor(
            label=label_col,
            problem_type="regression",
        )

        # NO tuning_data → AutoGluon makes its own validation from train_df
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
        )

        metrics = predictor.evaluate(test_df)  # true holdout
        print("Test metrics:", metrics)

        predictor.save(f"data/05_models/{target}")
        predictors[label_col] = predictor

    return predictors


def evaluate_multi_target_models(predictors: dict, test_data: pd.DataFrame, target_columns: list, show_heatmaps: bool = True):
    metrics = {}
    X_test = test_data.drop(columns=[t + "_target" for t in target_columns])

    all_preds = {}
    all_truths = {}

    for target in target_columns:
        full_target = target + "_target"
        y_true = test_data[full_target]
        y_pred = predictors[full_target].predict(X_test)

        # Metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        metrics[f"{target}_RMSE"] = round(rmse, 2)
        metrics[f"{target}_MAE"] = round(mean_absolute_error(y_true, y_pred), 2)
        metrics[f"{target}_R2"] = round(r2_score(y_true, y_pred), 2)
        metrics[f"{target}_MAPE"] = round(np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100, 2)
        metrics[f"{target}_SMAPE"] = round(smape(y_true, y_pred), 2)
        metrics[f"{target}_Correlation"] = round(np.corrcoef(y_true, y_pred)[0, 1], 2)

        all_preds[target] = y_pred
        all_truths[target] = y_true

    # Correlation heatmaps
    if show_heatmaps:
        plot_feature_target_correlation(test_data, target_columns)

    return metrics

def plot_feature_target_correlation(data: pd.DataFrame, target_columns: list):
    target_labels = [f"{col}_target" for col in target_columns]

    # Exclude non-numeric columns and target labels
    numeric_data = data.select_dtypes(include=[np.number])
    feature_cols = [col for col in numeric_data.columns if col not in target_labels]

    for target in target_labels:
        print(f"\n🔍 Correlation with target: {target}")
        corr_series = numeric_data[feature_cols + [target]].corr()[target].drop(target)

        # Print top correlations
        print(corr_series.sort_values(ascending=False).head(10))

        # Plot
        plt.figure(figsize=(10, 6))
        corr_series.sort_values(ascending=False).plot(kind='bar', color='skyblue')
        plt.title(f"Feature Correlation with {target}")
        plt.ylabel("Correlation Coefficient")
        plt.grid(True)
        plt.tight_layout()
        plt.show()