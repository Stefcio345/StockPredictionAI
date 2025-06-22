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

def split_data(data: pd.DataFrame, target_columns: list):
    # Ensure Date is datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Shift targets before splitting
    for col in target_columns:
        data[f"{col}_target"] = data.groupby("Symbol")[col].shift(-1)

    # Drop rows where the target couldn't be computed
    data = data.dropna(subset=[f"{col}_target" for col in target_columns])

    train_parts = []
    test_parts = []

    for symbol, group in data.groupby("Symbol"):
        group = group.sort_values("Date")
        # Get the cutoff date
        cutoff_date = group["Date"].max() - pd.Timedelta(days=36)

        # Split based on cutoff
        test_mask = group["Date"] > cutoff_date
        test_parts.append(group[test_mask])
        train_parts.append(group[~test_mask])

    train_data = pd.concat(train_parts)
    test_data = pd.concat(test_parts)

    return train_data, test_data


def train_multi_target_models(train_data: pd.DataFrame, target_columns: list):
    predictors = {}
    target_labels = [f"{col}_target" for col in target_columns]

    for target in target_columns:
        label_col = f"{target}_target"
        print(f"Training model for target: {target}")

        # Drop the label columns (future targets) of *other* targets
        other_target_labels = [col for col in target_labels if (col != label_col)]
        features = train_data.drop(columns=other_target_labels)

        print(f"Features: {features.columns}")

        predictor = MultiModalPredictor(label=label_col)
        predictors[label_col] = predictor.fit(
            train_data=features,
            time_limit=60
        )
        predictor.save(f"data/05_models/{target}")

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
        metrics[f"{target}_RMSE"] = rmse
        metrics[f"{target}_MAE"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{target}_R2"] = r2_score(y_true, y_pred)
        metrics[f"{target}_MAPE"] = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
        metrics[f"{target}_SMAPE"] = smape(y_true, y_pred)
        metrics[f"{target}_Correlation"] = np.corrcoef(y_true, y_pred)[0, 1]

        all_preds[target] = y_pred
        all_truths[target] = y_true

    # Correlation heatmaps
    if show_heatmaps:
        pred_df = pd.DataFrame(all_preds)
        true_df = pd.DataFrame(all_truths)
        plot_prediction_correlation_matrices(pred_df, true_df)

    return metrics

def plot_prediction_correlation_matrices(pred_df: pd.DataFrame, true_df: pd.DataFrame):
    # Correlation between each target's prediction and ground truth
    print("\n🔍 Correlation between predicted and actual values:")
    combined_corr = pred_df.corrwith(true_df)
    print(combined_corr)

    # 1. Correlation heatmap of predictions
    plt.figure(figsize=(8, 6))
    sns.heatmap(pred_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("📈 Prediction Cross-Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 2. Correlation heatmap of true targets
    plt.figure(figsize=(8, 6))
    sns.heatmap(true_df.corr(), annot=True, cmap="viridis", fmt=".2f")
    plt.title("📊 True Target Cross-Correlation Matrix")
    plt.tight_layout()
    plt.show()