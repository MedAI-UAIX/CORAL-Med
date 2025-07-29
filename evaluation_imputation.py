import pandas as pd
import numpy as np
import os
import logging


def calculate_metrics(original_data, imputed_data, missing_data):
    """
    Calculate imputation performance metrics including MAE, AR and AMAE, with separate metrics for each feature

    Args:
    - original_data: Original complete data
    - imputed_data: Imputed data
    - missing_data: Data with missing values

    Returns:
    - Dictionary containing various metrics
    - Feature-organized metrics DataFrame
    """
    # Create missing value mask (M matrix, 0 for missing, 1 for observed)
    mask = ~missing_data.isna()  # Note the negation to make 1 observed, 0 missing

    # Extract common columns
    common_cols = [col for col in original_data.columns if col in imputed_data.columns and col in missing_data.columns]

    # Separate numerical and categorical features
    categorical_cols = [
        'History_alcoholism', 'family_history', 'ascites', 'Position',
        'number', 'child_pugh', 'Distant_metastasis', 'HBsAg', 'anti-HCV',
        'Capsule', 'Surgical_margins', 'differentiation', 'T_stage',
        'LN_metastasis', 'Vascular_invasion', 'Perineural_invasion', 'MicroVascular_invasion',
        'Cirrhosis', 'shape', 'echo', 'margin', 'thrombi', 'dialation',
        'AP_level', 'PP_level', 'DP_level', 'pattern', 'artery',
        'rim_art', 'vein', 'en_margin', 'nonen_margin', 'marked'
    ]

    # Filter categorical columns that exist in the data
    categorical_cols = [col for col in categorical_cols if col in common_cols]
    # Numerical features are the remaining columns
    numerical_cols = [col for col in common_cols if col not in categorical_cols]

    # Calculate various metrics
    metrics = {}

    # Create feature-organized results list
    feature_metrics = []

    # Calculate MAE for each numerical feature separately
    mae_values = []

    for col in numerical_cols:
        # (1-M), i.e., missing value mask
        missing_idx = ~mask[col]

        # Create metrics dictionary for current feature
        feature_metric = {
            "Feature": col,
            "Type": "Numerical",
            "Missing_Count": missing_idx.sum()
        }

        if missing_idx.sum() > 0:
            # True values and imputed values
            true_values = original_data.loc[missing_idx, col].values
            pred_values = imputed_data.loc[missing_idx, col].values

            # Calculate MAE: ∑(1-mij)*|xij-x̂ij|/∑(1-mij)
            col_mae = np.sum(np.abs(true_values - pred_values)) / missing_idx.sum()
            metrics[f"MAE_{col}"] = col_mae
            mae_values.append(col_mae)
            feature_metric["MAE"] = col_mae

            # Add other useful statistics
            feature_metric["True_Mean"] = np.mean(true_values)
            feature_metric["Pred_Mean"] = np.mean(pred_values)
            feature_metric["True_Std"] = np.std(true_values)
            feature_metric["Pred_Std"] = np.std(pred_values)

        # Add feature metrics to list
        feature_metrics.append(feature_metric)

    # Calculate AR (Accuracy Error) for each categorical feature separately
    ar_values = []

    for col in categorical_cols:
        missing_idx = ~mask[col]

        # Create metrics dictionary for current feature
        feature_metric = {
            "Feature": col,
            "Type": "Categorical",
            "Missing_Count": missing_idx.sum()
        }

        if missing_idx.sum() > 0:
            true_values = original_data.loc[missing_idx, col].values
            pred_values = imputed_data.loc[missing_idx, col].values

            # Calculate AR: (1/∑(1-mij))*∑(1-mij)*I[xij ≠ x̂ij]
            # Note: AR is error rate, not accuracy rate
            incorrect = (true_values != pred_values)
            col_ar = np.sum(incorrect) / missing_idx.sum()
            metrics[f"AR_{col}"] = col_ar
            ar_values.append(col_ar)
            feature_metric["AR"] = col_ar

            # Also calculate accuracy for reference
            accuracy = 1 - col_ar
            metrics[f"Accuracy_{col}"] = accuracy
            feature_metric["Accuracy"] = accuracy

            # Add other useful statistics
            # Calculate accuracy for each category
            unique_values = np.unique(np.concatenate([true_values, pred_values]))
            category_accuracies = {}

            for val in unique_values:
                val_indices = (true_values == val)
                if np.sum(val_indices) > 0:
                    correct_count = np.sum(pred_values[val_indices] == val)
                    category_accuracies[f"Accuracy_for_{val}"] = correct_count / np.sum(val_indices)

            feature_metric["Category_Accuracies"] = category_accuracies

        # Add feature metrics to list
        feature_metrics.append(feature_metric)

    # Calculate overall metrics

    # 1. MAE (Mean Absolute Error) - average metric for numerical features
    if mae_values:
        metrics["Overall_MAE"] = np.mean(mae_values)

    # 2. AR (Accuracy Error) - average metric for categorical features
    if ar_values:
        metrics["Overall_AR"] = np.mean(ar_values)
        metrics["Overall_Categorical_Accuracy"] = 1 - metrics["Overall_AR"]

    # 3. AMAE (Average Mean Absolute Error) - combined metric for numerical and categorical features
    # AMAE(X,X̂) = (1/d)*(∑(j∈Fn) MAE(Xj,X̂j) + ∑(j∈Fc) AR(Xj,X̂j))
    if mae_values or ar_values:
        total_features = len(mae_values) + len(ar_values)
        mae_sum = sum(mae_values) if mae_values else 0
        ar_sum = sum(ar_values) if ar_values else 0
        metrics["AMAE"] = (mae_sum + ar_sum) / total_features

    # Create feature metrics DataFrame
    feature_metrics_df = pd.DataFrame([{k: v for k, v in fm.items() if k != "Category_Accuracies"}
                                       for fm in feature_metrics])

    # For categorical features, create a separate DataFrame for category accuracies
    category_accuracies_list = []
    for fm in feature_metrics:
        if "Category_Accuracies" in fm and fm["Category_Accuracies"]:
            for cat, acc in fm["Category_Accuracies"].items():
                category_accuracies_list.append({
                    "Feature": fm["Feature"],
                    "Category": cat.replace("Accuracy_for_", ""),
                    "Accuracy": acc
                })

    category_accuracies_df = pd.DataFrame(category_accuracies_list) if category_accuracies_list else None

    return metrics, feature_metrics_df, category_accuracies_df


def evaluate_imputation(original_path, imputed_path, missing_path, output_dir):
    """
    Evaluate imputation performance and save results

    Args:
    - original_path: Path to original complete data
    - imputed_path: Path to imputed data
    - missing_path: Path to data with missing values
    - output_dir: Output directory
    """
    logging.info("===== Starting imputation evaluation =====")
    logging.info(f"Original data path: {original_path}")
    logging.info(f"Imputed data path: {imputed_path}")
    logging.info(f"Missing data path: {missing_path}")
    logging.info(f"Output directory: {output_dir}")

    # Read data
    original_data = pd.read_csv(original_path)
    imputed_data = pd.read_csv(imputed_path)
    missing_data = pd.read_csv(missing_path)

    logging.info(f"Original data shape: {original_data.shape}")
    logging.info(f"Imputed data shape: {imputed_data.shape}")
    logging.info(f"Missing data shape: {missing_data.shape}")

    # Calculate metrics
    metrics, feature_metrics_df, category_accuracies_df = calculate_metrics(original_data, imputed_data, missing_data)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})

    # Save overall metrics
    metrics_path = os.path.join(output_dir, "imputation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save feature-level metrics
    feature_metrics_path = os.path.join(output_dir, "feature_level_metrics.csv")
    feature_metrics_df.to_csv(feature_metrics_path, index=False)

    # Save categorical feature category accuracies (if any)
    if category_accuracies_df is not None:
        category_accuracies_path = os.path.join(output_dir, "category_accuracies.csv")
        category_accuracies_df.to_csv(category_accuracies_path, index=False)

    logging.info(f"Imputation evaluation metrics saved to: {metrics_path}")
    logging.info(f"Feature-level metrics saved to: {feature_metrics_path}")
    if category_accuracies_df is not None:
        logging.info(f"Categorical feature category accuracies saved to: {category_accuracies_path}")

    # Print main metrics
    logging.info("\nMain imputation metrics:")
    logging.info(f"AMAE: {metrics.get('AMAE', 'N/A')}")
    logging.info(f"Numerical features MAE: {metrics.get('Overall_MAE', 'N/A')}")
    logging.info(f"Categorical features AR: {metrics.get('Overall_AR', 'N/A')}")
    logging.info(f"Categorical features accuracy: {metrics.get('Overall_Categorical_Accuracy', 'N/A')}")

    # Print feature-level metrics summary
    logging.info("\nFeature-level imputation metrics summary:")

    # Group by feature type and sort output
    numerical_features = feature_metrics_df[feature_metrics_df["Type"] == "Numerical"].sort_values(by="MAE")
    categorical_features = feature_metrics_df[feature_metrics_df["Type"] == "Categorical"].sort_values(by="Accuracy",
                                                                                                       ascending=False)

    # Print numerical feature metrics
    if not numerical_features.empty:
        logging.info("\nNumerical features (sorted by MAE):")
        cols_to_print = ["Feature", "Missing_Count", "MAE"]
        logging.info("\n" + numerical_features[cols_to_print].to_string(index=False))

    # Print categorical feature metrics
    if not categorical_features.empty:
        logging.info("\nCategorical features (sorted by accuracy):")
        cols_to_print = ["Feature", "Missing_Count", "AR", "Accuracy"]
        logging.info("\n" + categorical_features[cols_to_print].to_string(index=False))

    # Output features with most missing values
    top_missing = feature_metrics_df.sort_values(by="Missing_Count", ascending=False).head(5)
    if not top_missing.empty:
        logging.info("\nTop 5 features with most missing values:")
        logging.info("\n" + top_missing[["Feature", "Type", "Missing_Count"]].to_string(index=False))

    logging.info("===== Imputation evaluation completed =====")

    return metrics, feature_metrics_df, category_accuracies_df


if __name__ == "__main__":
    # Set logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # File paths
    original_path = "complete_survival_data_inverse.csv"
    imputed_path = "imputed_survival_data.csv"
    missing_path = "fold_1_train20_index.csv"
    output_dir = " "

    # Evaluate imputation performance
    evaluate_imputation(original_path, imputed_path, missing_path, output_dir)