"""
Refactored imputation main execution script

Execute missing value imputation for survival analysis datasets using modular imputation system
"""
import os
import sys
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add module path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import refactored modules
from config import DataConfig, TrainingConfig, ImputerConfig
from data_preprocessing import DataPreprocessor
from imputer_core import ModifiedNeuralGradFlowImputer
from utils import (
    enable_reproducible_results,
    setup_logging,
    copy_code_folder,
    save_parameters,
    create_timestamp_dir,
    ProgressReporter,
    ConfigValidator
)
from memory_utils import get_memory_manager, memory_efficient_context

# Import evaluation module (if exists)
try:
    import evaluation_imputation
    HAS_EVALUATION = True
except ImportError:
    HAS_EVALUATION = False
    logging.warning("Evaluation module not found, will skip evaluation step")

import warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Survival Analysis Dataset Missing Value Imputation - Refactored Version')

    # Data path parameters
    parser.add_argument('--complete_data', type=str,
                        default='fold_1_train.csv',
                        help='Complete data file path')
    parser.add_argument('--missing_data', type=str,
                        default='fold_1_train20_index.csv',
                        help='Missing data file path')
    parser.add_argument('--output_dir', type=str,
                        default=' ',
                        help='Output directory')

    # Basic training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=30, help='Imputation iteration count')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')

    # Model parameters
    parser.add_argument('--bandwidth', type=float, default=7.5, help='Kernel function bandwidth')
    parser.add_argument('--entropy_reg', type=float, default=15, help='Entropy regularization coefficient')
    parser.add_argument('--score_net_epoch', type=int, default=2000, help='Score network training epochs')
    parser.add_argument('--score_net_lr', type=float, default=5e-4, help='Score network learning rate')
    parser.add_argument('--mlp_hidden', type=str, default='1024,512,256,128,64',
                        help='MLP hidden layer structure, comma-separated')

    # Control parameters
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--backup_code', action='store_true', default=True, help='Whether to backup code folder')
    parser.add_argument('--config_preset', type=str, default='default',
                        choices=['default', 'fast', 'accurate', 'memory_efficient'],
                        help='Preset configuration type')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    return parser.parse_args()


def create_configs(args) -> tuple:
    """
    Create configuration objects from command line arguments

    Returns:
    - (data_config, training_config, imputer_config)
    """
    # Data configuration
    data_config = DataConfig(
        complete_data_path=args.complete_data,
        missing_data_path=args.missing_data,
        output_dir=args.output_dir
    )

    # Training configuration
    training_config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bandwidth=args.bandwidth,
        entropy_reg=args.entropy_reg,
        score_net_epoch=args.score_net_epoch,
        score_net_lr=args.score_net_lr,
        mlp_hidden=args.mlp_hidden,
        skip_evaluation=args.skip_evaluation,
        backup_code=args.backup_code
    )

    # Parse MLP hidden layer structure
    mlp_hidden = [int(x) for x in args.mlp_hidden.split(',')]

    # Imputer configuration
    imputer_config = ImputerConfig(
        entropy_reg=args.entropy_reg,
        bandwidth=args.bandwidth,
        score_net_epoch=args.score_net_epoch,
        niter=args.epochs,
        mlp_hidden=mlp_hidden,
        lr=args.lr,
        score_net_lr=args.score_net_lr,
        batchsize=args.batch_size
    )

    return data_config, training_config, imputer_config


def validate_and_adjust_configs(data_config, training_config, imputer_config):
    """Validate and adjust configurations"""
    # Validate paths
    if not os.path.exists(data_config.complete_data_path):
        raise FileNotFoundError(f"Complete data file does not exist: {data_config.complete_data_path}")

    if not os.path.exists(data_config.missing_data_path):
        raise FileNotFoundError(f"Missing data file does not exist: {data_config.missing_data_path}")

    # Validate imputer configuration
    imputer_params = {
        'lr': imputer_config.lr,
        'bandwidth': imputer_config.bandwidth,
        'entropy_reg': imputer_config.entropy_reg,
        'niter': imputer_config.niter,
        'batchsize': imputer_config.batchsize,
        'score_net_epoch': imputer_config.score_net_epoch
    }

    validated_params = ConfigValidator.validate_imputer_config(imputer_params)

    # Update configuration
    for key, value in validated_params.items():
        if hasattr(imputer_config, key):
            setattr(imputer_config, key, value)

    return data_config, training_config, imputer_config


def setup_experiment(args):
    """Setup experiment environment"""
    # Create timestamp directory
    run_dir = create_timestamp_dir(args.output_dir, "run")

    # Setup logging
    logger = setup_logging(run_dir)

    # Record start time
    start_time = datetime.datetime.now()
    logging.info(f"Experiment start time: {start_time}")
    logging.info(f"Output directory: {run_dir}")

    # Backup code folder
    if args.backup_code:
        copy_code_folder(str(Path(__file__).parent), run_dir)

    # Set random seed
    enable_reproducible_results(args.seed)

    return run_dir, logger, start_time


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()

    # Setup experiment environment
    run_dir, logger, start_time = setup_experiment(args)

    try:
        # Create configurations
        data_config, training_config, imputer_config = create_configs(args)

        # Validate and adjust configurations
        data_config, training_config, imputer_config = validate_and_adjust_configs(
            data_config, training_config, imputer_config
        )

        # Save configurations
        all_configs = {
            "command_line_args": vars(args),
            "data_config": data_config.__dict__,
            "training_config": training_config.__dict__,
            "imputer_config": imputer_config.__dict__
        }
        save_parameters(all_configs, os.path.join(run_dir, "configurations.json"))

        # Initialize memory management
        memory_manager = get_memory_manager(imputer_config.device)

        # Execute main pipeline in memory efficient context
        with memory_efficient_context(imputer_config.device, "main_pipeline"):
            # Execute data preprocessing and imputation
            execute_imputation_pipeline(
                data_config, training_config, imputer_config, run_dir, args.verbose
            )

        # Record completion time
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        logging.info(f"Experiment completion time: {end_time}")
        logging.info(f"Total runtime: {total_time}")
        logging.info(f"All results saved in: {run_dir}")

    except Exception as e:
        logging.error(f"Experiment execution failed: {e}", exc_info=True)
        raise


def execute_imputation_pipeline(data_config, training_config, imputer_config, run_dir, verbose):
    """Execute imputation pipeline"""
    progress = ProgressReporter(6, 1)  # 6 main steps
    progress.start()

    # Step 1: Load and preprocess data
    progress.step("Loading data")
    logging.info("Starting data preprocessing...")

    preprocessor = DataPreprocessor(data_config)
    df_complete, df_missing = preprocessor.load_data()

    # Step 2: Extract features and metadata
    progress.step("Extracting features")
    feature_cols, meta_data, continuous_cols = preprocessor.extract_features(df_missing)

    # Get data to be imputed
    X_missing = df_missing[feature_cols].copy()
    mask = X_missing.isna()

    # Step 3: Data transformation and standardization
    progress.step("Data transformation")
    # Handle problem features
    X_missing = preprocessor.handle_problem_features(df_complete, X_missing)

    # Standardize data
    X_scaled = preprocessor.standardize_data(df_complete, X_missing, continuous_cols, mask)

    # Prepare categorical variable information
    preprocessor.prepare_categorical_info(df_complete, feature_cols, mask)

    # Step 4: Initialize and configure imputer
    progress.step("Initializing imputer")
    categorical_indices = preprocessor.get_categorical_indices(feature_cols)

    # Update imputer configuration
    imputer_config.categorical_indices = categorical_indices

    # Create imputer
    imputer = ModifiedNeuralGradFlowImputer(
        categorical_indices=imputer_config.categorical_indices,
        entropy_reg=imputer_config.entropy_reg,
        bandwidth=imputer_config.bandwidth,
        score_net_epoch=imputer_config.score_net_epoch,
        niter=imputer_config.niter,
        mlp_hidden=imputer_config.mlp_hidden,
        lr=imputer_config.lr,
        score_net_lr=imputer_config.score_net_lr,
        batchsize=imputer_config.batchsize,
        device=imputer_config.device
    )

    # Set feature names
    imputer.feature_names = feature_cols

    # Step 5: Execute imputation
    progress.step("Executing imputation")
    logging.info("Starting data imputation...")

    with memory_efficient_context(imputer_config.device, "imputation_process"):
        X_imputed = imputer.fit_transform(
            X_scaled.values,
            categorical_indices=categorical_indices,
            verbose=verbose
        )

    # Step 6: Post-processing and save results
    progress.step("Post-processing and saving")

    # Convert to DataFrame
    df_imputed = pd.DataFrame(
        X_imputed.numpy() if hasattr(X_imputed, 'numpy') else X_imputed,
        columns=feature_cols
    )

    # Inverse transform data
    df_imputed = preprocessor.inverse_transform_data(df_imputed, feature_cols, continuous_cols, mask)

    # Process categorical variable results
    df_imputed = preprocessor.process_categorical_results(df_complete, df_imputed, mask)

    # Apply domain constraints
    df_imputed = preprocessor.apply_domain_constraints(df_complete, df_imputed, mask)

    # Merge metadata
    if meta_data is not None:
        df_imputed = pd.concat([meta_data.reset_index(drop=True), df_imputed.reset_index(drop=True)], axis=1)

    # Save results
    save_results(df_imputed, df_complete, preprocessor, run_dir, training_config, verbose)

    progress.finish()


def save_results(df_imputed, df_complete, preprocessor, run_dir, training_config, verbose):
    """Save imputation results"""
    # Save imputed data
    output_path = os.path.join(run_dir, "imputed_survival_data.csv")
    df_imputed.to_csv(output_path, index=False)
    logging.info(f"Imputation results saved to: {output_path}")

    # Save processed complete data (for evaluation)
    df_complete_processed = df_complete.copy()

    # Apply same transformations to complete data
    for col in preprocessor.config.problem_cols:
        if col in df_complete_processed.columns and col in preprocessor.transform_types:
            transform_type = preprocessor.transform_types[col]
            offset = preprocessor.transform_offsets.get(col, 0)

            # Apply same transformation and inverse transformation
            if offset > 0:
                df_complete_processed[col] = df_complete_processed[col] + offset

            # Transform
            if transform_type == 'double_log':
                df_complete_processed[col] = np.log1p(np.log1p(df_complete_processed[col]))
            elif transform_type == 'cbrt':
                df_complete_processed[col] = np.cbrt(df_complete_processed[col])
            else:
                df_complete_processed[col] = np.log1p(df_complete_processed[col])

            # Inverse transform
            if transform_type == 'double_log':
                df_complete_processed[col] = np.expm1(np.expm1(df_complete_processed[col]))
            elif transform_type == 'cbrt':
                df_complete_processed[col] = df_complete_processed[col] ** 3
            else:
                df_complete_processed[col] = np.expm1(df_complete_processed[col])

            # Subtract offset
            if offset > 0:
                df_complete_processed[col] = df_complete_processed[col] - offset

    output_complete_path = os.path.join(run_dir, "complete_survival_data_processed.csv")
    df_complete_processed.to_csv(output_complete_path, index=False)
    logging.info(f"Processed complete data saved to: {output_complete_path}")

    # Run evaluation (if available and not skipped)
    if HAS_EVALUATION and not training_config.skip_evaluation:
        logging.info("Starting imputation evaluation...")
        try:
            evaluation_imputation.evaluate_imputation(
                original_path=output_complete_path,
                imputed_path=output_path,
                missing_path=preprocessor.config.missing_data_path,
                output_dir=run_dir
            )
            logging.info("Evaluation completed")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")

    if verbose:
        print(f"\nImputation completed! Results saved in: {run_dir}")
        print(f"Imputed data: {output_path}")
        print(f"Complete data: {output_complete_path}")


if __name__ == "__main__":
    main()