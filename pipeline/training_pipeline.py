import sys
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.data.loader import load_churn_data
from src.data.preprocessor import DataPreprocessor
from src.models.ann import ChurnANN
from src.utils.metrics import evaluate_model_performance

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline() -> Tuple[ChurnANN, pd.DataFrame]:
    """
    Executes the end-to-end training and evaluation pipeline for SaaS Churn Prediction.
    Includes defensive checks, error handling, and structured exception propagation.
    """
    logger.info("Starting SaaS Customer Churn Training Pipeline...")

    # 1. Data Ingestion Stage
    try:
        logger.info(f"Step 1/5: Ingesting dataset from {config.DATA_PATH}")
        df = load_churn_data(str(config.DATA_PATH))
    except Exception as err:
        logger.error(f"Pipeline failed at Data Ingestion stage: {err}")
        raise

    # 2. Data Preprocessing & Splitting Stage
    try:
        logger.info("Step 2/5: Preprocessing data and creating PyTorch DataLoaders...")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process(df)

        train_loader = processed_data.get("train_loader")
        test_loader = processed_data.get("test_loader")
        y_test = processed_data.get("y_test")
        num_features = processed_data.get("num_features")

        if train_loader is None or test_loader is None or y_test is None:
            raise KeyError("Preprocessor output dictionary is missing essential keys.")

        if num_features is None or num_features <= 0:
            raise ValueError(f"Invalid input feature dimension detected: {num_features}")

    except Exception as err:
        logger.error(f"Pipeline failed at Data Preprocessing stage: {err}")
        raise

    # 3. Model Initialization & Training Stage
    try:
        logger.info(
            f"Step 3/5: Initializing and training ChurnANN model "
            f"(Epochs: {config.EPOCHS}, Learning Rate: {config.LEARNING_RATE}, Input Dim: {num_features})..."
        )
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        model = ChurnANN(input_dim=num_features)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        model.train()
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0 or (epoch + 1) == config.EPOCHS:
                avg_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{config.EPOCHS}] - Average Loss: {avg_loss:.4f}")

    except Exception as err:
        logger.error(f"Pipeline failed during Model Training stage: {err}")
        raise

    # 4. Model Inference Stage
    try:
        logger.info("Step 4/5: Generating predictions on test dataset...")
        model.eval()
        y_probs_list = []

        with torch.no_grad():
            for batch_x, _ in test_loader:
                outputs = model(batch_x)
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_probs_list.extend(probs)

        y_probs = np.array(y_probs_list).ravel()

        if y_probs.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Dimension mismatch between predictions ({y_probs.shape[0]}) "
                f"and ground truth labels ({y_test.shape[0]})."
            )

    except Exception as err:
        logger.error(f"Pipeline failed during Inference stage: {err}")
        raise

    # 5. Model Evaluation Stage
    try:
        logger.info("Step 5/5: Computing sprint performance metrics across decision thresholds...")

        metrics_default = evaluate_model_performance(
            y_true=y_test,
            y_probs=y_probs,
            threshold=config.DEFAULT_THRESHOLD,
            model_name="Default Threshold (0.50)",
            plot_cm=False
        )

        metrics_optimal = evaluate_model_performance(
            y_true=y_test,
            y_probs=y_probs,
            threshold=config.OPTIMAL_THRESHOLD,
            model_name="Optimized Threshold (0.40)",
            plot_cm=False
        )

        results_df = pd.DataFrame(
            [metrics_default, metrics_optimal],
            index=["Default Model (Th=0.50)", "Optimized Model (Th=0.40)"]
        )

        print("\n" + "=" * 65)
        print("MODEL PERFORMANCE EVALUATION REPORT")
        print("=" * 65)
        print(results_df.to_string())
        print("=" * 65 + "\n")

        logger.info("Pipeline execution completed successfully.")
        return model, results_df

    except Exception as err:
        logger.error(f"Pipeline failed during Evaluation stage: {err}")
        raise


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as critical_err:
        logger.critical(f"Execution terminated due to an unhandled exception: {critical_err}")
        sys.exit(1)