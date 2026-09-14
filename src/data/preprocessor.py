import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

from config import (
    DROP_COLUMNS, 
    CATEGORICAL_COLUMNS, 
    TARGET_COLUMN, 
    RANDOM_SEED, 
    TEST_SIZE, 
    BATCH_SIZE_TRAIN, 
    BATCH_SIZE_TEST
)

class ChurnDataset(Dataset):
    """Custom PyTorch Dataset for Churn tabular data."""
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        try:
            self.x = torch.tensor(x_data, dtype=torch.float32)
            y_arr = y_data.values if hasattr(y_data, 'values') else y_data
            self.y = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
        except Exception as e:
            print(f"[Error - ChurnDataset]: Failed to convert arrays to tensors: {e}")
            raise

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def process(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Processes dataframe and prepares PyTorch DataLoaders with error handling.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[Error - Preprocessor]: Expected a pandas DataFrame, received {type(df)}.")
            
        if df.empty:
            raise ValueError("[Error - Preprocessor]: Input DataFrame is empty.")

        if TARGET_COLUMN not in df.columns:
            raise KeyError(f"[Error - Preprocessor]: Target column '{TARGET_COLUMN}' is missing from data.")

        try:
            # 1. Drop non-predictive identifiers
            cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
            df_clean = df.drop(columns=cols_to_drop).copy()

            # 2. Target Column Mapping (Yes/No -> 1/0)
            if df_clean[TARGET_COLUMN].dtype == object or isinstance(df_clean[TARGET_COLUMN].iloc[0], str):
                df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).fillna(0)

            # 3. Handle Any Other Binary/String 'Yes'/'No' Columns
            for col in df_clean.select_dtypes(include=['object']).columns:
                if col != TARGET_COLUMN and set(df_clean[col].dropna().unique()).issubset({'Yes', 'No'}):
                    df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0}).fillna(0)

            # 4. One-Hot Encoding for categorical columns (like plan_type)
            existing_cats = [c for c in CATEGORICAL_COLUMNS if c in df_clean.columns]
            if existing_cats:
                df_encoded = pd.get_dummies(df_clean, columns=existing_cats, drop_first=True, dtype=int)
            else:
                df_encoded = df_clean

            # 5. Feature and Target Separation
            x = df_encoded.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
            y = df_encoded[TARGET_COLUMN].values.astype(np.float32)

            # 6. Stratified Split
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                stratify=y
            )

            # 7. Scaling
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)

            # 8. PyTorch Loaders
            train_dataset = ChurnDataset(x_train_scaled, y_train)
            test_dataset = ChurnDataset(x_test_scaled, y_test)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

            print(f"[Preprocessor]: Preprocessing completed successfully. Features count: {x_train_scaled.shape[1]}")

            return {
                "train_loader": train_loader,
                "test_loader": test_loader,
                "x_train": x_train_scaled,
                "x_test": x_test_scaled,
                "y_train": y_train,
                "y_test": y_test,
                "num_features": x_train_scaled.shape[1],
                "scaler": self.scaler
            }

        except KeyError as key_err:
            print(f"[Error - Preprocessor]: Key or Column issue: {key_err}")
            raise
        except ValueError as val_err:
            print(f"[Error - Preprocessor]: Value or Split error: {val_err}")
            raise
        except Exception as e:
            print(f"[Error - Preprocessor]: Unexpected preprocessing failure: {e}")
            raise