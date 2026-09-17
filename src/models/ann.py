import torch
import torch.nn as nn
from config import (
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    DROPOUT_1,
    DROPOUT_2,
    OUTPUT_DIM
)

class ChurnANN(nn.Module):
    """
    2-Layer Multi-Layer Perceptron (MLP) for Tabular Churn Classification.
    Features robust error handling for input shapes, NaN checks, and layer exceptions.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = HIDDEN_DIM_1,
        hidden_dim_2: int = HIDDEN_DIM_2,
        dropout_1: float = DROPOUT_1, 
        dropout_2: float = DROPOUT_2,
        output_dim: int = OUTPUT_DIM
    ):
        super(ChurnANN, self).__init__()
        
        # 1. Parameter Validations
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"[Error - ChurnANN]: input_dim must be a positive integer, got {input_dim}")
        
        for name, d_rate in [("dropout_1", dropout_1), ("dropout_2", dropout_2)]:
            if not (0.0 <= d_rate < 1.0):
                raise ValueError(f"[Error - ChurnANN]: {name} must be in range [0.0, 1.0), got {d_rate}")

        try:
            self.input_dim = input_dim
            self.net = nn.Sequential(
                # Layer 1
                nn.Linear(input_dim, hidden_dim_1),
                nn.BatchNorm1d(hidden_dim_1),
                nn.ReLU(),
                nn.Dropout(dropout_1),
                
                # Layer 2
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.BatchNorm1d(hidden_dim_2),
                nn.ReLU(),
                nn.Dropout(dropout_2),
                
                # Output Layer (Raw logits for BCEWithLogitsLoss)
                nn.Linear(hidden_dim_2, output_dim)
            )
        except Exception as e:
            print(f"[Error - ChurnANN Init]: Failed to construct network layers: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2. Input Type Check
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"[Error - ChurnANN Forward]: Input must be a torch.Tensor, got {type(x)}")

        # 3. Dimensionality & Shape Checks
        if x.ndim != 2:
            raise ValueError(f"[Error - ChurnANN Forward]: Expected 2D tensor (batch_size, features), got shape {tuple(x.shape)}")

        if x.shape[1] != self.input_dim:
            raise ValueError(f"[Error - ChurnANN Forward]: Feature dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}")

        # 4. BatchNorm Edge Case Check (BatchNorm1d requires batch_size > 1 during training)
        if self.training and x.shape[0] == 1:
            raise ValueError("[Error - ChurnANN Forward]: Batch size is 1 during training. BatchNorm requires batch_size > 1 to compute statistics.")

        # 5. Numerical Integrity Check
        if torch.isnan(x).any():
            raise ValueError("[Error - ChurnANN Forward]: Input tensor contains NaN values.")
            
        if torch.isinf(x).any():
            raise ValueError("[Error - ChurnANN Forward]: Input tensor contains Infinite values.")

        # 6. Forward Execution with Exception Handling
        try:
            return self.net(x)
        except Exception as e:
            print(f"[Error - ChurnANN Forward]: Forward propagation failed: {e}")
            raise