import os
import pandas as pd
from typing import Optional

def load_churn_data(file_path: str) -> pd.DataFrame:
    """
    Loads customer churn CSV dataset with comprehensive error handling.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or corrupted.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at path: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError(f"The file at {file_path} is empty.")
            
        print(f"Data successfully loaded. Shape: {df.shape}")
        return df

    except FileNotFoundError as fnf_err:
        print(f"[Error - Loader]: {fnf_err}")
        raise
    except pd.errors.EmptyDataError:
        print("[Error - Loader]: The specified CSV file contains no data.")
        raise
    except pd.errors.ParserError as parse_err:
        print(f"[Error - Loader]: CSV parsing error: {parse_err}")
        raise
    except Exception as e:
        print(f"[Error - Loader]: An unexpected error occurred while loading data: {e}")
        raise