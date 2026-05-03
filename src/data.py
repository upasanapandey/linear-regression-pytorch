"""
data.py — Data loading utilities for Linear Regression training.

Assumes CSV format:
    - All columns except the last are features (X)
    - The last column is the target (y)
"""

import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    PyTorch Dataset for loading tabular CSV data.

    Args:
        csv_file : Path to the input CSV file.
                   Last column is treated as the target variable.
    """

    def __init__(self, csv_file: str):
        if not csv_file:
            logger.error("No file path provided.")
            raise ValueError("No file path provided.")

        if not csv_file.endswith(".csv"):
            logger.error("File must be a .csv: %s", csv_file)
            raise ValueError(f"Expected a .csv file, got: {csv_file}")

        if not os.path.exists(csv_file):
            logger.error("File not found: %s", csv_file)
            raise FileNotFoundError(f"Data file not found: {csv_file}")

        logger.info("Loading data from %s", csv_file)
        self.data = pd.read_csv(csv_file)

        if self.data.empty:
            raise ValueError(f"CSV file is empty: {csv_file}")

        if self.data.shape[1] < 2:
            raise ValueError("CSV must have at least one feature column and one target column.")
        
        # Work on numpy directly — avoids pandas SettingWithCopyWarning
        X = self.data.iloc[:, :-1].values.astype(float)
        y = self.data.iloc[:, -1].values.astype(float)

        # Normalize features
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # Normalize target
        y = (y - y.mean()) / y.std()

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        logger.info("Loaded %d rows, %d features", len(self.data), self.data.shape[1] - 1)
        logger.info("y mean: %.4f  y std: %.4f", self.y.mean().item(), self.y.std().item())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_dataloader(csv_file: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Build a DataLoader from a CSV file.

    Args:
        csv_file   : Path to the CSV file.
        batch_size : Number of samples per batch (default: 32).
        shuffle    : Whether to shuffle data each epoch (default: True).

    Returns:
        A PyTorch DataLoader ready for training.
    """
    dataset = CustomDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)