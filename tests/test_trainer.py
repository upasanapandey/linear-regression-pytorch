"""
tests/test_trainer.py — Unit tests for the Trainer class.
Run with: pytest tests/
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import LinearModel
from src.trainer import Trainer


@pytest.fixture
def synthetic_dataloader():
    """
    Creates a small synthetic dataset:
    50 samples, 3 features, 1 target.
    y = 2*x1 + 3*x2 - x3 + noise
    """
    torch.manual_seed(42)
    X = torch.randn(50, 3)
    y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + 0.1 * torch.randn(50)
    y = y.unsqueeze(1)  # shape (50, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def trainer_instance(synthetic_dataloader):
    """Builds a fresh model + trainer ready to test."""
    model = LinearModel(input_dim=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    return Trainer(model, optimizer, criterion, epochs=50)


def test_loss_decreases(trainer_instance, synthetic_dataloader):
    """Loss after training should be lower than loss before training."""
    model = trainer_instance.model
    criterion = trainer_instance.criterion

    # Measure loss before training
    model.eval()
    X, y = next(iter(synthetic_dataloader))
    with torch.no_grad():
        initial_loss = criterion(model(X), y).item()

    # Train
    trainer_instance.fit(synthetic_dataloader)

    # Measure loss after training
    final_loss = trainer_instance.losses[-1]

    assert final_loss < initial_loss, (
        f"Expected loss to decrease, got initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_predict_output_shape(trainer_instance, synthetic_dataloader):
    """Predict should return a tensor with shape (n_samples, 1)."""
    trainer_instance.fit(synthetic_dataloader)
    predictions = trainer_instance.predict(synthetic_dataloader)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[1] == 1, f"Expected 1 output column, got {predictions.shape[1]}"


def test_save_and_load(trainer_instance, synthetic_dataloader, tmp_path):
    """Saved and reloaded model should produce identical predictions."""
    trainer_instance.fit(synthetic_dataloader)

    save_path = tmp_path / "model.pth"
    trainer_instance.save(str(save_path))

    # Use a fixed-order dataloader for comparison — no shuffle
    X, y = synthetic_dataloader.dataset.tensors
    eval_loader = DataLoader(
        TensorDataset(X, y),
        batch_size=16,
        shuffle=False        # ← key change
    )

    preds1 = trainer_instance.predict(eval_loader)

    fresh_model = LinearModel(input_dim=3)
    fresh_trainer = Trainer(
        fresh_model,
        torch.optim.SGD(fresh_model.parameters(), lr=0.01),
        torch.nn.MSELoss(),
        epochs=0,
    )
    fresh_trainer.load(str(save_path))
    preds2 = fresh_trainer.predict(eval_loader)   # ← same loader, same order

    assert torch.allclose(preds1, preds2), "Predictions differ after save/load"