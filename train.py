import argparse
import logging
import torch
import src.data as data
import src.model as model
import src.trainer as trainer 
import os
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear regression model on tabular data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimization.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting training with data from %s", args.data_path)

    # Load data
    dataloader = data.get_dataloader(args.data_path, batch_size=args.batch_size)

    # Initialize model, optimizer, and loss function
    input_dim = dataloader.dataset.data.shape[1] - 1  # Number of features
    output_dim = 1  # Single target variable
    model_instance = model.LinearModel(input_dim, output_dim)
    optimizer = torch.optim.SGD(model_instance.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    # Train the model
    trainer_instance = trainer.Trainer(model_instance, optimizer, criterion, args.epochs)
    trainer_instance.fit(dataloader)

    os.makedirs("models", exist_ok=True)
    trainer_instance.save("models/linear_model.pth")
    logger.info("Model saved to models/linear_model.pth")

    logger.info("Training completed.")

    os.makedirs("plots", exist_ok=True)
    plt.plot(trainer_instance.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("plots/loss.png")
    logger.info("Loss curve saved to plots/loss.png")