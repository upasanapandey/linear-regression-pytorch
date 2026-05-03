import torch
import logging
import mlflow
import numpy as np

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://localhost:5000")

class Trainer:
    def __init__(self, model, optimizer, criterion, epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

    def fit(self, dataloader, params:dict=None):
        self.model.train()
        self.losses = []

        # Set experiment name
        mlflow.set_experiment("California-Housing-Linear-Regression")

        with mlflow.start_run(run_name=f"lr={params.get('learning_rate')}_epochs={params.get('epochs')}_bs={params.get('batch_size')}"):                                  
            if params:                                              
                mlflow.log_params(params)  

            # Track first loss for improvement calculation
            first_loss = None

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_losses = []
                for features, target in dataloader:
                    self.optimizer.zero_grad()
                    output = self.model(features)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    batch_losses.append(loss.item())
                avg_loss = epoch_loss / len(dataloader)
                self.losses.append(avg_loss)

                if first_loss is None:
                    first_loss = avg_loss

                # ── Log metrics per epoch ──────────────────────────────
                mlflow.log_metrics({
                    "train_loss":       avg_loss,
                    "loss_improvement": first_loss - avg_loss,        # how much better vs epoch 1
                    "batch_loss_std":   float(np.std(batch_losses)),  # stability of training
                    "batch_loss_min":   float(np.min(batch_losses)),  # best batch this epoch
                    "batch_loss_max":   float(np.max(batch_losses)),  # worst batch this epoch
                }, step=epoch)

            
                if (epoch + 1) % 10 == 0:
                    logger.info("Epoch %d/%d — Loss: %.4f", epoch+1, self.epochs, avg_loss)

            # ── Log final summary metrics ──────────────────────────────
            mlflow.log_metrics({
                "final_loss":          self.losses[-1],
                "best_loss":           min(self.losses),
                "worst_loss":          max(self.losses),
                "total_improvement":   self.losses[0] - self.losses[-1],
                "pct_improvement":     ((self.losses[0] - self.losses[-1]) / self.losses[0]) * 100,
            })

            # Log model weights and artifacts
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_artifact("plots/loss.png")
            mlflow.log_artifact("models/linear_model.pth")  

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features, _ in dataloader:
                output = self.model(features)
                predictions.append(output)
        return torch.cat(predictions, dim=0)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
