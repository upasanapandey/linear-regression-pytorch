import torch
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, criterion, epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

    def fit(self, dataloader):
        self.model.train()
        self.losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for features, target in dataloader:
                self.optimizer.zero_grad()
                output = self.model(features)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            self.losses.append(avg_loss)
        
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d/%d — Loss: %.4f", epoch+1, self.epochs, avg_loss)

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
