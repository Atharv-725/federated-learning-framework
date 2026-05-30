import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import CNN

class FederatedClient:
    def __init__(self, client_id, dataloader, device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device

    def train(self, global_weights, epochs=1):
        """Train locally and return updated weights"""
        model = CNN().to(self.device)
        model.load_state_dict(global_weights)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f"  ✅ Client {self.client_id} finished local training")
        return model.state_dict()