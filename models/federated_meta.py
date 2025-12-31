"""Federated Meta-Learning"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class FederatedClient:
    def __init__(self, client_id, model, lr=0.001, local_epochs=5, device='cuda'):
        self.client_id = client_id
        self.model = model.to(device)
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device
        
    def train(self, train_loader):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = 0.0
        
        for epoch in range(self.local_epochs):
            for x, edge_index, y in train_loader:
                x, edge_index, y = x.to(self.device), edge_index.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, edge_index)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        return self.model, total_loss / (len(train_loader) * self.local_epochs), 5.0

class FederatedServer:
    def __init__(self, model):
        self.global_model = model
        
    def aggregate_models(self, client_models, client_weights):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
            for cid, model in client_models.items():
                weight = client_weights.get(cid, 1.0/len(client_models))
                global_dict[key] += weight * model.state_dict()[key]
        self.global_model.load_state_dict(global_dict)
        return self.global_model
