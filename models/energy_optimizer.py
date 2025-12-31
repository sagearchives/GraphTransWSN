"""Energy Optimization"""
import torch
import torch.nn as nn
import numpy as np

class EnergyOptimizer:
    def __init__(self, num_clients, initial_energy=100.0, threshold=20.0):
        self.num_clients = num_clients
        self.energy_levels = {i: initial_energy for i in range(num_clients)}
        self.threshold = threshold
        
    def select_clients(self, participation_rate=0.3, method='energy_aware'):
        num_select = max(1, int(self.num_clients * participation_rate))
        if method == 'random':
            return np.random.choice(self.num_clients, num_select, replace=False).tolist()
        
        probs = []
        for i in range(self.num_clients):
            e = self.energy_levels[i]
            probs.append(1.0 if e > 80 else 0.7 if e > 50 else 0.3 if e > self.threshold else 0.0)
        
        total = sum(probs)
        if total == 0: return []
        probs = [p/total for p in probs]
        return np.random.choice(self.num_clients, num_select, replace=False, p=probs).tolist()
    
    def update_energy(self, client_id, energy_used):
        self.energy_levels[client_id] = max(0, self.energy_levels[client_id] - energy_used)
    
    def get_stats(self):
        return {
            'avg_energy': np.mean(list(self.energy_levels.values())),
            'min_energy': min(self.energy_levels.values()),
            'max_energy': max(self.energy_levels.values())
        }
