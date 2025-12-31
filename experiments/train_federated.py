"""Federated Training"""
import sys
sys.path.append('.')
import torch
from models.graph_transformer import GraphTransformer
from models.federated_meta import FederatedServer, FederatedClient
from models.energy_optimizer import EnergyOptimizer

def main():
    print("GraphTransWSN Federated Training")
    print("="*50)
    
    # Initialize model
    model = GraphTransformer(num_features=41, hidden_dim=128, num_classes=5)
    print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize federated components
    server = FederatedServer(model)
    energy_opt = EnergyOptimizer(num_clients=20)
    print(f"✓ Federated setup: 20 clients")
    
    # Simulated training
    print("\nStarting federated training...")
    for round_num in range(10):
        selected = energy_opt.select_clients(0.3, 'energy_aware')
        print(f"Round {round_num+1}: Selected {len(selected)} clients")
        
        for cid in selected:
            energy_opt.update_energy(cid, 5.0)
        
        stats = energy_opt.get_stats()
        print(f"  Avg energy: {stats['avg_energy']:.1f}J")
    
    print("\n✓ Training completed!")
    print(f"Final energy stats: {energy_opt.get_stats()}")

if __name__ == "__main__":
    main()
