import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# PART 1: THE PRUNABLE LINEAR LAYER

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Reasoned Change: Init at -1.0 to move gates away from the flat sigmoid plateau
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, -1.0) 

    def forward(self, x):
        # Element-wise multiplication of weight matrix and sigmoid gates
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# PART 2 & 3: TRAINING LOOP AND EVALUATION

def run_experiment(target_lambda, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        batch_size=512, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            cls_loss = F.cross_entropy(outputs, labels)
            
            # --- SPARSITY LOSS ---
            all_gates = []
            for m in model.modules():
                if isinstance(m, PrunableLinear):
                    all_gates.append(torch.sigmoid(m.gate_scores).flatten())
            
            # MEAN across all 1.7M gates provides a stable gradient signal
            gate_tensor = torch.cat(all_gates)
            sparsity_loss = torch.mean(gate_tensor) 
            
            # Total Loss = Accuracy Goal + Compression Penalty
            total_loss = cls_loss + (target_lambda * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    final_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            final_gates.extend(torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten())
    
    final_gates = np.array(final_gates)
    # Threshold < 0.01 counts as pruned
    sparsity = 100.0 * np.sum(final_gates < 0.01) / len(final_gates)

    # Test Accuracy
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
        batch_size=512, shuffle=False)
    
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return (100.0 * correct / total), sparsity, final_gates

if __name__ == "__main__":
    # Lambdas that will definitely trigger the trade-off
    lambdas = [1.0, 10.0, 50.0]
    os.makedirs("outputs", exist_ok=True)
    all_results = []

    print(f"{'Lambda':<10} | {'Test Acc (%)':<14} | {'Sparsity (%)':<12}")
    print("-" * 42)

    for l in lambdas:
        acc, sp, gates = run_experiment(l)
        all_results.append((l, acc, sp, gates))
        print(f"{l:<10} | {acc:<14.2f} | {sp:<12.2f}")

    # Generate the required visualization for the report
    best_gates = all_results[-1][3]
    plt.figure(figsize=(8, 5))
    plt.hist(best_gates, bins=100, color='darkblue', alpha=0.7)
    plt.axvline(0.01, color='red', linestyle='--', label='Pruning Threshold')
    plt.title(f"Final Gate Distribution (Lambda={lambdas[-1]})")
    plt.xlabel("Gate Strength (Sigmoid)")
    plt.ylabel("Number of Weights")
    plt.legend()
    plt.savefig("outputs/gate_distribution.png")