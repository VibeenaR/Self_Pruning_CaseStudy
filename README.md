# Self-Pruning Neural Network: CIFAR-10 Weight Optimization

## 1. Project Overview
This repository contains a solution for the Self-Pruning Network case study. The goal was to build a 3-layer Feed-Forward Network (FFN) that learns to reduce its own complexity during training. By applying an $L1$ penalty to a learnable gating mechanism, the network attempts to identify and "prune" redundant connections without significant loss in accuracy.

## 2. Technical Implementation
- **Architecture:** 3-layer FFN (3072 -> 512 -> 256 -> 10).
- **Pruning Mechanism:** Custom `PrunableLinear` layer implementing element-wise multiplication between weights and Sigmoid-activated gate scores.
- **Optimization:** Adam optimizer with a composite loss function:
  $$Loss = CrossEntropy + \lambda \cdot Mean(\sigma(Gates))$$

## 3. Results Table
| Experiment | Lambda ($\lambda$) | Test Accuracy | Sparsity (%) |
| :--- | :--- | :--- | :--- |
| Baseline | 1.0 | ~55% | 0.00% |
| Aggressive | 10.0 | ~55% | 0.00% |
| Extreme | 50.0 | ~54% | 0.00% |

## 4. Analytical Reasoning (Problem Solving)
The observed sparsity remained at 0.00% across multiple experiments. While the code is architecturally correct and gradients are flowing, I have identified the following technical reasons for this behavior:

1. **Sigmoid Saturation:** The gates were initialized to $0.5$ ($\sigma(0)$). At this point, the Sigmoid function has its maximum gradient, but the "pressure" from the classification task on CIFAR-10 is orders of magnitude stronger than the $L1$ penalty, causing the gates to stay "open" to preserve accuracy.
2. **Signal Dominance:** With ~1.7 million parameters, the network prioritizes the complex feature extraction required for CIFAR-10 over the compression target. 
3. **Threshold Sensitivity:** The strict pruning threshold ($1e-2$) requires the gate scores to move significantly into the negative range. Within a 15-epoch window, the optimizer prioritized minimizing Cross-Entropy over reaching the sparsity threshold.

## 5. Proposed Future Improvements
If given more time, the following strategies would be implemented to force pruning:
- **Logit Bias:** Initializing gate scores at lower values (e.g., -1.0) to start the weights closer to the pruning threshold.
- **Hard Concrete Distributions:** Using non-saturating gating functions that allow the weights to reach exactly zero.
- **Penalty Scheduling:** Implementing a "warm-up" period where the network learns to classify before the sparsity penalty is gradually introduced.

## 6. How to Run
1. Ensure `torch`, `torchvision`, and `matplotlib` are installed.
2. Run the main script:
   ```bash
   python solution.py