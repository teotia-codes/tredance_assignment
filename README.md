# The Self-Pruning Neural Network — Case Study Report

**Submitted for:** Tredence Analytics — AI Engineering Internship 2025 Cohort  
**Task:** Implement a self-pruning feed-forward network on CIFAR-10 with learnable gates and sparsity regularization.

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Intuition

Each weight in the network is multiplied by a **gate value** `g ∈ (0, 1)`, produced by applying a sigmoid to a learnable score:

```
g = sigmoid(gate_score / temperature)
```

A gate value close to **0** means the corresponding weight is effectively removed (pruned). A value close to **1** means the weight is fully active.

### Why L1 and Not L2?

The total loss is:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

where `SparsityLoss = Σ |gᵢ|` over all gate values. Since gates are always positive (sigmoid output), this simplifies to `Σ gᵢ`.

The **key geometric reason** L1 encourages sparsity is that its gradient with respect to any gate is **constant** (always `+λ`), regardless of the gate's current magnitude. This means:

- Even a very small gate still receives a constant downward push.
- The optimizer has no incentive to stop at a small non-zero value — it will push the gate all the way to zero unless the classification loss strongly resists.

In contrast, **L2** (sum of squares) produces a gradient proportional to the gate's current value. As the gate shrinks, the penalty gradient shrinks too, and the optimizer settles at a small-but-nonzero value. L2 produces *shrinkage*, not *exact zeros*.

### The Temperature Mechanism

During early training (`TEMP_START = 2.0`), the sigmoid is **soft** — gates are smoother and gradients flow more freely, allowing the network to learn useful features. As training progresses, temperature anneals to `TEMP_END = 0.5`, sharpening the sigmoid and pushing gates toward binary (0 or 1) decisions. This two-phase schedule avoids premature pruning before the network has learned anything meaningful.

### Summary

| Factor | Effect |
|---|---|
| L1 on gates | Constant gradient pushes gates to exactly zero |
| Sigmoid output | Gates bounded in (0,1); always positive → L1 = sum |
| λ scaling | Controls how aggressively gates are pruned |
| Temperature annealing | Smooth early learning → sharp late pruning |

---

## 2. Results Summary

The network was trained for **40 epochs** on CIFAR-10 with a **5-epoch warmup** (no sparsity penalty applied during warmup). Three values of λ were evaluated:

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|---|---|---|
| `1e-6` | 60.00% | 68.91% |
| `3e-6` | 59.15% | 87.95% |
| `1e-5` | 56.53% | 97.27% |

### Observations

- **Low λ = 1e-6:** Minimal pruning pressure. The network achieves the best accuracy at **60.00%** while already pruning ~69% of gates — the warmup + temperature annealing alone drives significant early sparsity.
- **Medium λ = 3e-6:** A well-balanced trade-off. Accuracy drops only ~0.85% (to 59.15%) while sparsity jumps to **87.95%** — nearly 9 in 10 weights are pruned with negligible accuracy cost.
- **High λ = 1e-5:** Aggressive pruning. **97.27%** of gates are zeroed out — the network is nearly fully sparse — at the cost of ~3.5% accuracy (56.53%). The pruning pressure dominates the classification signal in later epochs.

### Best Model Selected: λ = 1e-6
Selected by the criterion: highest accuracy; if within 0.5%, prefer higher sparsity. λ = 1e-6 leads by >0.5% over λ = 3e-6, so it is chosen.

---

## 3. Plots

### Sparsity vs Accuracy Trade-off Across λ Values

![Sparsity vs Accuracy Trade-off](lambda_tradeoff.png)

The grouped bar chart clearly shows the inverse relationship: as λ increases, sparsity rises sharply (68.91% → 87.95% → 97.27%) while test accuracy falls gradually (60.00% → 59.15% → 56.53%). This confirms the self-pruning mechanism is working correctly — the λ hyperparameter gives precise control over the compression-accuracy trade-off.

---

### Best Model Gate Value Distribution (λ = 1e-6)

![Best Model Gate Distribution](best_model_gate_distribution.png)

The histogram shows a **massive spike at gate value ≈ 0** (over 2.5 million gates pruned, left of the red dashed threshold line at 0.01), with a long tail of surviving weights spread across low positive values. This bimodal-like distribution — a dominant near-zero cluster plus a sparse tail of active weights — is the hallmark of a successfully trained self-pruning network. The red dashed line at `threshold = 0.01` separates pruned from active gates.

---

## 4. Architecture and Implementation Notes

### Network Architecture

```
Input (32×32×3 = 3072)
    ↓ PrunableLinear(3072 → 1024) + BatchNorm + ReLU + Dropout(0.25)
    ↓ PrunableLinear(1024 → 512)  + BatchNorm + ReLU + Dropout(0.25)
    ↓ PrunableLinear(512 → 256)   + BatchNorm + ReLU + Dropout(0.25)
    ↓ PrunableLinear(256 → 10)
Output (10 class logits)
```

Total learnable parameters: weights + biases + gate_scores (doubles the parameter count vs. a standard network during training; at inference, pruned weights can be zeroed/removed).

### Optimizer Design

Gate scores use a **higher learning rate** (`5e-3`) than regular weights (`1e-3`). This ensures gates respond quickly to the sparsity penalty without destabilizing the classifier weights.

### Best Model Selection Logic

The best model is selected by:
1. Prefer the model with the highest test accuracy.
2. If two models are within **0.5% accuracy**, prefer the sparser one.

This reflects the real-world deployment goal: maximize compression without meaningfully sacrificing performance.

---

## 5. Key Takeaways

- Self-pruning via learnable sigmoid gates is an elegant, end-to-end differentiable approach to network compression.
- The L1 penalty is the correct choice for inducing exact sparsity — L2 would only shrink weights, not zero them.
- Temperature annealing is critical: it prevents premature collapse during early training and sharpens gate decisions by the end.
- Results confirm the method works decisively — **97.27% sparsity** is achievable while retaining **56.53% test accuracy** on CIFAR-10 with a purely feed-forward network.
- The λ hyperparameter provides a clean, interpretable knob for the accuracy–sparsity trade-off, making deployment decisions straightforward.
