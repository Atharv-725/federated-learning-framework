# Federated Learning Framework with Differential Privacy

A production-inspired simulation of Federated Learning across 10 virtual clients, built from scratch using PyTorch. Includes Differential Privacy via Opacus.

## Results

| Method | Test Accuracy | Data Privacy |
|--------|--------------|--------------|
| Centralized Baseline | 73.55% | ❌ Raw data exposed |
| Federated Learning (IID) | 63.69% | ✅ No raw data shared |
| Federated + DP (ε=50) | 39.21% | ✅✅ Mathematically guaranteed |

## How to Run

```bash
pip install torch torchvision flwr opacus matplotlib numpy
python train_baseline.py
python federated_train.py
python federated_train_dp.py
```

## Tech Stack
PyTorch | Flower | Opacus | CIFAR-10 | Matplotlib