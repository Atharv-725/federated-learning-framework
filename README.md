# Federated Learning Framework with Differential Privacy

A from-scratch implementation of Federated Learning (FedAvg) in PyTorch, simulating
10 clients training collaboratively on CIFAR-10 without sharing raw data — with an
added Differential Privacy layer (via Opacus) to quantify the accuracy/privacy tradeoff.

Unlike most FL demos that wrap an existing framework (e.g. Flower), the server-side
aggregation and client-side local training loop here are implemented manually
(`server/fl_server.py`, `clients/fl_client.py`) to make the mechanics of FedAvg explicit.

## What this shows

- How a global model can be trained across decentralized clients using **FedAvg**
  (local training → weight averaging → redistribution), with no raw data ever
  leaving a client.
- How adding formal **Differential Privacy** (via Opacus' DP-SGD) affects model
  utility — a real, measured tradeoff, not just a theoretical claim.

## Architecture

```
                    ┌─────────────────┐
                    │  Global Model    │
                    │  (CNN, CIFAR-10) │
                    └────────┬─────────┘
                             │ broadcast weights
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Client 0 │   │ Client 1 │  ...  Client 9 │
        │ local    │   │ local    │   │ local    │
        │ training │   │ training │   │ training │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Server: FedAvg   │
                    │ (weight average) │
                    └─────────────────┘
```

- **Model**: Simple CNN (2 conv layers + 2 FC layers), trained on CIFAR-10.
- **Data split**: IID — CIFAR-10 training set shuffled and split evenly across
  10 clients (each client currently sees a random, balanced subset).
- **Training loop**: 5 global rounds × 3 local epochs per client, Adam optimizer (lr=0.001).
- **Aggregation**: Standard FedAvg — server averages client weight tensors after each round.
- **Differential Privacy**: `federated_train_dp.py` swaps in Opacus' `PrivacyEngine`,
  using DP-SGD (per-sample gradient clipping + calibrated noise) with a target
  epsilon and delta=1e-5, instead of vanilla SGD.

## Results

| Method | Test Accuracy | Data Privacy |
|---|---|---|
| Centralized Baseline | 73.55% | Raw data centralized, no privacy guarantee |
| Federated Learning (IID, no DP) | 63.69% | No raw data leaves any client |
| Federated + DP-SGD (ε = 50, δ = 1e-5) | 39.21% | Formal (ε, δ)-DP guarantee, on top of FL |

**Observations:**
- Moving from centralized to federated training costs ~10 points of accuracy,
  a known tradeoff from decentralized, per-client optimization.
- Adding DP-SGD costs another ~24 points — a **larger accuracy drop than expected**
  for a relatively loose privacy budget (ε = 50 is considered weak/loose privacy;
  strong guarantees are usually ε ≤ 10, often ≤ 1). This is called out as an open
  question below rather than glossed over.

## Known limitations / next steps

- **IID split only.** Real federated learning is most interesting under non-IID
  data (each client with a different label/feature distribution). Adding a
  non-IID split (e.g. Dirichlet partition) is the highest-value next step.
- **Large accuracy drop at ε=50 needs investigation.** Candidates: the DP run
  uses plain SGD (lr=0.05) vs Adam in the non-DP run, which is a confound;
  per-sample gradient clipping (`max_grad_norm=1.0`) may be too aggressive for
  this batch size / model. A controlled ablation (same optimizer, sweep over
  epsilon and clip norm) would make this result publication-quality.
- **No communication-efficiency measurement** — e.g. bytes transferred per round,
  which matters in real FL deployments.

## How to Run

```bash
pip install torch torchvision opacus matplotlib numpy

python train_baseline.py        # centralized baseline
python federated_train.py       # federated learning (no DP)
python federated_train_dp.py    # federated learning + differential privacy
python plot_results.py          # generates results/accuracy_comparison.png
```

## Tech Stack
PyTorch · Opacus (DP-SGD) · CIFAR-10 · NumPy · Matplotlib
