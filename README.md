# Stable Image Regression

Code and reported results for:

**Stable Image Regression via Adaptive Aggregation of Noisy Predictions and Input-Gradient-Regularized Fine-Tuning**

The framework addresses image-to-scalar regression. It evaluates a base regressor on multiple Gaussian-noised versions of the same image, aggregates the scalar predictions with either a fixed or learned permutation-invariant operator, and uses input-gradient and aggregation-sensitivity regularization during fine-tuning.

The repository contains the training and evaluation scripts used for the three tasks in the manuscript:

- image quality assessment on KonIQ-10k;
- facial age estimation on IMDB-WIKI;
- crowd counting on ShanghaiTech Part B.

## Installation

```bash
conda create -n stable-regression python=3.10 -y
conda activate stable-regression
pip install -r requirements.txt
```

The experiments require a CUDA-enabled PyTorch installation. Install the PyTorch build compatible with the CUDA driver before installing the remaining packages if the default pip build is not suitable for your system.

## Data and checkpoints

Datasets and pretrained checkpoints are not included in this repository. See [`data/README.md`](data/README.md) for the expected directory structure and the required files.

The scripts use paths relative to the repository root, usually under `data/`. Large model checkpoints should be downloaded separately and should not be committed to GitHub.

## Main scripts

The main scripts are in [`scripts/`](scripts/):

| Task | Training | Evaluation |
|---|---|---|
| IQA | `train_iqa.py` | `evaluate_iqa.py` |
| Age estimation | `train_age.py` | `evaluate_age.py` |
| Crowd counting | `train_crowd.py` | `evaluate_crowd.py` |

The noise-free ablation scripts are also provided:

- `evaluate_noise_free_iqa.py`;
- `evaluate_noise_free_age.py`;
- `evaluate_noise_free_crowd.py`.

The scripts `benchmark_iqa_runtime.py`, `benchmark_age_runtime.py`, and `benchmark_crowd_runtime.py` reproduce the inference-time measurements as a function of the number of noisy predictions.

## Evaluation settings

The reported regimes are:

| Regime | $ε$ | $σ$ | $N$ |
|---|---:|---:|---:|
| Easy | 4/255 | 4/255 | 256 |
| Medium | 8/255 | 8/255 | 128 |
| Hard | 16/255 | 16/255 | 64 |

The equality $ε=σ$ is a matched-scale experimental convention. The two parameters have different meanings: $ε$ is the radius used for empirical local-sensitivity evaluation, whereas $σ$ is the standard deviation of the Gaussian input noise.

## Results

The values reported in the manuscript are provided in [`results/`](results/):

- `main_results.csv`: clean and attacked MSE and empirical stability score $B(x)$ for all tasks, regimes, and aggregation methods;
- `upper_tail_medium.csv`: Medium-setting p90 and p99 statistics;
- `noise_free_ablation.csv`: complete versus noise-free comparison;
- `runtime.csv`: inference time and FPS for learned and fixed-mean aggregation.

The score $B(x)$ is an empirical, procedure-dependent diagnostic. It is not a formal worst-case robustness certificate.

## Reproducibility notes

The exact training and evaluation commands may require adjusting the dataset and checkpoint paths for the local environment. The code uses the same model families, perturbation settings, aggregation baselines, and evaluation protocol as the manuscript. Random seeds, batch sizes, memory-efficient prediction chunks, and data-specific preprocessing are defined in the corresponding scripts.

## Citation

```bibtex

```
