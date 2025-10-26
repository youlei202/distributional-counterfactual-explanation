# Distributional Counterfactual Explanation (DCE)

This repository accompanies the paper **"Distributional Counterfactual Explanation With Optimal Transport"** (AISTATS 2025, Oral). It implements distributional counterfactual explanations that keep global prediction behaviour and feature distributions within user-specified optimal transport constraints. Alongside the proposed DCE method, the repo bundles baselines, experiment scripts, notebooks, and utility code used in the paper.

> **Please cite**  
> L. You ✉, L. Cao, M. Nilsson, B. Zhao, and L. Lei, "Distributional Counterfactual Explanation With Optimal Transport", International Conference on Artificial Intelligence and Statistics (AISTATS) 2025, accepted (Oral, top 2%).

## Project Highlights
- **Distributional Counterfactual Explainer (`explainers/dce.py`)** implementing interval-narrowing SGD with Wasserstein and sliced Wasserstein constraints.
- **Optimal transport distance tooling** (`explainers/distances.py`) with trimmed Wasserstein, bootstrap confidence intervals, and sliced projections.
- **Baselines:** GLOBE-CE and AReS reference implementations for comparison.
- **Reusable models and datasets:** black-box models in `models/`, dataset loaders/pre-processing in `utils/`.
- **Reproducible experiments and analysis:** Python scripts under `experiments/`, shell launchers in `scripts/`, and Jupyter notebooks for each dataset.

## Repository Layout
- `explainers/` — DCE algorithm, distance utilities, and baseline explainers.
- `experiments/` — dataset-specific experiment drivers that reproduce the results in the paper (cardio, german credit, hotel booking, etc.).
- `models/` — neural network, SVM, logistic regression, and RBF models used as black-box classifiers.
- `utils/` — dataset download/preparation helpers, logging config, and visualisation utilities.
- `data/` — local copies of processed datasets, experiment outputs, and saved explainers. Large raw datasets live in `datasets/` or are provided as archives (e.g. `adult.zip`).
- `pictures/`, `training_plots/` — figures and animations used in the manuscript.
- `analysis_*.ipynb`, `baseline_experiments_*.ipynb`, `*.ipynb` — notebooks exploring datasets and documenting experimental results.
- `scripts/` — SLURM/HPC-friendly shell scripts for batch runs (update paths before use).
- `setup.py`, `requirement.txt` — packaging stub and pinned dependency list.

## Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/youlei202/distributional-counterfactual-explanation.git
   cd distributional-counterfactual-explanation
   ```
2. **Create and activate a Python ≥3.9 environment** (virtualenv, Conda, etc.).
3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   # optional: install as a package
   pip install -e .
   ```
   Some experiments require GPU-enabled PyTorch; install the wheel that matches your platform if needed.

## Data
- Processed datasets and experiment artefacts are stored under `data/`. Many scripts expect pre-computed samples (e.g. `data/cardio/df_explain.csv`); generate them with the provided notebooks or adapt the experiment scripts.
- Raw datasets can be downloaded via `utils/datasets.py`. The helper wraps several UCI/ProPublica data sources and handles encoding/bucketing.
- Update hard-coded `absolute_path` entries inside `experiments/*.py` and `scripts/*.sh` to match your local layout before running.

## Running Experiments
- Each file in `experiments/` trains a model and optimises DCE for a specific dataset/architecture (e.g. `cardio_mlp_010.py`). Adjust configuration parameters at the top of the script as needed (number of projections, tolerance, Wasserstein bounds, etc.).
- Batch scripts in `scripts/` call the Python drivers with SLURM directives; they assume a UNIX environment and should be edited for local paths/modules.
- Logging is configured via `utils/logger_config.py`; outputs plus intermediate artefacts are written to the dataset-specific folders beneath `data/`.

## Notebooks
The notebooks in the repository reproduce figures, baseline comparisons, and diagnostic plots referenced in the paper:
- `analysis_*.ipynb` — statistical analyses and visualisations for each dataset.
- `baseline_experiments_*.ipynb` — reproductions of comparative experiments.
- `mnist.ipynb`, `hotel_booking.ipynb`, etc. — dataset-specific walkthroughs and qualitative examples.

## Contributing
Issues and pull requests are welcome. Please run formatting and keep comments concise. If you add new datasets or experiments, document them here so others can reproduce your setup.

## Citation
If you use this codebase or build upon the DCE method, please cite:
```
@inproceedings{you2025dce,
  author    = {Lei You and Lele Cao and Mikael Nilsson and Bing Zhao and Longfei Lei},
  title     = {Distributional Counterfactual Explanation With Optimal Transport},
  booktitle = {International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2025},
  note      = {Oral, top 2\%}
}
```

For questions, reach out to **Lei You** at `leiyo@dtu.dk`.
