## Properties Predicted

| Property | Key | Source |
|---|---|---|
| Total dielectric constant (ε total) | `e_total` | MP dielectric endpoint |
| Ionic dielectric constant (ε ionic) | `e_ionic` | MP dielectric endpoint |
| Band gap (eV) | `band_gap` | MP summary |
| Energy above hull (eV) | `energy_above_hull` | MP summary |

## Models

### Descriptor-based baselines (no 3D atomic coordinates)
- **XGBoost** — gradient-boosted trees on matminer features
- **MLP** — fully connected neural network on matminer features

Matminer features include:
- Composition features (~130 dims): ElementProperty (Magpie), Stoichiometry, ValenceOrbital, IonProperty
- Density features (~3 dims): density, volume per atom, packing fraction

### Graph neural networks (3D structure-aware)
- **TensorNet** — matgl PyG-native equivariant GNN
- **M3GNet** — matgl multi-body interaction GNN (DGL backend)
- **CHGNet** — Crystal Hamiltonian GNN (pretrained backbone + fresh head)
- **ALIGNN** — Atomistic Line Graph Neural Network (atom + bond + angle)

## Repository Structure

```
matprop-nn/
├── configs/
│   ├── default.yaml              # Base config
│   ├── smoke_test.yaml           # Quick CPU smoke test
│   ├── tasks/                    # Per-property configs
│   │   ├── total_dielectric.yaml
│   │   ├── ionic_dielectric.yaml
│   │   ├── band_gap.yaml
│   │   └── energy_above_hull.yaml
│   └── models/                   # Per-model configs
│       ├── xgboost.yaml
│       ├── mlp.yaml
│       ├── tensornet.yaml
│       ├── m3gnet.yaml
│       ├── chgnet.yaml
│       └── alignn.yaml
├── data/
│   ├── raw/                      # Raw API downloads
│   ├── processed/                # Feature matrices, summaries
│   └── splits/                   # Train/val/test split files
├── src/matprop_nn/
│   ├── datasets/                 # Data loading, MP API fetch
│   │   ├── fetch.py              # MP API data fetcher
│   │   ├── mp_dataset.py         # JSON → MatGL dataset
│   │   └── graph.py              # Structure → graph conversion
│   ├── features/                 # Matminer feature pipeline
│   │   ├── matminer_features.py  # Composition + density featurizers
│   │   └── descriptor_dataset.py # Tabular PyTorch dataset
│   ├── models/                   # Model implementations
│   │   ├── _base.py              # Abstract ModelEngine
│   │   ├── _tensornet.py         # TensorNet engine
│   │   ├── _m3gnet.py            # M3GNet engine
│   │   ├── _chgnet.py            # CHGNet engine
│   │   ├── _alignn.py            # ALIGNN engine
│   │   ├── _xgboost.py           # XGBoost regressor
│   │   └── _mlp.py               # MLP regressor + Lightning module
│   ├── tasks/                    # Training orchestration
│   │   ├── train.py              # Unified train entry point
│   │   └── regression.py         # Lightning regression module
│   ├── evaluation/               # Metrics, plots, benchmark
│   │   ├── metrics.py            # RMSE, MAE, R²
│   │   ├── plots.py              # Parity, histogram, bar charts
│   │   └── benchmark.py          # Result aggregation + report
│   └── utils/                    # Shared utilities
│       ├── config.py             # YAML loading + merging
│       └── splits.py             # Reproducible data splitting
├── scripts/
│   ├── fetch_data.py             # Download from MP API
│   ├── fetch_mp_dataset.py       # Full-featured MP data fetcher
│   ├── preprocess.py             # Create splits + features
│   ├── train.py                  # Train single model
│   ├── evaluate.py               # Evaluate checkpoint
│   ├── run_benchmark.py          # Full benchmark suite
│   └── generate_report.py        # Post-hoc report generation
├── notebooks/                    # Exploratory notebooks
├── results/                      # Generated benchmark outputs
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -e ".[gnn]"
# or
pip install -r requirements.txt
pip install -e .
```

### 2. Download data

```bash
# Set your Materials Project API key
export MP_API_KEY="gDeL3eAcYyN2IFRKj9oaEMitmCyiiPhn"

# Download dielectric materials with structures + band gap + energy above hull
python scripts/fetch_mp_dataset.py \
    --api-key $MP_API_KEY \
    --out-json mp_materials.json
```

### 3. Preprocess

```bash
# Create task-specific splits and dataset summaries
python scripts/preprocess.py --json mp_materials.json --out-dir data/processed

# Optionally generate matminer features (slow, ~30 min for ~7K materials)
python scripts/preprocess.py --json mp_materials.json --out-dir data/processed \
    --generate-features --n-jobs 4
```

### 4. Train a single model

```bash
# Train XGBoost on total dielectric
python scripts/train.py configs/default.yaml

# Train with merged task + model configs via the benchmark runner
python scripts/run_benchmark.py --models xgboost --tasks e_total
```

### 5. Run the full benchmark

```bash
# All models × all properties
python scripts/run_benchmark.py --skip-errors

# Subset
python scripts/run_benchmark.py \
    --models xgboost mlp tensornet alignn \
    --tasks e_total band_gap \
    --results-dir results

# Descriptor baselines only (fast)
python scripts/run_benchmark.py --models xgboost mlp --tasks e_total e_ionic band_gap energy_above_hull
```

### 6. Generate report

```bash
python scripts/generate_report.py --results-dir results
# Produces: results/REPORT.md, results/benchmark_results.csv, comparison plots
```

### 7. Evaluate a checkpoint

```bash
python scripts/evaluate.py configs/default.yaml path/to/best.ckpt --out-dir results/eval
```

## Configuration

Configs are composable YAML files. The benchmark runner merges a **task config** with a **model config**:

```
configs/tasks/band_gap.yaml + configs/models/alignn.yaml → merged config for ALIGNN on band gap
```

Key settings:

| Section | Key | Description |
|---|---|---|
| `data.json_path` | Path to raw MP JSON | |
| `data.target_key` | Target column name | `e_total`, `band_gap`, etc. |
| `data.split_file` | Path to split JSON | Ensures fair comparison |
| `model.arch` | Model architecture | `xgboost`, `mlp`, `tensornet`, `m3gnet`, `chgnet`, `alignn` |
| `training.max_epochs` | Max training epochs | |
| `training.patience` | Early stopping patience | |
| `training.seed` | Random seed | Fixed at 42 for reproducibility |

## Data Splitting

All models use the same split per task, persisted as JSON:

```json
{
  "seed": 42,
  "n_total": 6878,
  "n_train": 5502,
  "n_val": 688,
  "n_test": 688,
  "train": ["mp-1234", ...],
  "val": ["mp-5678", ...],
  "test": ["mp-9012", ...]
}
```

Default: 80/10/10 train/val/test. Material IDs (not indices) ensure stability.

## Evaluation Metrics

- **RMSE** 
- **MAE** 
- **R²** 

Generated outputs:
- Parity plots (predicted vs actual)
- Error histograms
- Comparison bar charts
- Summary CSV and markdown report

## Low-Resource Mode

For development/testing on limited hardware:

```bash
# CPU smoke test with tiny dataset
python scripts/train.py configs/smoke_test.yaml

# Run only fast models
python scripts/run_benchmark.py --models xgboost mlp --tasks e_total
```
