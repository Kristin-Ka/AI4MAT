## Properties Predicted

| Property | Key | Source |
|---|---|---|
| Total dielectric constant (ε total) | `e_total` | MP dielectric endpoint |
| Ionic dielectric constant (ε ionic) | `e_ionic` | MP dielectric endpoint |
| Band gap (eV) | `band_gap` | MP summary |
| Energy above hull (eV) | `energy_above_hull` | MP summary |

## Models

### Descriptor-based baselines (MatMiner)
- **`xgboost`** — gradient-boosted trees on composition + density
  features (135 dims: Magpie ElementProperty + DensityFeatures)
- **`mlp`** — fully connected neural network on the same 135-feature set
- **`xgboost_3d`** — same XGBoost model with **MatMiner3D** features
  added: composition + density + 3D structural descriptors (≈153 dims)
- **`mlp_3d`** — same MLP with MatMiner3D features

MatMiner feature groups:

| Group | Featurizers | Dims | Uses 3D coords? |
|---|---|---|---|
| `composition` | ElementProperty (Magpie), Stoichiometry, ValenceOrbital, IonProperty | 149 | no |
| `density` | DensityFeatures | 3 | lattice only |
| `structure3d` | GlobalSymmetryFeatures, MaximumPackingEfficiency, StructuralHeterogeneity, ChemicalOrdering | 18 | **yes** |

### Graph neural networks (3D structure-aware)
- **`tensornet`** — matgl PyG-native equivariant GNN
- **`m3gnet`** — matgl multi-body interaction GNN (DGL backend)
- **`chgnet`** — Crystal Hamiltonian GNN (pretrained backbone + fresh head)
- **`alignn`** — Atomistic Line Graph Neural Network (atom + bond + angle).
  Uses the 9-property CGCNN atomic-feature table by default
  (group / period / electronegativity / covalent radius / valence
  electrons / first ionization energy / electron affinity / block /
  atomic volume).

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
│       ├── xgboost.yaml           # MatMiner (elem_prop + density)
│       ├── xgboost_3d.yaml        # MatMiner3D (+ structure3d)
│       ├── mlp.yaml
│       ├── mlp_3d.yaml
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
│   ├── fetch_data.py             # Download from MP API (dielectric / summary / band_gap / e_above_hull)
│   ├── fetch_mp_dataset.py       # Lower-level MP data fetcher
│   ├── fix_mp_summary_ids.py     # One-off: canonicalise MP IDs in existing mp_summary.jsonl
│   ├── preprocess.py             # Create splits + features
│   ├── train.py                  # Train a single model from one merged YAML
│   ├── evaluate.py               # Evaluate checkpoint
│   ├── run_benchmark.py          # Full benchmark suite (task × model sweep)
│   ├── sweep_xgboost.py          # Hyperparameter grid search for XGBoost
│   ├── rerender_parity_plots.py  # Regenerate parity plots from predictions.npz
│   ├── build_report.py           # Aggregate metrics.json → results/REPORT.md
│   ├── generate_report.py        # Post-hoc report generation (alt. entry)
│   └── plot_figure_reference.py  # Reproduce the MatMiner/MatMiner3D/M3GNet/ALIGNN figure
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

The dielectric tasks use a small, filtered slice of MP (~7k materials).
Band-gap and energy-above-hull tasks need the **full** MP summary
(~150k materials with structure), which is downloaded into a JSONL file
for streaming load.

```bash
export MP_API_KEY="gDeL3eAcYyN2IFRKj9oaEMitmCyiiPhn"

# Dielectric subset (~7k records, ~40 MB JSON)
python scripts/fetch_data.py --task dielectric --out mp_materials.json

# Full MP summary (~155k records, ~1-2 GB JSONL). Takes ~20-30 min.
# Resumes automatically if interrupted.
python scripts/fetch_data.py --task summary --out mp_summary.jsonl

# Smoke test with a small slice
python scripts/fetch_data.py --task summary --out /tmp/mp_smoke.jsonl --max-records 50
```

``band_gap.yaml`` and ``energy_above_hull.yaml`` point at
``mp_summary.jsonl`` by default — the large file gives the reference
N ≈ 147k / 150k sizes from the Matbench / ALIGNN papers.

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
# Train XGBoost (MatMiner baseline) on total dielectric
python scripts/run_benchmark.py --models xgboost --tasks e_total

# MatMiner3D XGBoost (adds 3D structural descriptors)
python scripts/run_benchmark.py --models xgboost_3d --tasks e_total

# ALIGNN on band gap (large dataset — needs GPU)
python scripts/run_benchmark.py --models alignn --tasks band_gap

# Or use a pre-merged YAML directly
python scripts/train.py configs/default.yaml
```

`scripts/run_benchmark.py` merges the task YAML and model YAML at
run-time, then calls the unified `matprop_nn.tasks.train.run(cfg)`
entry point.  Each run writes to
`results/<model_name>/<arch>_<task>/` and produces:
`metrics.json`, `predictions.npz`, `parity_plot.png`,
`error_histogram.png`, and the Lightning `version_0/metrics.csv`.

### 5. Run the full benchmark

```bash
# All 8 models × 4 tasks (skips any model that fails, so the run
# doesn't die if e.g. CHGNet can't load on this box)
python scripts/run_benchmark.py --skip-errors

# Dielectric-only sweep (~7k records, fast — good for first pass)
python scripts/run_benchmark.py \
    --models xgboost mlp xgboost_3d mlp_3d tensornet m3gnet chgnet alignn \
    --tasks e_total e_ionic --skip-errors

# Just the heavy large-N tasks
python scripts/run_benchmark.py \
    --models xgboost_3d alignn m3gnet \
    --tasks band_gap energy_above_hull --skip-errors

# Descriptor baselines only (no GPU needed)
python scripts/run_benchmark.py --models xgboost mlp xgboost_3d mlp_3d --tasks e_total e_ionic band_gap energy_above_hull
```

### 6. Aggregate results

```bash
# Build results/REPORT.md with a full (model × task) metrics table.
# Uses the top-level folder under results/ as the model identifier,
# so xgboost and xgboost_3d are reported as distinct rows.
python scripts/build_report.py

# Reproduce the reference figure: left panel = Test RMSE on Total /
# Ionic Dielectric; right panel = dual-y-axis RMSE on Band Gap and
# Energy Above Hull.  Four bars per group (MatMiner, MatMiner3D,
# M3GNet, ALIGNN).
python scripts/plot_figure_reference.py --out results/figure_reference.png
```

### 7. Utilities

```bash
# Hyperparameter search for XGBoost (72-config grid on the 90:10 split
# of a single task; reports best by R²).
python scripts/sweep_xgboost.py --task e_total

# Re-render all existing parity plots from saved predictions.npz
# (useful after tweaking plots.py).
python scripts/rerender_parity_plots.py

# Evaluate a checkpoint against the saved test split
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
| `data.json_path` | Path to raw MP JSON / JSONL | `mp_materials.json` (dielectric) or `mp_summary.jsonl` (band gap, e above hull) |
| `data.target_key` | Target column name | `e_total`, `e_ionic`, `band_gap`, `energy_above_hull` |
| `data.split_file` | Path to split JSON | Ensures fair comparison across models |
| `data.target_transform` | `"identity"` or `"log1p"` | `log1p` is used for heavy-tailed targets (dielectric, e_above_hull) |
| `data.test_frac`, `data.val_frac` | Split fractions | Currently 0.10 / 0.00 — see "Data splitting" below |
| `model.arch` | Model architecture | `xgboost`, `mlp`, `tensornet`, `m3gnet`, `chgnet`, `alignn` |
| `model.feature_groups` | MatMiner feature groups | e.g. `["elem_prop", "density"]` or `["elem_prop", "density", "structure3d"]` |
| `training.max_epochs` | Max training epochs | |
| `training.patience` | Early stopping patience | |
| `training.seed` | Random seed | Fixed at 42 for reproducibility |

## Data splitting

All models use the **same** saved split per task (material IDs, not
indices — stable even if the underlying JSON is reshuffled):

```json
{
  "seed": 42,
  "n_total": 7327,
  "n_train": 6594,
  "n_val": 0,
  "n_test": 733,
  "train": ["mp-1234", ...],
  "val": [],
  "test": ["mp-9012", ...]
}
```

Default is **90 / 10 train / test** (`test_frac=0.10`, `val_frac=0.0`),
matching the Petousis 2017 / Matbench setup:

- **Descriptor models** (XGBoost, MLP) carve an internal validation
  subset out of the training set for early stopping.
- **Graph models** auto-carve 10 % of the training set as a validation
  fold (deterministic, seeded) inside `_run_graph_model`, because
  Lightning's `EarlyStopping(monitor="val_loss")` needs one.

If a split file exists but its `n_total` or IDs disagree with the
currently-loaded dataset (e.g. you swapped `mp_materials.json` for
`mp_summary.jsonl`), `_get_or_create_split` regenerates the split
from scratch and overwrites the old JSON.

## Evaluation metrics

- **RMSE**, **MAE**, **R²** — reported in both the training space
  (e.g. `log1p`) and original units when a target transform is active.

Generated outputs per run:
- `metrics.json` — per-task metrics (training-space + `original_units`)
- `predictions.npz` — `y_true`, `y_pred`, `y_train_true`, `y_train_pred`
- `parity_plot.png` — train/test overlay with marginal KDE curves
- `error_histogram.png`
- `version_0/metrics.csv` — Lightning per-epoch training curves (GNN only)

Aggregate outputs:
- `results/REPORT.md` — Markdown table of every (model × task) run
- `results/benchmark_results.{csv,json}` — flat result table
- `results/comparison_<metric>.png` — grouped bar chart per task
- `results/figure_reference.png` — reproduction of the reference
  MatMiner / MatMiner3D / M3GNet / ALIGNN dual-panel plot

## Reproducing the reference figure

The slide-style figure compares four models (MatMiner, MatMiner3D,
M3GNet, ALIGNN) on four tasks (Total Dielectric, Ionic Dielectric,
Band Gap, Energy Above Hull).  Minimum commands to reproduce it
end-to-end on a fresh machine (GPU needed for the GNNs):

```bash
# 0. Data (dielectric subset already in mp_materials.json; re-fetch
#    mp_summary.jsonl if it's missing)
export MP_API_KEY="..."
python scripts/fetch_data.py --task summary --out mp_summary.jsonl

# 1. Descriptor baselines on every task (fast — CPU only)
python scripts/run_benchmark.py \
    --models xgboost mlp xgboost_3d mlp_3d \
    --tasks e_total e_ionic band_gap energy_above_hull --skip-errors

# 2. GNN baselines on every task (GPU recommended; 150k-row
#    band_gap / e_above_hull are multi-hour runs — start detached)
setsid nohup python scripts/run_benchmark.py \
    --models alignn m3gnet chgnet tensornet \
    --tasks e_total e_ionic band_gap energy_above_hull --skip-errors \
    > logs/gnn_all.log 2>&1 & disown

# 3. Aggregate + plot
python scripts/build_report.py
python scripts/plot_figure_reference.py --out results/figure_reference.png
```

The `figure_reference.png` script tolerates missing runs — any
`metrics.json` absent from `results/<model_dir>/<arch>_<task>/`
just leaves a blank bar, so you can iterate on one model at a time.

## Low-resource / smoke mode

For local validation on a small GPU (or CPU):

```bash
# CPU smoke test with tiny dataset (defined in smoke_test.yaml)
python scripts/train.py configs/smoke_test.yaml

# Only the instant-to-train descriptor models
python scripts/run_benchmark.py --models xgboost mlp --tasks e_total
```

## Known footguns

- **`ChemicalOrdering` is heavy.** Part of `structure3d` but pinned to
  `n_jobs=1` via `_SERIAL_FEATURIZERS` (bypasses matminer's
  `multiprocessing.Pool`, which otherwise hangs for 10+ minutes during
  cleanup on Python 3.11 — CPython #105826). Memory footprint is
  4-6 GB RSS on a 7 k-row pass. Recommended: run it on a machine with
  ≥ 24 GB system RAM, and **not concurrently with a GNN training job**.
  Results cache per-material to `data/processed/_feat_cache/ordering.parquet`,
  so a Ctrl-C + re-run resumes where it stopped. To disable it, drop
  `"structure3d"` from `model.feature_groups` or list individual
  featurizers explicitly, e.g.
  `feature_groups: ["elem_prop", "density", "sym", "packing", "heterogeneity"]`.
- **CUDA import shim.** `src/matprop_nn/_torch_preload.py` preloads
  `libcusparseLt.so.0` from the `nvidia-cusparselt-cu12` wheel before
  `torch` is imported. Always `import matprop_nn` before `import torch`
  in ad-hoc scripts.
- **matgl backend.** M3GNet v2+ needs `matgl.set_backend("DGL")`.
  `M3GNetEngine._ensure_dgl_backend()` handles this and reloads
  `matgl.layers` / `matgl.models` after switching.
