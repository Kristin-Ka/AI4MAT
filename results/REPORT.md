# Benchmark Summary

90:10 train/test split (matminer convention).  Descriptor models carve
a small internal validation fold from train for XGBoost early stopping.

* **subset=MP-full** — 154,879 MP materials from `mp_summary.jsonl`.
* **subset=dielectric-7k** — 7,327-material dielectric subset (legacy).

| task              | model   | subset        |   N_test |   RMSE (orig) |   MAE (orig) |   R² (orig) |   R² (trained) | trained in   |
|-------------------|---------|---------------|----------|---------------|--------------|-------------|----------------|--------------|
| band_gap          | mlp     | MP-full       |    15488 |        0.7259 |       0.4359 |      0.7701 |         0.7701 | identity     |
| band_gap          | xgboost | MP-full       |    15488 |        0.6655 |       0.4043 |      0.8068 |         0.8068 | identity     |
| e_ionic           | mlp     | dielectric-7k |      733 |       65.3577 |      10.7473 |      0.0347 |         0.5609 | log1p        |
| e_ionic           | xgboost | dielectric-7k |      733 |       63.5284 |      10.2574 |      0.088  |         0.6015 | log1p        |
| e_total           | mlp     | dielectric-7k |      733 |       64.9541 |      11.5297 |      0.0844 |         0.5808 | log1p        |
| e_total           | xgboost | dielectric-7k |      733 |       63.698  |      11.0232 |      0.1195 |         0.6334 | log1p        |
| energy_above_hull | mlp     | MP-full       |    15488 |        0.2685 |       0.0856 |      0.6388 |         0.6709 | log1p        |
| energy_above_hull | xgboost | MP-full       |    15488 |        0.2669 |       0.0898 |      0.6431 |         0.6693 | log1p        |
