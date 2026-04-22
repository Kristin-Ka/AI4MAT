#!/usr/bin/env python3
"""Run the full benchmark: all models × all properties.

Merges task + model YAML configs, runs training for each combination,
collects results into a benchmark table, and generates comparison plots.

Usage:
    # Full benchmark
    python scripts/run_benchmark.py

    # Subset of models/tasks
    python scripts/run_benchmark.py --models xgboost mlp alignn --tasks e_total band_gap

    # Single run
    python scripts/run_benchmark.py --models xgboost --tasks e_total
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

TASK_CONFIGS = {
    "e_total": "configs/tasks/total_dielectric.yaml",
    "e_ionic": "configs/tasks/ionic_dielectric.yaml",
    "band_gap": "configs/tasks/band_gap.yaml",
    "energy_above_hull": "configs/tasks/energy_above_hull.yaml",
}

MODEL_CONFIGS = {
    "xgboost": "configs/models/xgboost.yaml",
    "mlp": "configs/models/mlp.yaml",
    "xgboost_3d": "configs/models/xgboost_3d.yaml",
    "mlp_3d": "configs/models/mlp_3d.yaml",
    "tensornet": "configs/models/tensornet.yaml",
    "m3gnet": "configs/models/m3gnet.yaml",
    "chgnet": "configs/models/chgnet.yaml",
    "alignn": "configs/models/alignn.yaml",
}

DEFAULT_MODEL_ORDER = [
    "xgboost", "mlp", "xgboost_3d", "mlp_3d",
    "tensornet", "m3gnet", "chgnet", "alignn",
]
DEFAULT_TASK_ORDER = ["e_total", "e_ionic", "band_gap", "energy_above_hull"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full benchmark suite.")
    parser.add_argument(
        "--models", nargs="*", default=DEFAULT_MODEL_ORDER,
        help=f"Models to run (default: {DEFAULT_MODEL_ORDER})",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=DEFAULT_TASK_ORDER,
        help=f"Tasks/targets to run (default: {DEFAULT_TASK_ORDER})",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-errors", action="store_true",
                        help="Continue on model failures instead of aborting.")
    args = parser.parse_args()

    from matprop_nn.utils.config import load_config, merge_configs
    from matprop_nn.tasks.train import run
    from matprop_nn.evaluation.benchmark import BenchmarkTable, BenchmarkResult
    from matprop_nn.evaluation.plots import (
        comparison_bar_chart, multi_property_comparison,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    table = BenchmarkTable()

    total = len(args.models) * len(args.tasks)
    done = 0

    for task_key in args.tasks:
        if task_key not in TASK_CONFIGS:
            logger.error("Unknown task: %s (available: %s)", task_key, list(TASK_CONFIGS))
            continue
        task_cfg = load_config(TASK_CONFIGS[task_key])

        for model_name in args.models:
            if model_name not in MODEL_CONFIGS:
                logger.error("Unknown model: %s (available: %s)", model_name, list(MODEL_CONFIGS))
                continue

            done += 1
            logger.info(
                "=" * 60 + "\n[%d/%d] Task: %s | Model: %s\n" + "=" * 60,
                done, total, task_key, model_name,
            )

            model_cfg = load_config(MODEL_CONFIGS[model_name])
            merged = merge_configs(task_cfg, model_cfg)

            merged.setdefault("data", {})
            merged["data"]["target_key"] = task_cfg.get("task", {}).get("target_key", task_key)
            if "json_path" not in merged["data"]:
                merged["data"]["json_path"] = "mp_materials.json"

            merged.setdefault("training", {})
            merged["training"]["log_dir"] = str(args.results_dir / model_name)

            try:
                result = run(merged)
                if result:
                    table.add(result)
                    logger.info(
                        "Result: RMSE=%.4f  MAE=%.4f  R²=%.4f",
                        result.rmse, result.mae, result.r2,
                    )
            except Exception:
                logger.error(
                    "FAILED: %s / %s\n%s", model_name, task_key, traceback.format_exc(),
                )
                if not args.skip_errors:
                    return 1

    table.save(args.results_dir / "benchmark_results.csv")
    table.save_json(args.results_dir / "benchmark_results.json")

    df = table.to_dataframe()
    if not df.empty:
        for metric in ["rmse", "mae", "r2"]:
            pivot = table.summary(metric)
            if not pivot.empty:
                logger.info("\n%s summary:\n%s", metric.upper(), pivot.to_string())

        all_results: dict = {}
        for _, row in df.iterrows():
            prop = row["property_name"]
            model = row["model_name"]
            all_results.setdefault(prop, {})[model] = {
                "RMSE": row["rmse"], "MAE": row["mae"], "R2": row["r2"],
            }

        for metric in ["RMSE", "R2"]:
            multi_property_comparison(
                all_results, metric=metric,
                save_path=args.results_dir / f"comparison_{metric.lower()}.png",
            )

        for prop, models in all_results.items():
            comparison_bar_chart(
                models, metric="RMSE",
                title=f"{prop} — RMSE",
                save_path=args.results_dir / f"comparison_{prop}_rmse.png",
            )

        report = table.markdown_report()
        report_path = args.results_dir / "REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("Report saved to %s", report_path)

    logger.info("Benchmark complete. Results in %s", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
