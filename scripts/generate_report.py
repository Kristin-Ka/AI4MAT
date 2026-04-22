#!/usr/bin/env python3
"""Generate a final markdown benchmark report from collected results.

Scans the results directory for metrics.json files, aggregates them,
and produces a summary table, comparison plots, and a markdown report.

Usage:
    python scripts/generate_report.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark report.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    from matprop_nn.evaluation.benchmark import BenchmarkTable, BenchmarkResult
    from matprop_nn.evaluation.plots import multi_property_comparison, comparison_bar_chart

    table = BenchmarkTable()

    csv_path = args.results_dir / "benchmark_results.csv"
    if csv_path.exists():
        table = BenchmarkTable.load(csv_path)
        logger.info("Loaded %d results from %s", len(table.results), csv_path)
    else:
        for metrics_file in sorted(args.results_dir.rglob("metrics.json")):
            rel = metrics_file.relative_to(args.results_dir)
            parts = list(rel.parts)
            if len(parts) < 2:
                continue
            model_name = parts[0]
            run_dir = parts[1] if len(parts) > 2 else parts[0]
            dir_name = metrics_file.parent.name
            if "_" in dir_name:
                idx = dir_name.index("_")
                target_key = dir_name[idx + 1:]
            else:
                target_key = dir_name

            with open(metrics_file) as f:
                metrics = json.load(f)

            table.add(BenchmarkResult(
                property_name=target_key,
                model_name=model_name,
                rmse=metrics.get("RMSE", float("nan")),
                mae=metrics.get("MAE", float("nan")),
                r2=metrics.get("R2", float("nan")),
            ))
        logger.info("Found %d result files", len(table.results))

    if not table.results:
        logger.warning("No results found in %s", args.results_dir)
        return 1

    table.save(args.results_dir / "benchmark_results.csv")

    df = table.to_dataframe()
    all_results: dict = {}
    for _, row in df.iterrows():
        prop = row["property_name"]
        model = row["model_name"]
        all_results.setdefault(prop, {})[model] = {
            "RMSE": row["rmse"], "MAE": row["mae"], "R2": row["r2"],
        }

    for metric in ["RMSE", "MAE", "R2"]:
        multi_property_comparison(
            all_results, metric=metric,
            save_path=args.results_dir / f"comparison_{metric.lower()}.png",
        )

    report = table.markdown_report()
    report_path = args.results_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("Report:\n%s", report)
    logger.info("Saved to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
