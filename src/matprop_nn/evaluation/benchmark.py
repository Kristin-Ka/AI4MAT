"""Benchmark result aggregation and summary table."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single model × property evaluation result."""
    property_name: str
    model_name: str
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    rmse: float = float("nan")
    mae: float = float("nan")
    r2: float = float("nan")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d


class BenchmarkTable:
    """Collects :class:`BenchmarkResult` entries and produces summary tables."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def summary(self, metric: str = "rmse") -> pd.DataFrame:
        """Pivot table: rows = property, columns = model, values = metric."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.pivot_table(
            index="property_name", columns="model_name", values=metric,
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Saved benchmark table (%d rows) to %s", len(df), path)

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkTable":
        df = pd.read_csv(path)
        table = cls()
        for _, row in df.iterrows():
            table.add(BenchmarkResult(
                property_name=row["property_name"],
                model_name=row["model_name"],
                n_train=int(row.get("n_train", 0)),
                n_val=int(row.get("n_val", 0)),
                n_test=int(row.get("n_test", 0)),
                rmse=float(row.get("rmse", float("nan"))),
                mae=float(row.get("mae", float("nan"))),
                r2=float(row.get("r2", float("nan"))),
            ))
        return table

    def markdown_report(self) -> str:
        """Generate a markdown summary of all benchmark results."""
        df = self.to_dataframe()
        if df.empty:
            return "No benchmark results available."

        lines = ["# Benchmark Summary\n"]

        properties = df["property_name"].unique()
        lines.append(f"**Properties evaluated**: {len(properties)}")
        lines.append(f"**Models compared**: {df['model_name'].nunique()}\n")

        for metric in ["rmse", "mae", "r2"]:
            pivot = self.summary(metric)
            if pivot.empty:
                continue
            lines.append(f"## {metric.upper()}\n")
            lines.append(pivot.to_markdown())
            lines.append("")

        for prop in properties:
            prop_df = df[df["property_name"] == prop]
            best_row = prop_df.loc[prop_df["rmse"].idxmin()]
            lines.append(
                f"**{prop}** — best model: **{best_row['model_name']}** "
                f"(RMSE={best_row['rmse']:.4f}, R²={best_row['r2']:.4f})"
            )

        return "\n".join(lines)
