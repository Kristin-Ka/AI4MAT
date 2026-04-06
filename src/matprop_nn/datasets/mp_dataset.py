"""Materials Project dataset loader — converts a JSON dump into MatGL graphs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pymatgen.core import Structure

from matgl.ext._pymatgen_pyg import get_element_list
from matgl.graph._data_pyg import MGLDataset

from .graph import get_converter

logger = logging.getLogger(__name__)


class MPDataset:
    """Load a Materials Project JSON file and expose a MatGL-ready dataset.

    Parameters
    ----------
    json_path:
        Path to a JSON list of records, each containing ``"structure"``
        (pymatgen dict) and at least one scalar target field.
    target_key:
        Name of the scalar target column (e.g. ``"e_total"``, ``"band_gap"``).
    cutoff:
        Graph edge cutoff radius in Angstroms.
    element_types:
        Explicit element tuple.  Inferred from the structures when ``None``.
    cache_dir:
        Where MatGL caches processed PyG graphs on disk.
    """

    def __init__(
        self,
        json_path: str | Path,
        target_key: str,
        cutoff: float = 5.0,
        element_types: tuple[str, ...] | None = None,
        cache_dir: str = "MGLDataset",
    ):
        self.json_path = Path(json_path)
        self.target_key = target_key
        self.cutoff = cutoff

        records = self._load_records()
        self.structures, self.targets, self.material_ids = self._parse(records)

        if element_types is None:
            element_types = get_element_list(self.structures)
        self.element_types = element_types

        self.converter = get_converter(
            self.structures,
            cutoff=cutoff,
            element_types=self.element_types,
        )

        self.mgl_dataset = MGLDataset(
            structures=self.structures,
            labels={target_key: self.targets},
            converter=self.converter,
            root=cache_dir,
            save_cache=True,
        )

    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)

    def _parse(self, records: list[dict]):
        structures: list[Structure] = []
        targets: list[float] = []
        material_ids: list[str] = []

        skipped = 0
        for rec in records:
            struct_dict = rec.get("structure")
            target_val = rec.get(self.target_key)
            if struct_dict is None or target_val is None:
                skipped += 1
                continue
            try:
                structures.append(Structure.from_dict(struct_dict))
            except Exception:
                skipped += 1
                continue
            targets.append(float(target_val))
            material_ids.append(str(rec.get("material_id", "")))

        if skipped:
            logger.warning("Skipped %d records (missing structure or target).", skipped)
        logger.info("Loaded %d structures with target '%s'.", len(structures), self.target_key)
        return structures, targets, material_ids

    def __len__(self) -> int:
        return len(self.mgl_dataset)

    def __getitem__(self, idx):
        return self.mgl_dataset[idx]
