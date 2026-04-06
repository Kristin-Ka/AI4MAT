"""Fetch material data from the Materials Project API."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from emmet.core.summary import HasProps
from monty.json import jsanitize
from mp_api.client import MPRester
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default fields for the dielectric endpoint — swap for other properties.
DEFAULT_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "nsites",
    "nelements",
    "volume",
    "density",
    "symmetry",
    "total",
    "ionic",
    "electronic",
    "e_total",
    "e_ionic",
    "e_electronic",
    "n",
    "deprecated",
    "property_name",
]

SUMMARY_FIELDS = [
    "material_id",
    "structure",
]


def _batched(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _doc_to_record(doc) -> dict:
    data = doc.model_dump()
    st = getattr(doc, "structure", None)
    if st is not None and hasattr(st, "as_dict"):
        data["structure"] = st.as_dict()
    return jsanitize(data)


def fetch_mp_data(
    api_key: str,
    *,
    out_json: str | Path,
    has_props: list[HasProps] | None = None,
    property_endpoint: str = "dielectric",
    property_fields: list[str] | None = None,
    batch_size: int = 400,
    chunk_size: int = 1000,
    include_deprecated: bool = False,
    material_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Query MP for materials with a given property and save to JSON.

    Parameters
    ----------
    has_props:
        Property filter for the summary endpoint (default: ``[HasProps.dielectric]``).
    property_endpoint:
        Sub-endpoint to query (e.g. ``"dielectric"``, ``"elasticity"``).
    property_fields:
        Fields to request from the property endpoint.
    """
    if has_props is None:
        has_props = [HasProps.dielectric]
    if property_fields is None:
        property_fields = DEFAULT_FIELDS

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key=api_key) as mpr:
        if material_ids:
            mpids = list(dict.fromkeys(material_ids))
        else:
            logger.info("Querying summary for material IDs with requested property...")
            kwargs: dict = {
                "has_props": has_props,
                "fields": ["material_id"],
                "num_chunks": None,
                "chunk_size": chunk_size,
            }
            if not include_deprecated:
                kwargs["deprecated"] = False
            docs = mpr.materials.summary.search(**kwargs)
            mpids = list(dict.fromkeys(d.material_id for d in docs))
            logger.info("Found %d material(s).", len(mpids))

        if not mpids:
            raise RuntimeError("No material IDs found.")

        endpoint = getattr(mpr.materials, property_endpoint)

        logger.info("Fetching property documents...")
        prop_docs = []
        for batch in tqdm(list(_batched(mpids, batch_size)), desc=property_endpoint, unit="batch"):
            prop_docs.extend(endpoint.search(material_ids=batch, fields=property_fields))

        fetched_ids = [str(d.material_id) for d in prop_docs]
        logger.info("Fetching structures for %d material(s)...", len(fetched_ids))
        struct_map: dict = {}
        for batch in tqdm(list(_batched(fetched_ids, batch_size)), desc="structures", unit="batch"):
            for d in mpr.materials.summary.search(material_ids=batch, fields=SUMMARY_FIELDS):
                st = getattr(d, "structure", None)
                if st is not None and hasattr(st, "as_dict"):
                    struct_map[str(d.material_id)] = st.as_dict()

    records = [_doc_to_record(d) for d in prop_docs]
    for rec in records:
        rec["structure"] = jsanitize(struct_map.get(str(rec.get("material_id"))))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d records to %s", len(records), out_json)

    return pd.DataFrame.from_records(records)
