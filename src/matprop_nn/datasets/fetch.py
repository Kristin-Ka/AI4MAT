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


SUMMARY_BULK_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "nsites",
    "nelements",
    "volume",
    "density",
    "symmetry",
    "band_gap",
    "energy_above_hull",
    "formation_energy_per_atom",
    "is_stable",
    "is_metal",
    "structure",
    "deprecated",
]


def _doc_to_summary_record(doc) -> dict:
    """Convert a summary doc → flat dict, structure serialised to dict.

    Note: ``doc.model_dump()`` serialises ``MPID`` through emmet's custom
    Pydantic serializer, which produces a **non-canonical short alias**
    (e.g. ``mp-cpjas`` instead of ``mp-1183694``).  That form is accepted
    by the MP API but cannot be cross-referenced against any other MP
    dump, so we overwrite it with the canonical ``str(MPID)``.
    """
    data = doc.model_dump()
    data["material_id"] = str(doc.material_id)
    st = getattr(doc, "structure", None)
    if st is not None and hasattr(st, "as_dict"):
        data["structure"] = st.as_dict()
    return jsanitize(data)


def fetch_mp_summary(
    api_key: str,
    *,
    out_path: str | Path,
    fields: list[str] | None = None,
    include_deprecated: bool = False,
    chunk_size: int = 1000,
    batch_size: int = 500,
    target_filter: str | None = None,
    max_records: int | None = None,
    resume: bool = True,
) -> int:
    """Download every MP material with structure + band_gap + e_above_hull.

    The query is run in two phases so progress streams to disk instead of
    waiting for a single huge response:

    1. **IDs only** — list all non-deprecated material IDs (fast, a few MB).
    2. **Batched fetch** — for each batch of ~500 IDs, call the summary
       endpoint with the full ``fields`` list and write each record to
       JSONL.  This lets 100k+ fetches resume if the connection drops.

    Parameters
    ----------
    api_key, fields, chunk_size, include_deprecated:
        See MP-API ``MPRester.materials.summary.search``.
    batch_size:
        IDs per property-fetch batch (500 is a good default — larger
        values occasionally time out on the MP side).
    target_filter:
        If set, drop records where this field is ``None`` (e.g.
        ``"band_gap"`` keeps only materials with a reported band gap).
    max_records:
        Stop after this many records (smoke tests).
    resume:
        If True and *out_path* already exists, skip IDs that were written
        previously (keyed by ``material_id``).
    """
    if fields is None:
        fields = SUMMARY_BULK_FIELDS

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already: set[str] = set()
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    mid = json.loads(line).get("material_id")
                    if mid:
                        already.add(str(mid))
                except json.JSONDecodeError:
                    continue
        logger.info("Resuming: %d record(s) already on disk.", len(already))

    with MPRester(api_key=api_key) as mpr:
        logger.info("Phase 1/2: listing material IDs ...")
        id_kwargs: dict = {
            "fields": ["material_id"],
            "num_chunks": None,
            "chunk_size": chunk_size,
        }
        if not include_deprecated:
            id_kwargs["deprecated"] = False
        id_docs = mpr.materials.summary.search(**id_kwargs)
        all_ids = [str(d.material_id) for d in id_docs]
        logger.info("  → %d material IDs available.", len(all_ids))

        pending = [mid for mid in all_ids if mid not in already]
        if max_records is not None:
            pending = pending[: max(0, max_records - len(already))]
        logger.info("Phase 2/2: fetching %d record(s) in batches of %d.",
                    len(pending), batch_size)

        n_written = len(already)
        n_skipped = 0
        mode = "a" if already else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            for batch in tqdm(list(_batched(pending, batch_size)),
                              desc="summary", unit="batch"):
                docs = mpr.materials.summary.search(
                    material_ids=batch, fields=fields,
                )
                for doc in docs:
                    rec = _doc_to_summary_record(doc)
                    if target_filter is not None and rec.get(target_filter) is None:
                        n_skipped += 1
                        continue
                    if rec.get("structure") is None:
                        n_skipped += 1
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                f.flush()
                if max_records is not None and n_written >= max_records:
                    break

    logger.info("Wrote %d total records to %s (%d skipped this run).",
                n_written, out_path, n_skipped)
    return n_written

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
