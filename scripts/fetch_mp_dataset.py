#!/usr/bin/env python3
"""
Query Materials Project (mp-api) for materials with a given property, fetch
relaxed structures and property fields. Saves a JSON file and a pandas DataFrame
(pickle). Requires MP_API_KEY in the environment or pass --api-key.

Defaults to the dielectric endpoint; change --property-endpoint and
--has-props to target other properties (e.g. elasticity, piezoelectric).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from emmet.core.summary import HasProps
from monty.json import jsanitize
from mp_api.client import MPRester
from tqdm import tqdm

# Default fields for the dielectric endpoint — override via --fields if needed.
DEFAULT_PROPERTY_FIELDS = [
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
    "origins",
    "deprecated",
    "deprecation_reasons",
    "property_name",
]

# Extra fields fetched from the summary endpoint and merged into each record.
SUMMARY_EXTRA_FIELDS = [
    "band_gap",
    "energy_above_hull",
]

SUMMARY_FIELDS = [
    "material_id",
    "structure",
] + SUMMARY_EXTRA_FIELDS


def doc_to_record(doc) -> dict:
    """Convert an API document to a JSON-safe dict, serialising pymatgen objects."""
    data = doc.model_dump()
    # Normalise material_id to the canonical numeric string form (e.g. "mp-149")
    # because model_dump() may return the base-36 encoded form (e.g. "mp-ft").
    data["material_id"] = str(doc.material_id)
    st = getattr(doc, "structure", None)
    if st is not None and hasattr(st, "as_dict"):
        data["structure"] = st.as_dict()
    return jsanitize(data)


def batched(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def collect_material_ids(
    mpr: MPRester,
    *,
    has_props: list[HasProps],
    chunk_size: int,
    deprecated: bool | None,
) -> list[str]:
    kwargs: dict = {
        "has_props": has_props,
        "fields": ["material_id"],
        "num_chunks": None,
        "chunk_size": chunk_size,
    }
    if deprecated is not None:
        kwargs["deprecated"] = deprecated
    docs = mpr.materials.summary.search(**kwargs)
    mpids = [d.material_id for d in docs]
    return list(dict.fromkeys(mpids))


def fetch_property_docs(
    mpr: MPRester,
    material_ids: list[str],
    *,
    property_endpoint: str,
    batch_size: int,
    fields: list[str],
) -> list:
    endpoint = getattr(mpr.materials, property_endpoint)
    all_docs = []
    for batch in tqdm(
        list(batched(material_ids, batch_size)),
        desc=f"{property_endpoint}.search",
        unit="batch",
    ):
        all_docs.extend(endpoint.search(material_ids=batch, fields=fields))
    return all_docs


def fetch_summary_data(
    mpr: MPRester,
    material_ids: list[str],
    *,
    batch_size: int,
) -> dict:
    """Return {material_id: {structure, band_gap, energy_above_hull, ...}} from summary."""
    summary_map: dict = {}
    for batch in tqdm(
        list(batched(material_ids, batch_size)),
        desc="summary.search (structures + extras)",
        unit="batch",
    ):
        docs = mpr.materials.summary.search(material_ids=batch, fields=SUMMARY_FIELDS)
        for d in docs:
            mid = str(d.material_id)
            st = getattr(d, "structure", None)
            entry: dict = {
                "structure": st.as_dict() if st is not None and hasattr(st, "as_dict") else None,
            }
            for field in SUMMARY_EXTRA_FIELDS:
                entry[field] = getattr(d, field, None)
            summary_map[mid] = entry
    return summary_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MP material property data (structure + fields) via mp-api."
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("MP_API_KEY"),
        help="Materials Project API key (default: MP_API_KEY env var).",
    )
    p.add_argument(
        "--property-endpoint",
        default="dielectric",
        help="MP sub-endpoint name, e.g. 'dielectric', 'elasticity' (default: dielectric).",
    )
    p.add_argument(
        "--has-props",
        nargs="*",
        default=["dielectric"],
        help="HasProps filter values for summary query (default: dielectric).",
    )
    p.add_argument(
        "--material-ids",
        nargs="*",
        metavar="MPID",
        help="If set, skip the summary query and fetch only these material IDs.",
    )
    p.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated materials in the summary ID query.",
    )
    p.add_argument(
        "--summary-chunk-size",
        type=int,
        default=1000,
        help="Chunk size when listing material IDs from summary (default: 1000).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="Number of material IDs per property search request (default: 400).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("mp_materials.json"),
        help="Output JSON path (default: ./mp_materials.json).",
    )
    p.add_argument(
        "--out-dataframe",
        type=Path,
        default=Path("mp_materials.pkl"),
        help="Output pandas DataFrame pickle path (default: ./mp_materials.pkl).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Error: set MP_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    has_props = [HasProps(p) for p in args.has_props]

    with MPRester(api_key=args.api_key) as mpr:
        if args.material_ids:
            material_ids = list(dict.fromkeys(args.material_ids))
        else:
            print(f"Querying summary for material IDs with {args.has_props} data...")
            material_ids = collect_material_ids(
                mpr,
                has_props=has_props,
                chunk_size=args.summary_chunk_size,
                deprecated=None if args.include_deprecated else False,
            )
            print(f"Found {len(material_ids)} material(s).")

        if not material_ids:
            print("No material IDs to fetch.", file=sys.stderr)
            return 1

        print(f"Fetching {args.property_endpoint} documents...")
        docs = fetch_property_docs(
            mpr,
            material_ids,
            property_endpoint=args.property_endpoint,
            batch_size=args.batch_size,
            fields=DEFAULT_PROPERTY_FIELDS,
        )

        fetched_ids = [str(d.material_id) for d in docs]
        print(f"Fetching structures + summary extras for {len(fetched_ids)} material(s)...")
        summary_map = fetch_summary_data(mpr, fetched_ids, batch_size=args.batch_size)

    records = [doc_to_record(d) for d in docs]
    for rec in records:
        mid = str(rec.get("material_id", ""))
        entry = summary_map.get(mid, {})
        rec["structure"] = jsanitize(entry.get("structure"))
        for field in SUMMARY_EXTRA_FIELDS:
            rec[field] = entry.get(field)
    df = pd.DataFrame.from_records(records)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_dataframe.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(args.out_dataframe, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(records)} row(s) to {args.out_json}")
    print(f"Wrote DataFrame ({df.shape[0]} rows, {df.shape[1]} cols) to {args.out_dataframe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
