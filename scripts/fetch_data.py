#!/usr/bin/env python3
"""Download MP material data — dielectric subset OR full summary.

Two fetch modes are supported:

  --task dielectric     Query ``has_props=dielectric`` → ~7k records with
                        e_total / e_ionic / e_electronic.  Writes a JSON
                        list (small enough to ``json.load`` at once).

  --task summary        Stream the full MP summary endpoint → ~150k records
                        covering band_gap + energy_above_hull + structure.
                        Writes JSONL (one record per line) so the file is
                        streamable at load time.

  --task band_gap       Full summary but skip records without ``band_gap``.
  --task e_above_hull   Full summary but skip records without ``energy_above_hull``.

Usage:
    # Dielectric subset (default, matches the old behaviour)
    python scripts/fetch_data.py --task dielectric --out mp_materials.json

    # Full summary for band_gap / e_above_hull training
    python scripts/fetch_data.py --task summary --out mp_summary.jsonl

    # API key via env var:
    MP_API_KEY=XXXX python scripts/fetch_data.py --task summary
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _run_dielectric(api_key: str, out_path: str, batch_size: int, include_deprecated: bool):
    from matprop_nn.datasets.fetch import fetch_mp_data
    df = fetch_mp_data(
        api_key=api_key,
        out_json=out_path,
        batch_size=batch_size,
        include_deprecated=include_deprecated,
    )
    print(f"Dielectric DataFrame shape: {df.shape}")


def _run_summary(api_key: str, out_path: str, include_deprecated: bool,
                 target_filter: str | None, max_records: int | None):
    from matprop_nn.datasets.fetch import fetch_mp_summary
    n = fetch_mp_summary(
        api_key=api_key,
        out_path=out_path,
        include_deprecated=include_deprecated,
        target_filter=target_filter,
        max_records=max_records,
    )
    print(f"Summary records written: {n}")


def main() -> int:
    p = argparse.ArgumentParser(description="Download MP material dataset.")
    p.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    p.add_argument(
        "--task", default="dielectric",
        choices=["dielectric", "summary", "band_gap", "e_above_hull"],
        help="Which fetch mode to run.",
    )
    p.add_argument(
        "--out",
        help=(
            "Output path.  Default depends on --task: "
            "'dielectric' → mp_materials.json, "
            "others → mp_summary.jsonl."
        ),
    )
    p.add_argument("--batch-size", type=int, default=400,
                   help="mp-api request batch size (dielectric task only).")
    p.add_argument("--include-deprecated", action="store_true")
    p.add_argument("--max-records", type=int, default=None,
                   help="Truncate summary fetch to N records (smoke tests).")
    args = p.parse_args()

    if not args.api_key:
        print("Error: set MP_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    out_path = args.out or (
        "mp_materials.json" if args.task == "dielectric" else "mp_summary.jsonl"
    )

    if args.task == "dielectric":
        _run_dielectric(args.api_key, out_path, args.batch_size, args.include_deprecated)
    elif args.task == "summary":
        _run_summary(args.api_key, out_path, args.include_deprecated,
                     target_filter=None, max_records=args.max_records)
    elif args.task == "band_gap":
        _run_summary(args.api_key, out_path, args.include_deprecated,
                     target_filter="band_gap", max_records=args.max_records)
    elif args.task == "e_above_hull":
        _run_summary(args.api_key, out_path, args.include_deprecated,
                     target_filter="energy_above_hull", max_records=args.max_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
