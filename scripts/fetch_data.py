#!/usr/bin/env python3
"""Download MP material property data (dielectric + band gap + energy above hull).

This script uses the comprehensive fetch_mp_dataset.py fetcher under the hood
or the matprop_nn library directly for a clean interface.

Usage:
    python scripts/fetch_data.py --api-key YOUR_KEY --out-json mp_materials.json

    # Or with env var:
    MP_API_KEY=YOUR_KEY python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description="Download MP material dataset.")
    p.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    p.add_argument("--out-json", default="mp_materials.json")
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--include-deprecated", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        print("Error: set MP_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    from matprop_nn.datasets.fetch import fetch_mp_data
    df = fetch_mp_data(
        api_key=args.api_key,
        out_json=args.out_json,
        batch_size=args.batch_size,
        include_deprecated=args.include_deprecated,
    )
    print(f"DataFrame shape: {df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
