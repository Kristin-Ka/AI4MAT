#!/usr/bin/env python3
"""Rewrite ``material_id`` in an ``mp_summary.jsonl`` produced before the
MPID serialization fix.

The pre-fix fetcher used ``doc.model_dump()`` which invokes emmet's custom
Pydantic serializer for ``MPID``, producing a non-canonical short alias
(e.g. ``mp-cpjas``) instead of the canonical form (``mp-1183694``).  The
short form is accepted by the MP API, so we can batch-query MP to
translate short → canonical without re-downloading structures.

Usage:
    MP_API_KEY=... python scripts/fix_mp_summary_ids.py mp_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("fix-ids")


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    if not args.api_key:
        print("set MP_API_KEY", file=sys.stderr)
        return 1

    # Phase 1: scan file, collect short-form IDs.
    short_ids: list[str] = []
    logger.info("Scanning %s for material_ids ...", args.input)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            short_ids.append(json.loads(line)["material_id"])
    logger.info("  → %d records", len(short_ids))

    # Phase 2: ask MP for canonical form of each short id.
    from mp_api.client import MPRester

    short_to_canonical: dict[str, str] = {}
    with MPRester(args.api_key) as mpr:
        for batch in tqdm(
            list(_batched(short_ids, args.batch_size)),
            desc="translate", unit="batch",
        ):
            docs = mpr.materials.summary.search(
                material_ids=batch, fields=["material_id"],
            )
            for d in docs:
                # ``d.material_id`` is an MPID whose ``__str__`` gives the
                # canonical long form.  But there's no way to know which
                # short id in the batch it corresponds to from the doc
                # alone, so we ask MP once more with the canonical form
                # to cross-check — skip that; since the MP response docs
                # are returned in the same order as the query (verified
                # empirically), we index by position.
                pass
            # The API doesn't guarantee ordering, so re-derive via
            # per-doc lookup: d.model_dump()["material_id"] is the short
            # alias (because emmet's serializer), and str(d.material_id)
            # is the canonical form.
            for d in docs:
                try:
                    short = d.model_dump()["material_id"]
                    canonical = str(d.material_id)
                    short_to_canonical[short] = canonical
                except Exception:
                    continue

    missing = [s for s in short_ids if s not in short_to_canonical]
    logger.info(
        "Translated %d/%d IDs (missing %d).",
        len(short_to_canonical), len(short_ids), len(missing),
    )
    if missing[:5]:
        logger.warning("Example missing: %s", missing[:5])

    # Phase 3: rewrite the file in place (with a .bak backup).
    bak = args.input.with_suffix(args.input.suffix + ".bak")
    if not bak.exists():
        args.input.rename(bak)
        logger.info("Backup created: %s", bak)
    tmp = args.input.with_suffix(args.input.suffix + ".tmp")

    n_fixed = 0
    with open(bak, encoding="utf-8") as f_in, open(tmp, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=len(short_ids), desc="rewrite"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = rec["material_id"]
            canonical = short_to_canonical.get(mid)
            if canonical is not None and canonical != mid:
                rec["material_id"] = canonical
                n_fixed += 1
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.rename(args.input)
    logger.info("Fixed material_id on %d record(s). Output: %s", n_fixed, args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
