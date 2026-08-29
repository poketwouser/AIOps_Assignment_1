#!/usr/bin/env python3
"""Build a CSV index of every image under data/ - one row per file, plus a header."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "labels.csv"

rows = []
for p in sorted(DATA.rglob("*.jpg")):          # sorted() => deterministic across runs
    rows.append((
        p.relative_to(HERE).as_posix(),        # filepath
        p.parent.name,                         # label: cats | dogs
        p.parent.parent.name,                  # split: train | validation
    ))

with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filepath", "label", "split"])
    w.writerows(rows)

print(f"{OUT.name}: {len(rows)} data rows + 1 header = {len(rows) + 1} lines")
