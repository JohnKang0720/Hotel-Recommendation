"""Reproduce my original real-data pipeline (optional).

Drop the TripAdvisor CSVs in `dataset/` and run this to build the same
`data/hotel_ratings.csv` the demo uses — from real reviews instead of the
bundled synthetic set.

Expected inputs (the format my notebook used):
    dataset/offerings.csv   columns include: id, name
    dataset/reviews.csv     columns include: offering_id, author (dict), ratings (dict)

Run:  python scripts/prepare_tripadvisor.py --n 4000
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="sample this many reviews")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ds = REPO / "dataset"
    offerings = pd.read_csv(ds / "offerings.csv")[["id", "name"]]
    reviews = pd.read_csv(ds / "reviews.csv")[["ratings", "author", "offering_id"]]

    # ratings/author come in as stringified dicts — pull out the fields I need.
    reviews["rating"] = reviews["ratings"].apply(ast.literal_eval).apply(lambda d: d["overall"])
    reviews["user"] = reviews["author"].apply(ast.literal_eval).apply(lambda d: d["username"])

    merged = (reviews.merge(offerings, left_on="offering_id", right_on="id")
              .rename(columns={"name": "hotel"})[["user", "hotel", "rating"]]
              .dropna())
    merged = merged.sample(min(args.n, len(merged)), random_state=args.seed)
    merged["city"] = ""  # TripAdvisor dump has no clean city column

    out = REPO / "data" / "hotel_ratings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {len(merged):,} real ratings -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
