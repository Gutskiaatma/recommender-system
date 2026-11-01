"""Create blind labelling files for the evaluation.

    python scripts/make_labeling_sets.py

Writes:
    evaluation/labeling/relevance_pool.csv   (fill the `grade` column: 0/1/2)
    evaluation/labeling/absa_sample.csv      (fill `label` and `aspect_correct`)

Then label them with `python scripts/label_cli.py relevance` / `absa`
(or in a spreadsheet) and save the results as
    evaluation/relevance_labels.csv and evaluation/absa_labels.csv
Existing label files are never overwritten by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from reviewrec.config import get_settings  # noqa: E402
from reviewrec.evaluation.labeling import build_absa_sample, build_relevance_pool  # noqa: E402
from reviewrec.ranking.recommender import Recommender  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="pool depth per method")
    parser.add_argument("--absa-n", type=int, default=330)
    args = parser.parse_args()

    s = get_settings()
    eval_dir = s.path("paths.evaluation_dir")
    out_dir = eval_dir / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recommender.from_settings(s)
    queries = pd.read_csv(eval_dir / "queries.csv")
    pool = build_relevance_pool(rec, queries, k=args.k, seed=int(s.get("seed", 42)))
    pool.to_csv(out_dir / "relevance_pool.csv", index=False)
    sample = build_absa_sample(rec.store.mentions, n=args.absa_n, seed=int(s.get("seed", 42)))
    sample.to_csv(out_dir / "absa_sample.csv", index=False)
    print(f"Relevance pool: {len(pool)} (query, product) pairs for {pool['query_id'].nunique()} queries")
    print(f"ABSA sample: {len(sample)} mentions")


if __name__ == "__main__":
    main()
