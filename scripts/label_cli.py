"""Tiny terminal labelling tool. Progress is saved after every answer, so you
can stop with Ctrl+C and resume later.

    python scripts/label_cli.py relevance --annotator yourname
    python scripts/label_cli.py absa --annotator yourname
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from reviewrec.config import get_settings  # noqa: E402
from reviewrec.evaluation.labeling import RUBRIC  # noqa: E402


def _load_existing(path: Path, keys: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=keys)


def label_relevance(eval_dir: Path, annotator: str) -> None:
    pool = pd.read_csv(eval_dir / "labeling" / "relevance_pool.csv")
    out = eval_dir / "relevance_labels.csv"
    done = _load_existing(out, ["query_id", "product_id", "grade", "annotator"])
    done_keys = set(zip(done["query_id"], done["product_id"]))
    todo = pool[[(q, p) not in done_keys for q, p in zip(pool["query_id"], pool["product_id"])]]
    print(f"Rubric: {RUBRIC}\n{len(todo)} pairs left.\n")
    for row in todo.itertuples(index=False):
        print("=" * 80)
        print(f"QUERY [{row.query_id}]: {row.query}")
        print(f"PRODUCT: {row.title}\n  brand={row.brand} price={row.price} rating={row.mean_rating} reviews={row.n_reviews}")
        print(textwrap.indent(str(row.evidence), "  "))
        grade = ""
        while grade not in {"0", "1", "2"}:
            grade = input("Grade 0/1/2 (q to quit): ").strip()
            if grade == "q":
                return
        done.loc[len(done)] = [row.query_id, row.product_id, grade, annotator]
        done.to_csv(out, index=False)


def label_absa(eval_dir: Path, annotator: str) -> None:
    sample = pd.read_csv(eval_dir / "labeling" / "absa_sample.csv")
    out = eval_dir / "absa_labels.csv"
    done = _load_existing(out, ["mention_id", "aspect", "label", "aspect_correct", "annotator"])
    todo = sample[~sample["mention_id"].isin(set(done["mention_id"]))]
    print(f"{len(todo)} mentions left. Labels: p=positive n=negative u=neutral; aspect correct: y/n\n")
    codes = {"p": "positive", "n": "negative", "u": "neutral"}
    for row in todo.itertuples(index=False):
        print("=" * 80)
        print(f"ASPECT: {row.aspect} (term '{row.term}')\nSENTENCE: {row.sentence}\nCLAUSE:   {row.clause}")
        label = correct = ""
        while label not in codes:
            label = input("Sentiment about this aspect p/n/u (q to quit): ").strip().lower()
            if label == "q":
                return
        while correct not in {"y", "n"}:
            correct = input("Is the sentence really about this aspect? y/n: ").strip().lower()
        done.loc[len(done)] = [row.mention_id, row.aspect, codes[label], "yes" if correct == "y" else "no", annotator]
        done.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["relevance", "absa"])
    parser.add_argument("--annotator", default="me")
    args = parser.parse_args()
    eval_dir = get_settings().path("paths.evaluation_dir")
    try:
        (label_relevance if args.task == "relevance" else label_absa)(eval_dir, args.annotator)
    except KeyboardInterrupt:
        print("\nSaved. Run again to continue.")


if __name__ == "__main__":
    main()
