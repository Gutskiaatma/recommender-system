"""Grid-search the scoring weights on the DEV queries only.

    python scripts/tune_weights.py

Prints the best (alpha, beta, gamma) by mean NDCG@10 on dev queries. Copy the
chosen values into config/settings.yaml (scoring section) and then run
scripts/evaluate.py on the untouched test queries.

Caveat: products that were never in the labelling pool count as non-relevant,
which slightly favours weightings close to the ones used to build the pool.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from reviewrec.config import get_settings  # noqa: E402
from reviewrec.evaluation.runner import evaluate_rankings, labels_by_query  # noqa: E402
from reviewrec.ranking.recommender import Recommender  # noqa: E402
from reviewrec.ranking.scoring import Weights  # noqa: E402


def main() -> None:
    s = get_settings()
    eval_dir = s.path("paths.evaluation_dir")
    rec = Recommender.from_settings(s)
    queries = pd.read_csv(eval_dir / "queries.csv")
    dev = queries[queries["split"] == "dev"]
    qrels = labels_by_query(pd.read_csv(eval_dir / "relevance_labels.csv"))
    steps = [round(x * 0.05, 2) for x in range(0, 21)]
    configs = {}
    for a, b in itertools.product(steps, steps):
        g = round(1 - a - b, 2)
        if a >= 0.2 and g >= 0:
            configs[f"a={a},b={b},g={g}"] = ("hybrid", Weights(a, b, g))
    _, summary = evaluate_rankings(rec, dev, qrels, configs)
    table = pd.DataFrame(summary).T.sort_values(["NDCG@10", "P@5"], ascending=False)
    print(table[["NDCG@10", "P@5", "MRR", "judged@10"]].head(10).to_string())
    out = eval_dir / "results" / "weight_tuning_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table.head(20).round(4).to_dict(orient="index"), indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
