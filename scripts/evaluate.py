"""Run the full evaluation and write results to evaluation/results/.

    python scripts/evaluate.py               # test split (default)
    python scripts/evaluate.py --split dev

Needs evaluation/relevance_labels.csv (and optionally evaluation/absa_labels.csv),
created with scripts/make_labeling_sets.py + scripts/label_cli.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from reviewrec.config import get_settings  # noqa: E402
from reviewrec.evaluation.runner import (  # noqa: E402
    ablation_configs,
    default_configs,
    evaluate_absa,
    evaluate_rankings,
    labels_by_query,
    paired_bootstrap,
    relevance_agreement,
    signal_correlation,
)
from reviewrec.ranking.recommender import Recommender  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()

    s = get_settings()
    eval_dir = s.path("paths.evaluation_dir")
    results_dir = eval_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rec = Recommender.from_settings(s)

    queries = pd.read_csv(eval_dir / "queries.csv")
    if args.split != "all":
        queries = queries[queries["split"] == args.split]
    labels_path = eval_dir / "relevance_labels.csv"
    if not labels_path.exists():
        sys.exit("evaluation/relevance_labels.csv not found. Create labels first (see README > Evaluation).")
    labels = pd.read_csv(labels_path, dtype={"grade": int})
    qrels = labels_by_query(labels)
    notes_path = eval_dir / "LABELS_NOTE.txt"
    labels_note = notes_path.read_text(encoding="utf-8").strip() if notes_path.exists() else "Manually labelled pooled judgements."

    configs = {**default_configs(rec), **ablation_configs(rec)}
    per_query, summary = evaluate_rankings(rec, queries, qrels, configs)
    per_query.to_csv(results_dir / f"ranking_per_query_{args.split}.csv", index=False)
    methods = {m: summary[m] for m in ("tfidf", "semantic", "hybrid")}
    ablations = {k: v for k, v in summary.items() if k not in methods}
    ranking = {
        "split": args.split,
        "n_queries": int(per_query["query_id"].nunique()),
        "methods": methods,
        "ablations": ablations,
        "significance_ndcg10": {
            "hybrid_vs_semantic": paired_bootstrap(per_query, "hybrid", "semantic"),
            "hybrid_vs_tfidf": paired_bootstrap(per_query, "hybrid", "tfidf"),
            "semantic_vs_tfidf": paired_bootstrap(per_query, "semantic", "tfidf"),
        },
        "weights": {"alpha": rec.weights.alpha, "beta": rec.weights.beta, "gamma": rec.weights.gamma},
        "aspect_quality_correlation": signal_correlation(rec),
        "annotator_agreement": relevance_agreement(labels),
        "labels_note": labels_note,
        "embedding_model": rec.store.manifest.get("embedding_model"),
        "sentiment_backend": rec.store.manifest.get("sentiment_backend"),
    }
    (results_dir / "ranking_summary.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(json.dumps({"methods": methods, "ablations": ablations, "significance": ranking["significance_ndcg10"]}, indent=2))

    absa_path = eval_dir / "absa_labels.csv"
    if absa_path.exists():
        absa_labels = pd.read_csv(absa_path)
        absa = evaluate_absa(
            rec.store.mentions,
            absa_labels,
            float(s.get("sentiment.neutral_band", 0.3)),
            float(s.get("sentiment.vader_neutral_band", 0.05)),
        )
        absa["labels_note"] = labels_note
        (results_dir / "absa_summary.json").write_text(json.dumps(absa, indent=2), encoding="utf-8")
        (results_dir / "lexicon_summary.json").write_text(json.dumps(absa["lexicon"], indent=2), encoding="utf-8")
        print(json.dumps({k: absa[k] for k in ("n", "backends", "lexicon")}, indent=2))


if __name__ == "__main__":
    main()
