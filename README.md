# ReviewLens: Review-Based Product Recommendation with NLP and Aspect-Based Sentiment Analysis

You describe the phone you want in plain English, for example *"a phone with an excellent camera, long battery life and good gaming performance under $400"*. ReviewLens ranks real phones by:

- how well each phone matches your request (semantic search over titles and review sentences),
- what customers say about the aspects you care about (aspect-based sentiment analysis), and
- how satisfied owners are overall (a Bayesian-adjusted star rating).

Every recommendation is explained with **real review sentences** from the dataset.

> **Status:** code, tests and evaluation tooling are complete. The numbers in [Results](#results) are filled in after you run the pipeline and evaluation on the dataset (see [Quick start](#quick-start)). No result in this repository is hand-written.

---

## Features

- **Free-text preferences.** The query parser finds:
  - 11 aspects (camera, battery, display, performance, value, durability, audio, connectivity, software, storage, size)
  - emphasis ("excellent", "must have")
  - aspects you want ignored ("don't care about the camera")
  - brands to include or exclude, including aliases such as "iPhone" → Apple
  - price limits ("under $300", "between 100 and 250")
- **Three ranking methods, compared side by side:**
  - `tfidf`: TF-IDF with cosine similarity (baseline)
  - `semantic`: Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) over product titles plus the top-k most similar review sentences
  - `hybrid` (review-aware): `S = α·R + β·A + γ·Q` (relevance, aspect sentiment, quality)
- **Aspect-based sentiment.** A lexicon detects aspects. Each aspect's clause is then classified by a weakly supervised logistic-regression model on sentence embeddings, compared against a VADER baseline. An optional pretrained ABSA transformer backend is also included.
- **Honest explanations:**
  - For each requested aspect: the score, the catalogue average, positive/neutral/negative counts and a confidence value.
  - Up to two positive and one negative real quote, each with its review ID.
  - "Not enough evidence" is shown when there are fewer than 3 mentions.
- **Comparison** of up to 3 phones: aspect scores, star distributions and the best phone per aspect.
- **Insights dashboard:** dataset statistics, the data-quality report and evaluation results.
- **Reproducible pipeline.** One command rebuilds everything from the raw CSV, with fixed seeds.
- **Tests:** 44 pytest tests (unit, integration and API, running offline in about 2 seconds) and frontend unit tests.

## Architecture

```
Raw CSV → prepare → aspects → embed → sentiment → index → artifacts/ → FastAPI → React UI
          (clean,    (sentences,  (MiniLM)  (weak-sup. LR   (TF-IDF, Bayesian
           variants)  mentions)              + VADER)        aspect scores)
```

See [docs/architecture.md](docs/architecture.md) for the full diagram, request flow and design decisions.

## Dataset

| | |
|---|---|
| Name | *Amazon Reviews: Unlocked Mobile Phones* (PromptCloud), via Kaggle |
| Link | https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones |
| Size | About 400k reviews of about 4.4k phone listings, about 130 MB CSV |
| Columns | `Product Name`, `Brand Name`, `Price` (product-level); `Rating`, `Reviews`, `Review Votes` (review-level) |
| Licence | Check the licence shown on the Kaggle page before redistributing. **The data is not included in this repository.** |

**Why this dataset and not Amazon Reviews 2023:** it contains phones only, with no cases or chargers, and it is a single manageable file. The loader uses an adapter (`reviewrec/data/load.py`), so another source can be added by mapping it to the same schema.

### Preprocessing

All of these are recorded in `data/processed/data_quality_report.json`:

1. HTML unescaping, Unicode normalisation, URL removal, collapsing repeated punctuation.
2. Rows without a title or with a rating outside 1–5 are removed.
3. Missing brands are inferred from the first word of the title. Price outliers are set to missing.
4. Accessory listings are removed (keyword rules that check phone hints in the title).
5. Exact duplicate reviews are removed. **Variant merging:** colour and storage variants that share the same review pool are merged into one *product family* using union-find on review overlap, so the same review is never counted twice.
6. Short (fewer than 3 words) or non-English reviews are *flagged*. They still count towards the star rating but are not used for NLP.
7. **Cold start:** products with fewer than 15 reviews are excluded. Text reviews are capped at 250 per product (most helpful first, seeded).

## Scoring formula

```
S(p) = α·R(p) + β·A(p) + γ·Q(p)        α=0.55, β=0.30, γ=0.15 (tune on dev queries)

R  = min-max( 0.4·cos(q, title) + 0.6·mean(top-5 cos(q, review sentences)) )   within the top-100 candidates
A  = Σ_a w_a · score(p,a) / Σ_a w_a                                            over the aspects in the query
     score(p,a) = ( (Σv + k·μ_a)/(n + k) + 1 ) / 2      v ∈ [-1,1] per mention, k = 5, μ_a = catalogue mean
Q  = (bayes_rating − 1) / 4,  bayes_rating = (n·r̄ + m·C)/(n + m),  m = 20
```

**How double counting is handled:**

- `R` measures *topic* match: "battery dies by noon" is very similar to "long battery life". `A` measures *polarity*. They are complementary, and this is exactly why semantic-only ranking can recommend phones that people complain about.
- `A` and `Q` both come from reviews and are correlated, so `Q` gets the smallest weight. The evaluation reports their correlation.
- Review count is never added as a popularity term. It only controls how much the smoothing trusts a product's own data.
- If the query names no aspect, β is redistributed to α and γ.

## Quick start

Requirements: **Python 3.11 or 3.12**, **Node.js 20.19+ (22 LTS recommended)**, and Git. A GPU is not needed. The commands below are for Windows **PowerShell**. On macOS/Linux, use `source .venv/bin/activate` instead.

```powershell
git clone https://github.com/Gutskiaatma/recommender-system.git
cd recommender-system

# 1. Python environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 2. Data: download Amazon_Unlocked_Mobile.csv from Kaggle (link above) into data\raw\

# 3. Build everything (first run downloads the ~90 MB MiniLM model; ~20-40 min on a laptop CPU)
python scripts/run_pipeline.py
#    quick smoke run on a subset:  python scripts/run_pipeline.py --nrows 40000

# 4. Tests
pytest

# 5. API  →  http://localhost:8000/docs
uvicorn backend.app.main:app --port 8000

# 6. Frontend (second terminal)  →  http://localhost:5173
cd frontend
npm install
npm run dev
```

To serve the built UI from FastAPI on one port, run `npm run build` in `frontend/` and open http://localhost:8000.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Status, number of products, models in use |
| POST | `/api/recommend` | `{query, method: tfidf\|semantic\|hybrid, top_k}` → ranked, explained results |
| POST | `/api/compare` | `{product_ids: [2-3 ids], query?}` → side-by-side aspects and match scores |
| GET | `/api/products?search=` | Search the catalogue by title or brand |
| GET | `/api/products/{id}` | Product detail with every aspect and its evidence |
| GET | `/api/aspects`, `/api/examples` | Aspect lexicon and example queries |
| GET | `/api/stats` | Dataset stats, data-quality report, evaluation results |

## Evaluation

There are no ground-truth relevance labels for this task, so we build a small, honest test collection. The 30 queries are in `evaluation/queries.csv`: 10 for development and tuning, 20 for testing.

```powershell
python scripts/make_labeling_sets.py                     # blind pool: top-10 of every method, shuffled, no scores shown
python scripts/label_cli.py relevance --annotator you    # grade 0/1/2 using the rubric
python scripts/label_cli.py absa --annotator you         # sentiment + "is it really about this aspect?"
python scripts/tune_weights.py                           # grid search α, β, γ on DEV queries only
python scripts/evaluate.py                               # metrics on TEST queries → evaluation/results/
python scripts/eda.py                                    # figures + docs/EDA.md
```

**Metrics:**

- **Precision@5:** the share of the top 5 that is relevant.
- **Recall@10:** the share of all *labelled* relevant products found in the top 10. Because labels come from a pool, this is pooled recall.
- **NDCG@10:** uses the graded labels (0/1/2) and rewards putting the best phones first.
- **MRR:** how early the first relevant phone appears.
- Also reported: a 95% paired-bootstrap confidence interval for NDCG differences, ablations (without `A`, without `Q`) and judged@10 pool coverage.

For sentiment, accuracy and macro-F1 of VADER and the weak-supervised classifier are reported on hand-labelled clauses, together with lexicon precision. If two people label the same items, Cohen's kappa is reported as well.

## Results

_Run the pipeline and `scripts/evaluate.py`, then copy the tables from `evaluation/results/ranking_summary.json` and `absa_summary.json` here._

| Method | P@5 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|
| TF-IDF | – | – | – | – |
| Semantic | – | – | – | – |
| Review-aware (hybrid) | – | – | – | – |

## Limitations and future work

- **Thin product metadata.** The dataset has titles but no descriptions or specs, so relevance relies heavily on review text.
- **Lexicon-based aspect detection** misses wording it doesn't know ("shots" for camera). Future work: embedding-based aspect detection or a fine-tuned extractor.
- **Weak labels are noisy.** A 1-star review can praise the camera. The pretrained ABSA backend (`sentiment.backend: transformer`) is available but slower.
- **Cold start:** products with fewer than 15 reviews are excluded, and aspects with fewer than 3 mentions are shown as uncertain.
- **Small evaluation set** (20 test queries). Pooled labels underestimate recall for products no method retrieved.
- **No personalisation or collaborative filtering:** the dataset has no user IDs.
- **Old listings.** Prices and phones reflect when the dataset was scraped.

## Project structure

```
config/        settings.yaml (all tunables), aspects.yaml (lexicon)
reviewrec/     core library: data, text, aspects, retrieval, ranking, explain, evaluation, pipeline
scripts/       run_pipeline, eda, make_labeling_sets, label_cli, tune_weights, evaluate
backend/app/   FastAPI app, routers, Pydantic schemas
frontend/      React + Vite + Tailwind + Recharts
tests/         pytest suite (synthetic fixture data, offline)
evaluation/    queries, labels, results
docs/          architecture, EDA, interview notes
```

## Licence

MIT for the code. The dataset is subject to its own terms on Kaggle.
