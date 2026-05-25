# Fraud Detection API — Optimised Pipeline

This document records the full analytical process carried out to upgrade the baseline fraud
detection pipeline to a senior-grade production ML system. The process follows a structured,
systematic approach to "inventive problem-solving": starting from a definition of what the system
should ideally deliver, moving through an honest assessment of what the baseline actually
delivers, a rigorous function analysis of every component and its interactions, identification
of all bad functions and contradictions, and finally a complete set of engineered solutions
derived from that analysis.

---

## 1. Ideality Audit

The ideality audit defines what the system must deliver in the best possible world. It is the polar star against which every engineering decision is evaluated.

### 1.1 Primary Benefit

A real-time decision engine that correctly identifies whether a financial transaction is fraudulent, delivered within the payment authorisation window, so that fraud is blocked before money leaves the account.

### 1.2 Secondary Benefits

1. Per-prediction explanation of which features drove the fraud decision, enabling compliance officers, fraud analysts, and customers to understand and contest decisions.
2. Continuous detection of model degradation (data drift, concept drift, model staleness) before performance visibly drops.
3. Decision boundary adjustable at runtime without redeployment, allowing fraud ops to tighten or relax sensitivity in response to attack campaigns.
4. Temporal and velocity patterns captured per card and per merchant, not just point-in-time transaction features.
5. No customer group experiences systematically higher false positive rates than others.
6. Every prediction immutably logged with its inputs, model version, score, and explanation, reconstructable months later.
7. Uncertain predictions routed to a fraud analyst queue with full context rather than decided by a hard threshold alone.
8. New model versions deployed and validated in staging with automatic rollback, without interrupting the payment authorisation path.
9. Rule-based fallback activates automatically when the ML model is unavailable, maintaining fraud coverage without manual intervention.
10. Inference cost per transaction within AWS free tier for portfolio demonstration; architecture scales to production cost profiles without redesign.
11. EU AI Act Articles 9–15 satisfied through architectural choices, not retrofitted as checkboxes.
12. Trained model exported to ONNX, enabling deployment on any runtime without framework lock-in and providing a measurable inference speedup.
13. Any past model version fully reconstructable from versioned data, code, and hyperparameters.
14. Pre-aggregated behavioural features available at inference time with sub-millisecond lookup.
15. Two model versions serving traffic simultaneously for statistical comparison before full rollout.

### 1.3 Problem Conditions

1. Latency: p99 < 50ms end-to-end.
2. Throughput: ≥ 1,000 requests/second at peak.
3. Availability: 99.99% uptime; rule-based fallback activates within one failed inference attempt.
4. Recall ≥ 90% on the fraud class.
5. False Positive Rate ≤ 1% on legitimate transactions.
6. AUPRC ≥ 0.90.
7. FPR parity across segments: no segment exceeds 2× overall FPR.
8. AUPRC parity across segments: no segment below 0.7× overall AUPRC.
9. All predictions logged immutably with input features, score, decision, model version, and timestamp (EU AI Act Art. 12).
10. Per-prediction explanation available on request (EU AI Act Art. 13).
11. Uncertain predictions (score in [0.3, 0.7]) routed to human review queue (EU AI Act Art. 14).
12. Bias gate on every deployment: blocks production if any fairness condition is violated (EU AI Act Art. 10).
13. Model card auto-generated and stored on every deployment (EU AI Act Art. 11).
14. Data residency: EU only (eu-central-1, Frankfurt).
15. Chronological train/validation/test split — no data leakage across time boundary.
16. Class imbalance handled explicitly in loss function.
17. PSI monitored on input features; concept drift monitored on predicted probability distribution.
18. Retraining triggered on multi-signal condition (minimum 2-of-N signals).
19. All infrastructure reproducible from code.
20. ONNX export verified numerically against original model on real data before deployment.

### 1.4 Scope of Application

**Who:** Payment processors, fintech companies, and digital banks operating in the EU.
**Where:** AWS eu-central-1 (Frankfurt). Lambda + API Gateway for bursty, pay-per-request workloads.
**Data:** Kaggle Credit Card Fraud Detection dataset (Worldline + ULB, 284,807 transactions, 0.172% fraud rate, PCA-anonymised V1–V28 + Time + Amount).
**Operating conditions:** Fraud rate between 0.1% and 2%. Transaction amount distribution stable within PSI < 0.2 of training baseline. Model valid for up to 90 days post-deployment without retraining trigger.

### 1.5 Costs

| Cost                                                                    | Classification          |
| ----------------------------------------------------------------------- | ----------------------- |
| AWS Lambda + API Gateway + DynamoDB + S3 within free tier               | Acceptable              |
| Training compute on local machine or Colab                              | Acceptable              |
| Optuna tuning runtime 1–2 hours for 100 trials                          | Acceptable              |
| Cold start latency 1–3 seconds on first Lambda invocation               | Acceptable              |
| Engineering time for online feature store                               | Acceptable — high value |
| Any inference cost making the system uneconomical at production volumes | Unacceptable            |
| Retraining costs requiring manual intervention                          | Unacceptable            |
| Vendor lock-in preventing migration without retraining                  | Unacceptable            |

### 1.6 Harms

| Harm                                                                                | Classification                         |
| ----------------------------------------------------------------------------------- | -------------------------------------- |
| SHAP disabled at serving layer due to Lambda ABI constraint                         | Acceptable if resolved architecturally |
| Cold start latency spike on Lambda warm-up                                          | Acceptable                             |
| Kaggle dataset PCA-anonymised — velocity engineering constrained to Amount and Time | Acceptable                             |
| False negatives on high-value fraud going undetected without escalation             | Unacceptable                           |
| Any production deployment bypassing the bias gate                                   | Unacceptable                           |
| Predictions made without being written to the audit log                             | Unacceptable                           |
| Model serving a version that has not passed staging smoke tests                     | Unacceptable                           |
| Silent failures — any component failure must surface explicitly                     | Unacceptable                           |

### 1.7 Goal

Deploy a fraud detection system that catches ≥ 90% of fraudulent transactions at ≤ 1% false positive rate, explains every decision, treats all customer segments fairly, and satisfies EU AI Act requirements — demonstrating that systematic engineering produces a senior-grade production ML system.

---

## 2. Baseline vs Ideality Comparison

The following table maps every ideality requirement to what the baseline pipeline (`fraud-detection-api`) actually delivers.

| Ideality requirement                            | Current pipeline state                                                                 | Problem gap                                                              | Status    |
| ----------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------- |
| Recall ≥ 90%, FPR ≤ 1%                          | AUPRC 0.7942; recall 0.76 at threshold 0.5; no threshold satisfies both simultaneously | Pipeline captures (insufficient) frauds via probabilistic classification | Partial   |
| Per-prediction SHAP explanation                 | SHAP permanently disabled — ABI crash; `/explain` returns empty `shap_values: {}`      | SHAP explainer explains (absent) fraud decisions                         | Absent    |
| Continuous drift and concept drift monitoring   | PSI on 4 features only; concept drift invisible; no retraining trigger                 | Drift monitor detects (insufficient) distribution shift                  | Partial   |
| Adaptive threshold at runtime                   | Threshold hardcoded in model bundle; requires full redeployment                        | Inference API adjusts (absent) decision threshold                        | Absent    |
| Velocity and temporal features per card         | `hour_of_day` only; stateless Lambda cannot aggregate per-card state                   | Feature engineering captures (absent) behavioural velocity signal        | Absent    |
| Fairness across all segments + directional bias | 4 fixed segments; FPR parity only; directional over-scoring invisible                  | Bias tester evaluates (insufficient) segment fairness                    | Partial   |
| Full audit log — all 31 inputs + SHAP           | Amount + 3 engineered features only; V1–V28 absent; SHAP empty                         | Audit logger records (insufficient) prediction inputs                    | Partial   |
| Human review with full context                  | Override queue routes uncertain predictions; no SHAP context for reviewer              | Override queue informs (insufficient) fraud analyst                      | Partial   |
| Zero-downtime canary deployment                 | 100% traffic switches immediately; no gradual rollout                                  | CI/CD pipeline promotes (absent) model version via canary                | Absent    |
| Rule-based fallback on model failure            | HTTP 503 on model unavailability                                                       | Inference API serves (absent) fraud score via fallback                   | Absent    |
| EU AI Act Art. 9–15 satisfied structurally      | Art. 12 partial; Art. 13 absent; Art. 14 partial; Art. 15 partial                      | Pipeline satisfies (insufficient) EU AI Act obligations                  | Partial   |
| ONNX serving — 4.2× speedup                     | ONNX export verified but never served; pickle bundle used                              | Model loader serves (absent) optimised inference via ONNX                | Absent    |
| Reproducible training — versioned artifacts     | Model versioned; hyperparameters fixed; no data versioning                             | Training pipeline versions (insufficient) experiment reproducibility     | Partial   |
| Calibrated uncertainty score                    | Hard-coded band proxy: 0.9/0.4 — not a calibrated probability                          | Inference API quantifies (harmful) prediction uncertainty via proxy      | Absent    |
| A/B testing capability                          | Single Lambda alias; no traffic splitting                                              | CI/CD pipeline compares (absent) model versions                          | Absent    |
| AWS free tier cost                              | Within free tier                                                                       | Sufficient                                                               | Satisfied |
| Automated retraining                            | No retraining trigger; manual script only                                              | Pipeline triggers (absent) automated retraining                          | Absent    |
| High-value fraud escalation                     | No amount-based escalation path                                                        | Inference API escalates (absent) high-value fraud alerts                 | Absent    |
| Explicit failure surfacing                      | SHAP failure silent; drift CloudWatch failure non-fatal                                | Pipeline surfaces (insufficient) component failures                      | Partial   |
| Bias gate never bypassed                        | Hard exit code 1 enforced in CD                                                        | Sufficient                                                               | Satisfied |
| Model serving only after staging smoke test     | CD gates on staging before production                                                  | Sufficient                                                               | Satisfied |

---

## 3. Component Analysis

### 3.1 Component Costs

| Component                              | Cost level | Dominant cost dimension                                                                                     |
| -------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------- |
| Raw dataset `creditcard.csv`           | Low        | Data debt: PCA anonymisation blocks velocity engineering and limits drift monitoring                        |
| Training pipeline `train.py`           | Low        | Technical debt: hardcoded hyperparameters produce sub-target AUPRC; Optuna tuner exists but is never called |
| Hyperparameter tuner `tune.py`         | High       | Pure opportunity cost: AUPRC gap persists entirely because this component is decoupled from training        |
| Feature engineering `preprocessing.py` | Medium     | Technical debt: hardcoded mean/std constants; only 3 features from 31-field dataset; duplicated logic       |
| Model bundle `xgboost_fraud_v1.pkl`    | Medium     | Fragility: pickle format ties system to exact library versions; silent scoring errors on mismatch           |
| SHAP background sample                 | Medium     | Complexity with zero payoff: loaded at startup, SHAP disabled; creates illusion of explainability           |
| ONNX exporter `export_onnx.py`         | High       | Opportunity cost: 4.2× speedup documented and achievable but entirely unrealised                            |
| Schema validation `schemas.py`         | Low        | Narrow scope: no range checks on V features; adversarial inputs pass silently                               |
| Inference API `app.py`                 | Medium     | Two harmful sub-functions: fake confidence band; no fallback on model failure                               |
| Decision threshold                     | High       | Rigidity: hardcoded in bundle; cannot adapt to changing fraud patterns without redeployment                 |
| Audit logger                           | Medium     | Incomplete record: V1–V28 absent; Art. 12 reconstructability partially unmet                                |
| Drift monitor `drift.py`               | High       | False confidence: monitors 4 of 31 features; concept drift entirely invisible                               |
| PSI baseline `training_baseline.json`  | Low        | Narrow scope inherited from drift monitor design                                                            |
| Bias tester `bias_tester.py`           | Medium     | Coverage gap: 4 fixed segments; directional bias invisible; compliance exposure                             |
| SHAP explainer `shap_explainer.py`     | Too High   | OP=0 at runtime; compliance theatre; consumes code complexity while delivering nothing                      |
| Model loader `model_loader.py`         | Low        | Cold start latency contribution; no ONNX loading path                                                       |
| Override queue DynamoDB                | Medium     | Context poverty: reviewer receives score and metadata but no SHAP or analogous cases                        |
| CI/CD pipeline                         | High       | Static IAM keys (security); 100% immediate traffic switch (deployment risk)                                 |
| Model card generator                   | Low        | Auto-generated; always in sync; no dominant cost                                                            |
| Load tester `locustfile.py`            | Medium     | Disconnected from CD: p99 SLA never automatically verified before deployment                                |

### 3.2 Function Analysis — OP and OE Scores

| Component              | Function              | Object                                   | OP (0–5) | OE (0–5) |
| ---------------------- | --------------------- | ---------------------------------------- | -------- | -------- |
| Raw dataset            | supplies              | labelled transaction records             | 4        | 5        |
| Training pipeline      | fits                  | XGBoost classifier                       | 4        | 5        |
| Hyperparameter tuner   | optimises             | XGBoost hyperparameters                  | 0        | 5        |
| Feature engineering    | transforms            | raw transaction fields                   | 3        | 3        |
| Model bundle           | stores                | trained model + threshold + version      | 3        | 5        |
| SHAP background sample | enables               | SHAP TreeExplainer initialisation        | 0        | 3        |
| ONNX exporter          | converts              | trained model to ONNX format             | 4        | 1        |
| Schema validation      | validates             | incoming request fields                  | 4        | 3        |
| Inference API          | serves                | fraud probability score                  | 3        | 5        |
| Decision threshold     | classifies            | fraud probability into binary decision   | 3        | 5        |
| Audit logger           | records               | prediction inputs, outputs, metadata     | 3        | 4        |
| Drift monitor          | detects               | distributional shift in input features   | 3        | 3        |
| PSI baseline           | anchors               | drift detection to training distribution | 4        | 2        |
| Bias tester            | evaluates             | model fairness across segments           | 3        | 4        |
| SHAP explainer         | explains              | per-prediction feature contributions     | 0        | 5        |
| Model loader           | loads                 | model bundle into Lambda memory          | 4        | 5        |
| Override queue         | routes                | uncertain predictions to human review    | 4        | 3        |
| CI/CD pipeline         | validates and deploys | new model and application version        | 3        | 4        |
| Model card generator   | documents             | model metadata and performance           | 5        | 2        |
| Load tester            | measures              | API latency under load                   | 3        | 3        |

---

## 4. Function Model

The function model maps every component interaction as a directed arrow typed by its quality.
The interactive function model is available at [`docs/function_model.html`](docs/function_model.html).
Open it in a browser to inspect all component interactions, arrow types, and supersystem nodes.

**Arrow key:**

- Green solid → Useful function
- Green dashed → Insufficient function (delivers, but too weak or too narrow)
- Red solid → Harmful function
- Grey dotted → Absent function (needed but not delivered)
- Dashed box → Absent component (needed but not built)

**Supersystem nodes** (external actors): Transaction System / Payment Gateway, Fraudulent Users, Legitimate Customers, Regulatory Bodies (EU AI Act / GDPR), Fraud Analysts, IT Infrastructure (AWS Cloud).

### 4.1 Identified Contradictions

| ID  | Contradiction                                                                                                  |
| --- | -------------------------------------------------------------------------------------------------------------- |
| C1  | Inference API → Legitimate Customers: useful: approves legitimate \| harmful: false positive blocks legitimate |
| C2  | Inference API → SHAP Explainer: useful: calls explainer \| harmful: crashes explainer via ABI mismatch         |
| C3  | Audit Logger → Regulatory Bodies: useful: evidences compliance \| harmful: incomplete record violates Art. 12  |
| C4  | Decision Threshold → Fraud: useful: blocks fraud \| harmful: rigid threshold misses adaptive fraud patterns    |
| C5  | Model Bundle → Model Loader: useful: provides model \| harmful: pickle lock-in creates silent version skew     |
| C6  | CI/CD Pipeline → IT Infrastructure: useful: deploys service \| harmful: static IAM keys expose credentials     |
| C7  | Bias Tester → Regulatory Bodies: useful: gates biased models \| insufficient: 4 segments miss directional bias |

### 4.2 All Bad Functions

**Harmful**

- H1. Fraudulent users — injects (harmful) — adversarial transactions into Transaction System
- H2. Inference API — blocks (harmful) — legitimate customers via false positive
- H3. Inference API — misleads (harmful) — downstream consumers via fake confidence score
- H4. Inference API — crashes (harmful) — SHAP explainer via ABI mismatch
- H5. Model bundle — locks (harmful) — runtime portability via pickle serialisation
- H6. CI/CD pipeline — exposes (harmful) — cloud credentials via static IAM keys
- H7. Audit logger — violates (harmful) — Art. 12 reconstructability via incomplete record
- H8. Feature engineering — introduces (harmful) — train-serve skew via hardcoded mean/std constants
- H9. Decision threshold — misses (harmful) — adaptive fraud patterns via rigid hardcoded value

**Insufficient**

- I1. HPO tuner — optimises (insufficient) — XGBoost hyperparameters (never executed)
- I2. Training pipeline — exports (insufficient) — ONNX model (disconnected from serving path)
- I3. Feature engineering — transforms (insufficient) — raw inputs (3 features; no velocity signal)
- I4. Schema validation — validates (insufficient) — incoming requests (no range or outlier checks)
- I5. SHAP background — enables (insufficient) — SHAP explainer initialisation (explainer disabled)
- I6. Audit logger — feeds (insufficient) — drift monitor (V1–V28 absent from records)
- I7. Override queue — routes (insufficient) — fraud analyst context (no SHAP, no similar cases)
- I8. Drift monitor — publishes (insufficient) — PSI metrics (4 features; concept drift blind)
- I9. Bias tester — evaluates (insufficient) — segment fairness (4 segments; no directional metric)
- I10. Load tester — measures (insufficient) — latency SLA (disconnected from CD gate)
- I11. Drift monitor — triggers (insufficient) — retraining signal (no multi-signal logic exists)
- I12. Bias tester — evidences (insufficient) — regulatory compliance (coverage too narrow)
- I13. SHAP explainer — populates (insufficient) — audit log SHAP values (OP=0 at runtime)

**Absent**

- A1. Online feature store — aggregates (absent) — velocity features per card and merchant
- A2. Retraining trigger — initiates (absent) — automated retraining on multi-signal condition
- A3. Calibrated confidence — quantifies (absent) — prediction uncertainty via Platt/isotonic scaling
- A4. Rule-based fallback — scores (absent) — transactions when model is unavailable
- A5. Canary deployer — promotes (absent) — model version via weighted Lambda alias
- A6. Inference API — escalates (absent) — high-value fraud alerts via risk-tiering logic
- A7. CI/CD pipeline — validates (absent) — model versions via shadow evaluation

---

## 5. Trimming Decisions

| ID  | Component trimmed                                       | What is removed                                                          | How the function is preserved                                                                                                                             |
| --- | ------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T1  | Feature engineering — duplicated implementation         | Separate training and serving implementations of the same logic          | Single canonical module in `src/features/engineering.py` imported by both `train.py` and `preprocessing.py`; hardcoded mean/std removed into model bundle |
| T2  | SHAP background sample — serving artifact               | `shap_background.pkl` shipped to Lambda                                  | Used once at training time for offline SHAP computation; not included in Lambda deployment package                                                        |
| T3  | ONNX exporter — standalone script                       | `scripts/export_onnx.py` as separate step                                | ONNX export absorbed as final step of `train.py`; automatic output of every training run                                                                  |
| T4  | Schema validation — structural overlap with API Gateway | Pydantic field-presence and type checks duplicated at gateway and Lambda | Move structural checks upstream to API Gateway request models; Pydantic retains range and outlier checks only                                             |
| T5  | PSI baseline — standalone JSON artifact                 | `data/baselines/training_baseline.json` as separately versioned file     | Embedded in model bundle at training time; loaded by Model Loader; always in sync with model that generated it                                            |
| T6  | Model card generator — standalone CD script             | `scripts/generate_model_card.py` running in CD                           | Absorbed into `train.py` as post-fit step; model card uploaded to S3 as training artifact                                                                 |
| T7  | Override queue — separate DynamoDB table                | `fraud-override-queue` as a second DynamoDB table                        | Merged into Audit Logger as `requires_review` flag with conditional 30-day TTL on the same `fraud-audit-log` table                                        |
| T8  | Load tester — standalone Locust CD dependency           | Locust running in CD pipeline                                            | p99 latency assertion (20 concurrent `httpx` requests, assert p99 < 50ms) embedded in CD smoke test; Locust retained for manual deep testing              |

---

## 6. Master Solution Set

The following 42 solutions and 8 trimming decisions define the complete upgrade from baseline to optimised pipeline.

| #   | Problem                                                             | Resolution                                                                                                                                                     |
| --- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Pipeline captures (insufficient) frauds                             | Combine fast shallow model + tuned XGBoost ensemble; integrate Optuna (100 trials, AUPRC objective) into `train.py` via `--tune` flag                          |
| 2   | SHAP explainer explains (absent) fraud decisions                    | Compute SHAP offline post-training; store pre-computed values in DynamoDB keyed by prediction hash; Lambda reads from storage — never imports SHAP             |
| 3   | Drift monitor detects (insufficient) distribution shift             | Add feature extraction bridge to read V1–V28 from full audit records; add second PSI monitor on predicted probability distribution as concept drift proxy      |
| 4   | Inference API adjusts (absent) decision threshold                   | Store threshold in DynamoDB `fraud-config` table; Model Loader reads at startup; `/config` endpoint for runtime updates                                        |
| 5   | Feature engineering captures (absent) velocity signal               | Add DynamoDB `fraud-velocity-store`; feature engineering queries per-card aggregates at inference time; async Lambda writes new transaction post-score         |
| 6   | Bias tester evaluates (insufficient) segment fairness               | Dynamic segment config file; add directional metric (mean predicted fraud probability per segment); asymmetric FPR thresholds by transaction value tier        |
| 7   | Audit logger records (insufficient) prediction inputs               | Add all 31 input features to DynamoDB audit record; compress V1–V28 with msgpack (~300 bytes)                                                                  |
| 8   | Override queue informs (insufficient) fraud analyst                 | At queue-write time retrieve pre-computed SHAP values and top-3 features; add nearest-neighbour lookup for 3 analogous resolved cases; embed in queue record   |
| 9   | CI/CD pipeline promotes (absent) model via canary                   | Lambda weighted alias routing (10% canary); CloudWatch alarm on error rate + fraud flag rate; auto-promote or rollback after 30-minute window                  |
| 10  | Inference API serves (absent) fraud score via fallback              | Add heuristic scorer (Amount z-score > 3 OR hour in [1,5] → suspicious) to lifespan handler; activates when bundle is None                                     |
| 11  | Pipeline satisfies (insufficient) EU AI Act obligations             | Generate structured compliance manifest JSON at CD time mapping Art. 9–15 to implementing components; add manifest gate blocking deployment on absent articles |
| 12  | Model loader serves (absent) optimised inference                    | Replace pickle loading with `onnxruntime.InferenceSession`; integrate ONNX export as final step of `train.py`; 4.2× speedup realised                           |
| 13  | Training pipeline versions (insufficient) reproducibility           | Write experiment manifest JSON at end of every training run (git SHA, dataset hash, hyperparameters, metrics, S3 paths)                                        |
| 14  | Inference API quantifies (harmful) uncertainty via proxy            | Switch off hard-coded band proxy; fit isotonic regression calibration on validation set at training time; store calibrator in model bundle                     |
| 15  | CI/CD pipeline compares (absent) model versions                     | Shadow evaluation in CD: replay last 1,000 audit log records through champion and challenger offline; compare AUPRC, recall, FPR                               |
| 16  | Lambda runtime delays (insufficient) first-request                  | CloudWatch Events warming rule (ping `/health` every 5 min); ONNX serving reduces cold start from ~2s to ~300ms                                                |
| 17  | Model bundle constrains (harmful) runtime portability               | Switch off pickle as serving format; ONNX becomes primary serving artifact; pickle retained only as training checkpoint                                        |
| 18  | Pipeline triggers (absent) automated retraining                     | Retraining orchestrator Lambda on daily CloudWatch Events schedule; reads 4 signals; triggers GitHub Actions workflow dispatch on 2-of-4 condition             |
| 19  | Inference API escalates (absent) high-value fraud alerts            | Tiered response: Amount > €1,000 AND fraud_probability > 0.3 → escalation flag + SNS notification to high-value alert queue                                    |
| 20  | Pipeline surfaces (insufficient) component failures                 | Add `ComponentFailure` CloudWatch metric published from every component catch block; single alarm on any dimension triggers SNS alert                          |
| 21  | Fraudulent users inject (harmful) adversarial transactions          | Multivariate anomaly detection at schema validation: flag if V_i exceeds ±10σ OR cosine similarity to fraud centroids > 0.98                                   |
| 22  | Inference API blocks (harmful) legitimate customers via FP          | Resolved false positives become labelled hard-negative training examples; replace hard block with step-up authentication (OTP/3DS)                             |
| 23  | Inference API misleads (harmful) via fake confidence                | Identical to #14: isotonic regression calibration replaces band proxy                                                                                          |
| 24  | Inference API crashes (harmful) SHAP explainer via ABI              | Identical to #2: offline SHAP computation; Lambda never imports SHAP                                                                                           |
| 25  | Model bundle locks (harmful) runtime portability                    | Identical to #17: ONNX replaces pickle in serving path                                                                                                         |
| 26  | CI/CD pipeline exposes (harmful) credentials via static keys        | Replace static IAM keys with OIDC (`aws-actions/configure-aws-credentials`, `role-to-assume`); token expires at job end                                        |
| 27  | Audit logger violates (harmful) Art. 12 reconstructability          | Identical to #7: add V1–V28 msgpack-compressed to every audit record                                                                                           |
| 28  | Feature engineering introduces (harmful) train-serve skew           | Compute mean/std from training data; store in model bundle; `preprocessing.py` reads from loaded bundle                                                        |
| 29  | Decision threshold misses (harmful) adaptive fraud patterns         | Dynamic threshold from DynamoDB config (#4); time-aware pattern: threshold tightens during high-risk hours (01:00–05:00)                                       |
| 30  | HPO tuner optimises (insufficient) hyperparameters (never run)      | Embed Optuna in `train.py` behind `--tune` flag; 100 trials, AUPRC objective, chronological val split                                                          |
| 31  | Training pipeline exports (insufficient) ONNX (disconnected)        | ONNX export as final step of `train.py`; upload to S3; Model Loader defaults to ONNX                                                                           |
| 32  | Feature engineering transforms (insufficient) raw inputs            | Identical to #5: add velocity store; feature engineering queries per-card temporal aggregates                                                                  |
| 33  | Schema validation validates (insufficient) incoming requests        | Add V-feature ±10σ bounds validators and Amount z-score > 5 warning to Pydantic schema                                                                         |
| 34  | SHAP background enables (insufficient) disabled explainer           | Used once at training time; not shipped to Lambda; disappears after delivering useful effect                                                                   |
| 35  | Audit logger feeds (insufficient) drift monitor (missing V1–V28)    | Resolved by #7: once audit record contains all 31 features, drift monitor reads them directly                                                                  |
| 36  | Override queue routes (insufficient) analyst context                | Identical to #8: context assembly at queue-write time embeds SHAP + nearest-neighbour cases                                                                    |
| 37  | Drift monitor publishes (insufficient) PSI (4 features, no concept) | Combine input-feature PSI (extended via #35) with output-distribution PSI; two monitors, one CloudWatch publish cycle                                          |
| 38  | Bias tester evaluates (insufficient) fairness (4 segments)          | Identical to #6: dynamic config + directional metric + asymmetric thresholds                                                                                   |
| 39  | Load tester measures (insufficient) latency (disconnected from CD)  | Embed 20-request httpx p99 assertion into CD smoke test step                                                                                                   |
| 40  | Drift monitor triggers (insufficient) retraining (no logic)         | Identical to #18: retraining orchestrator Lambda with 2-of-4 multi-signal condition                                                                            |
| 41  | Bias tester evidences (insufficient) compliance (narrow)            | Combine bias report with compliance manifest (#11); Art. 10 marked satisfied only when all segments pass both metrics                                          |
| 42  | SHAP explainer populates (insufficient) audit log (OP=0)            | Identical to #2: pre-computed SHAP in DynamoDB; audit logger reads at prediction time                                                                          |
| T1  | Feature engineering duplicated implementation                       | Single canonical `src/features/engineering.py`; mean/std into bundle                                                                                           |
| T2  | SHAP background sample in serving artifact                          | Remove from Lambda package; training-only use                                                                                                                  |
| T3  | ONNX exporter standalone script                                     | Absorb into `train.py`                                                                                                                                         |
| T4  | Schema validation overlap with API Gateway                          | Structural checks upstream; Pydantic for range/outlier only                                                                                                    |
| T5  | PSI baseline standalone JSON                                        | Embed in model bundle                                                                                                                                          |
| T6  | Model card generator standalone CD script                           | Absorb into `train.py` post-fit                                                                                                                                |
| T7  | Override queue separate DynamoDB table                              | Merge into audit log with `requires_review` flag                                                                                                               |
| T8  | Load tester standalone Locust CD dependency                         | p99 gate in CD smoke test; Locust for manual use only                                                                                                          |

---

## 7. Architecture — Optimised Pipeline

### 7.1 New and Modified Components

| Component                              | Status   | Change                                                                                                                                                     |
| -------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/features/engineering.py`          | New      | Canonical feature engineering module (canonical, shared by training and serving)                                                                           |
| `src/features/velocity.py`             | New      | DynamoDB velocity store client — per-card aggregates at inference time                                                                                     |
| `scripts/train.py`                     | Modified | Absorbs Optuna tuning, ONNX export, model card generation, experiment manifest, PSI baseline, isotonic calibration                                         |
| `src/utils/model_loader.py`            | Modified | Loads ONNX via onnxruntime; reads threshold and PSI baseline from bundle; reads threshold from DynamoDB config                                             |
| `src/api/app.py`                       | Modified | Rule-based fallback; risk-tiering escalation; ComponentFailure CloudWatch metric; `/config` endpoint; warming handler                                      |
| `src/api/schemas.py`                   | Modified | V-feature ±10σ validators; Amount z-score outlier detection                                                                                                |
| `src/api/preprocessing.py`             | Modified | Reads mean/std from bundle; calls canonical engineering module                                                                                             |
| `src/monitoring/audit_logger.py`       | Modified | Stores all 31 features (msgpack-compressed); `requires_review` flag replaces separate override queue table; reads SHAP from DynamoDB pre-computation store |
| `src/monitoring/drift.py`              | Modified | Extended to V1–V28 from audit records; adds output-distribution PSI for concept drift; adds retraining signal publishing                                   |
| `src/monitoring/bias_tester.py`        | Modified | Dynamic segment config; directional metric; asymmetric thresholds                                                                                          |
| `src/monitoring/compliance.py`         | New      | Compliance manifest generator mapping Art. 9–15 to evidence artifacts                                                                                      |
| `src/monitoring/retraining_trigger.py` | New      | Multi-signal retraining orchestrator (2-of-4 condition)                                                                                                    |
| `src/explainability/shap_offline.py`   | New      | Offline SHAP batch computation post-training; stores to DynamoDB                                                                                           |
| `.github/workflows/cd.yml`             | Modified | OIDC auth; Lambda weighted alias canary; shadow evaluation; compliance manifest gate; p99 latency assertion in smoke test                                  |
| `infra/scripts/create_dynamodb.sh`     | Modified | Adds `fraud-config` table; adds `fraud-velocity-store` table; adds `fraud-shap-store` table; merges override queue into audit log                          |

### 7.2 Runtime Request Flow (Optimised)

```
Client
  → API Gateway (structural schema enforcement)
  → Lambda / FastAPI (Mangum)
  → Pydantic validation (range checks, outlier detection)
  → Velocity store lookup (DynamoDB — per-card aggregates)
  → Canonical feature engineering (31 features + velocity)
  → Anomaly detection (cosine similarity, V-feature bounds)
  → ONNX inference (onnxruntime, ~5ms)
  → Risk tiering (Amount > €1,000 + probability check)
  → Isotonic calibration (calibrated confidence score)
  → Dynamic threshold application (from DynamoDB config)
  → Audit log write (all 31 features + SHAP from pre-computation store)
  → Override queue flag (requires_review on audit record)
  → Response (prediction_id, is_fraud, fraud_probability, confidence_score, shap_values)
```

### 7.3 EU AI Act Compliance Map

| Article | Requirement             | Implementation                                                                         |
| ------- | ----------------------- | -------------------------------------------------------------------------------------- |
| Art. 9  | Risk management         | Bias gate + anomaly detection + high-value escalation                                  |
| Art. 10 | Data governance         | Dynamic segment bias testing with directional metric                                   |
| Art. 11 | Technical documentation | Model card generated at training time; experiment manifest                             |
| Art. 12 | Record-keeping          | Audit log with all 31 features + SHAP; append-only; permanent                          |
| Art. 13 | Transparency            | Pre-computed SHAP served per prediction; `/explain` endpoint                           |
| Art. 14 | Human oversight         | `requires_review` flag; override queue with SHAP context                               |
| Art. 15 | Accuracy and robustness | ONNX serving; input + concept drift monitoring; rule-based fallback; canary deployment |

---

_Dataset citation: Dal Pozzolo, Andrea et al. — Calibrating Probability with Undersampling for Unbalanced Classification. IEEE CIDM 2015. Worldline + ULB Machine Learning Group._
