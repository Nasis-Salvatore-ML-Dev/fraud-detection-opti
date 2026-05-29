# Environment Variables

## Runtime configuration

| Variable | Default | Description |
|---|---|---|
| `CONFIG_TABLE` | `fraud-config` | DynamoDB table name for runtime configuration (threshold). |
| `CONFIG_API_KEY` | *(required for POST /config)* | API key checked against the `X-Config-Api-Key` request header. If unset, POST /config returns 503. |
| `AUDIT_TABLE` | `fraud-audit-log` | DynamoDB table for prediction audit records. |
| `SHAP_TABLE` | `fraud-shap-store` | DynamoDB table for pre-computed SHAP values. |
| `AWS_DEFAULT_REGION` | `eu-central-1` | AWS region for all DynamoDB operations. |
| `MODEL_PATH` | `/tmp/xgboost_fraud_v1.pkl` | Local path to the model bundle pkl. |
| `MODEL_S3_BUCKET` | `fraud-model-artifacts-209998132741` | S3 bucket from which the model bundle and ONNX model are downloaded on cold-start. |
| `MODEL_S3_KEY` | `models/xgboost_fraud_v1.pkl` | S3 key for the pkl bundle. |
| `BASELINE_PATH` | *(auto-resolved)* | Path to `training_baseline.json`; auto-set after S3 download. |
| `HIGH_VALUE_SNS_ARN` | *(optional)* | SNS topic ARN for high-value fraud alerts (`Amount > 1000` and `fraud_probability > 0.3`). If unset, alerts are skipped with a warning log. |
| `AWS_REGION` | `eu-central-1` | AWS region used by CloudWatch for `ComponentFailure` metric publishing (distinct from `AWS_DEFAULT_REGION` used by DynamoDB/SNS). |
| `BIAS_SEGMENTS_PATH` | *(auto-resolved)* | Path to a JSON file defining bias evaluation segments. Defaults to `data/baselines/bias_segments.json`. Each entry must include `name`, `feature`, `threshold`, `comparison` (gt/gte/lt/lte), `fpr_multiplier_limit`, and `auprc_ratio_limit`. |
| `VELOCITY_TABLE` | `fraud-velocity-store` | DynamoDB table for per-card velocity features (transaction counts, amount sums, time-since-last-tx). Items expire after 7 days via TTL on `expires_at`. Optional — if DynamoDB is unavailable, zero defaults are used and prediction continues. |

## CI/CD pipeline

| Variable | Default | Description |
|---|---|---|
| `AWS_ROLE_ARN` | *(required)* | IAM role ARN assumed via OIDC in CI jobs that deploy to AWS (deploy-staging, shadow-eval, deploy-production). Replaces static `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. |
| `STAGING_ENDPOINT` | *(required)* | Base URL of the staging Lambda (e.g. `https://abc123.execute-api.eu-central-1.amazonaws.com/staging`). Used by smoke-test (health check + latency gate) and shadow-eval. |

## Retraining orchestrator

The `src/monitoring/retraining_trigger.py` module is designed to run as a scheduled AWS Lambda triggered by CloudWatch Events (EventBridge) on a daily schedule. It reads 4 CloudWatch drift signals and dispatches the `train.yml` GitHub Actions workflow when 2 or more signals fire.

| Variable | Default | Required | Description |
|---|---|---|---|
| `GITHUB_REPO` | *(none)* | Yes | GitHub repository in `owner/repo` format. Used to check in-progress workflow runs and dispatch `train.yml`. |
| `GITHUB_TOKEN` | *(none)* | Yes | GitHub personal access token or Actions token with `actions:write` permission for workflow dispatch. |
| `MODEL_S3_BUCKET` | *(none)* | No | S3 bucket containing the model bundle pkl. Used by the model age guard to skip retraining for recently-trained models (< 7 days). If unset, guard is skipped. |
| `MODEL_S3_KEY` | `models/xgboost_fraud_v1.pkl` | No | S3 key for the model bundle pkl. Used alongside `MODEL_S3_BUCKET` for the model age guard. |
