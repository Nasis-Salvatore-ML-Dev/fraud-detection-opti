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
