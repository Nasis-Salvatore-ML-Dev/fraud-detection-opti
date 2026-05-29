# Retraining Orchestrator

## Purpose

`src/monitoring/retraining_trigger.py` evaluates four CloudWatch drift signals on a daily schedule and dispatches the `train.yml` GitHub Actions workflow when 2 or more signals fire simultaneously. It is designed to run as a scheduled AWS Lambda triggered by CloudWatch Events (EventBridge).

Signals evaluated:

| Signal | CloudWatch metric | Threshold |
|---|---|---|
| `input_drift` | `FraudPSI` | any datapoint > 0.2 in last 24h |
| `model_age` | `ModelAgeDays` | latest > 90 |
| `fraud_flag_rate` | `FraudFlagRateDelta` | latest > 0.1 |
| `concept_drift` | `ConceptDriftPSI` | latest > 0.2 |

A model age guard prevents retraining if the model bundle was trained within the last 7 days (requires `MODEL_S3_BUCKET` and `MODEL_S3_KEY`).

---

## Required environment variables

| Variable | Required | Description |
|---|---|---|
| `GITHUB_REPO` | Yes | Repository in `owner/repo` format. |
| `GITHUB_TOKEN` | Yes | GitHub token with `actions:write` permission. |
| `MODEL_S3_BUCKET` | No | S3 bucket for model bundle (model age guard). |
| `MODEL_S3_KEY` | No | S3 key for model bundle pkl (model age guard). |
| `AWS_DEFAULT_REGION` | No | AWS region for CloudWatch reads (default: `us-east-1`). |

---

## Deployment instructions

### 1. Package the Lambda

```bash
pip install -r requirements.txt -t package/
cp -r src/ package/
cp scripts/run_retraining_check.py package/
cd package && zip -r ../retraining_orchestrator.zip . && cd ..
```

### 2. Create the Lambda function

```bash
aws lambda create-function \
  --function-name fraud-retraining-orchestrator \
  --runtime python3.12 \
  --role arn:aws:iam::<ACCOUNT_ID>:role/fraud-lambda-role \
  --handler src.monitoring.retraining_trigger.check_and_trigger \
  --zip-file fileb://retraining_orchestrator.zip \
  --timeout 60 \
  --environment Variables="{
    GITHUB_REPO=owner/repo,
    GITHUB_TOKEN=<token>,
    MODEL_S3_BUCKET=fraud-model-artifacts-<account>,
    MODEL_S3_KEY=models/xgboost_fraud_v1.pkl,
    AWS_DEFAULT_REGION=us-east-1
  }"
```

### 3. Create the EventBridge rule (daily trigger)

```bash
# Create rule to fire at 06:00 UTC every day
aws events put-rule \
  --name fraud-retraining-daily \
  --schedule-expression "cron(0 6 * * ? *)" \
  --state ENABLED

# Attach Lambda as target
aws events put-targets \
  --rule fraud-retraining-daily \
  --targets "Id=1,Arn=arn:aws:lambda:<REGION>:<ACCOUNT_ID>:function:fraud-retraining-orchestrator"

# Grant EventBridge permission to invoke the Lambda
aws lambda add-permission \
  --function-name fraud-retraining-orchestrator \
  --statement-id AllowEventBridgeInvoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:<REGION>:<ACCOUNT_ID>:rule/fraud-retraining-daily
```

---

## Manual invocation

**Dry run** (evaluates signals, no dispatch):

```bash
PYTHONPATH=. python scripts/run_retraining_check.py --dry-run
```

**Live run** (dispatches GitHub Actions if 2+ signals fire):

```bash
GITHUB_REPO=owner/repo GITHUB_TOKEN=<token> \
  PYTHONPATH=. python scripts/run_retraining_check.py
```

**Via AWS CLI** (invoke deployed Lambda):

```bash
aws lambda invoke \
  --function-name fraud-retraining-orchestrator \
  --payload '{"dry_run": true}' \
  response.json && cat response.json
```
