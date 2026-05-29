#!/usr/bin/env bash
# Create DynamoDB tables for the fraud-detection-api.
# Idempotent — safe to run multiple times.
#
# Usage:
#   ./infra/scripts/create_dynamodb.sh
#   REGION=us-east-1 ./infra/scripts/create_dynamodb.sh

set -euo pipefail

REGION="${REGION:-eu-central-1}"

# ---------------------------------------------------------------------------
# Helper: return 0 if table exists, 1 if not
# ---------------------------------------------------------------------------
table_exists() {
    aws dynamodb describe-table \
        --table-name "$1" \
        --region "$REGION" \
        --output text \
        --query "Table.TableName" \
        > /dev/null 2>&1
}

echo "Region: $REGION"
echo ""

# ---------------------------------------------------------------------------
# 1. fraud-audit-log
#    PK: prediction_id (S)
#    GSI: model_version-timestamp-index
#    GSI: requires-review-index (partition key: requires_review S)
#    TTL: review_expires_at (epoch seconds, 30-day expiry on flagged records)
#    Billing: PAY_PER_REQUEST
# ---------------------------------------------------------------------------
TABLE_AUDIT="fraud-audit-log"

aws dynamodb create-table \
    --table-name "$TABLE_AUDIT" \
    --region "$REGION" \
    --billing-mode PAY_PER_REQUEST \
    --attribute-definitions \
        AttributeName=prediction_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
        AttributeName=model_version,AttributeType=S \
        AttributeName=requires_review,AttributeType=S \
    --key-schema \
        AttributeName=prediction_id,KeyType=HASH \
    --global-secondary-indexes '[
        {
            "IndexName": "model_version-timestamp-index",
            "KeySchema": [
                {"AttributeName": "model_version", "KeyType": "HASH"},
                {"AttributeName": "timestamp",     "KeyType": "RANGE"}
            ],
            "Projection": {"ProjectionType": "ALL"}
        },
        {
            "IndexName": "requires-review-index",
            "KeySchema": [
                {"AttributeName": "requires_review", "KeyType": "HASH"}
            ],
            "Projection": {"ProjectionType": "ALL"}
        }
    ]' \
    --output text \
    --query "TableDescription.TableName" \
    > /dev/null 2>&1 || true

aws dynamodb update-time-to-live \
    --table-name "$TABLE_AUDIT" \
    --region "$REGION" \
    --time-to-live-specification \
        Enabled=true,AttributeName=review_expires_at \
    > /dev/null 2>&1 || true

echo "[OK]      $TABLE_AUDIT  (TTL: review_expires_at, GSI: requires-review-index)"

# ---------------------------------------------------------------------------
# 3. fraud-shap-store
#    PK: prediction_hash (S)
#    Billing: PAY_PER_REQUEST   TTL: expires_at (epoch seconds, 90-day expiry)
# ---------------------------------------------------------------------------
TABLE_SHAP="fraud-shap-store"

aws dynamodb create-table \
    --table-name "$TABLE_SHAP" \
    --region "$REGION" \
    --billing-mode PAY_PER_REQUEST \
    --attribute-definitions \
        AttributeName=prediction_hash,AttributeType=S \
    --key-schema \
        AttributeName=prediction_hash,KeyType=HASH \
    --output text \
    --query "TableDescription.TableName" \
    > /dev/null 2>&1 || true

aws dynamodb update-time-to-live \
    --table-name "$TABLE_SHAP" \
    --region "$REGION" \
    --time-to-live-specification \
        Enabled=true,AttributeName=expires_at \
    > /dev/null 2>&1 || true

echo "[OK]      $TABLE_SHAP  (TTL: expires_at)"

# ---------------------------------------------------------------------------
# 3. fraud-config
#    PK: config_key (S)
#    Billing: PAY_PER_REQUEST
#    Seed: threshold = 0.5 (written only on first table creation)
# ---------------------------------------------------------------------------
TABLE_CONFIG="fraud-config"

if aws dynamodb create-table \
    --table-name "$TABLE_CONFIG" \
    --region "$REGION" \
    --billing-mode PAY_PER_REQUEST \
    --attribute-definitions \
        AttributeName=config_key,AttributeType=S \
    --key-schema \
        AttributeName=config_key,KeyType=HASH \
    --output text \
    --query "TableDescription.TableName" \
    > /dev/null 2>&1; then
    aws dynamodb put-item \
        --table-name "$TABLE_CONFIG" \
        --region "$REGION" \
        --item '{"config_key": {"S": "threshold"}, "value": {"S": "0.5"}}' \
        --condition-expression "attribute_not_exists(config_key)" \
        > /dev/null 2>&1 || true
fi

echo "[OK]      $TABLE_CONFIG  (seed: threshold=0.5)"

# ---------------------------------------------------------------------------
# 4. fraud-velocity-store
#    PK: card_hash (S)
#    Billing: PAY_PER_REQUEST   TTL: expires_at (7-day expiry per card)
# ---------------------------------------------------------------------------
TABLE_VELOCITY="fraud-velocity-store"

aws dynamodb create-table \
    --table-name "$TABLE_VELOCITY" \
    --region "$REGION" \
    --billing-mode PAY_PER_REQUEST \
    --attribute-definitions \
        AttributeName=card_hash,AttributeType=S \
    --key-schema \
        AttributeName=card_hash,KeyType=HASH \
    --output text \
    --query "TableDescription.TableName" \
    > /dev/null 2>&1 || true

aws dynamodb update-time-to-live \
    --table-name "$TABLE_VELOCITY" \
    --region "$REGION" \
    --time-to-live-specification \
        Enabled=true,AttributeName=expires_at \
    > /dev/null 2>&1 || true

echo "[OK]      $TABLE_VELOCITY  (TTL: expires_at)"

echo ""
echo "DynamoDB setup complete."
