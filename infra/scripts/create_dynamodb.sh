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
#    PK: prediction_id (S)   SK: timestamp (S)
#    GSI: model_version-timestamp-index
#    Billing: PAY_PER_REQUEST   TTL: none (permanent)
# ---------------------------------------------------------------------------
TABLE_AUDIT="fraud-audit-log"

if table_exists "$TABLE_AUDIT"; then
    echo "[SKIP]    $TABLE_AUDIT — already exists"
else
    aws dynamodb create-table \
        --table-name "$TABLE_AUDIT" \
        --region "$REGION" \
        --billing-mode PAY_PER_REQUEST \
        --attribute-definitions \
            AttributeName=prediction_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
            AttributeName=model_version,AttributeType=S \
        --key-schema \
            AttributeName=prediction_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --global-secondary-indexes '[
            {
                "IndexName": "model_version-timestamp-index",
                "KeySchema": [
                    {"AttributeName": "model_version", "KeyType": "HASH"},
                    {"AttributeName": "timestamp",     "KeyType": "RANGE"}
                ],
                "Projection": {"ProjectionType": "ALL"}
            }
        ]' \
        --output text \
        --query "TableDescription.TableName" \
        > /dev/null

    echo "[CREATED] $TABLE_AUDIT"
fi

# ---------------------------------------------------------------------------
# 2. fraud-override-queue
#    PK: prediction_id (S)   SK: timestamp (S)
#    Billing: PAY_PER_REQUEST   TTL: ttl (epoch seconds, 30-day expiry)
# ---------------------------------------------------------------------------
TABLE_OVERRIDE="fraud-override-queue"

if table_exists "$TABLE_OVERRIDE"; then
    echo "[SKIP]    $TABLE_OVERRIDE — already exists"
else
    aws dynamodb create-table \
        --table-name "$TABLE_OVERRIDE" \
        --region "$REGION" \
        --billing-mode PAY_PER_REQUEST \
        --attribute-definitions \
            AttributeName=prediction_id,AttributeType=S \
            AttributeName=timestamp,AttributeType=S \
        --key-schema \
            AttributeName=prediction_id,KeyType=HASH \
            AttributeName=timestamp,KeyType=RANGE \
        --output text \
        --query "TableDescription.TableName" \
        > /dev/null

    aws dynamodb wait table-exists \
        --table-name "$TABLE_OVERRIDE" \
        --region "$REGION"

    aws dynamodb update-time-to-live \
        --table-name "$TABLE_OVERRIDE" \
        --region "$REGION" \
        --time-to-live-specification \
            Enabled=true,AttributeName=ttl \
        > /dev/null

    echo "[CREATED] $TABLE_OVERRIDE  (TTL: ttl)"
fi

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

echo ""
echo "DynamoDB setup complete."
