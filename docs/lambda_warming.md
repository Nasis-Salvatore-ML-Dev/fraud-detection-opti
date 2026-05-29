# Lambda Warming

## Purpose

AWS Lambda cold-starts add 1–4 seconds of latency to the first request after
a container is recycled. For a fraud-scoring API operating under SLA constraints,
this tail latency is unacceptable. Pinging `/warmup` every 5 minutes keeps at
least one container warm and the XGBoost model bundle resident in memory,
eliminating cold-start penalty for the first real prediction.

The `/warmup` endpoint is deliberately lightweight — it inspects the in-memory
state of `_bundle` and returns without loading weights or running inference.
Response time is consistently under 5 ms.

## EventBridge rule configuration

Create an EventBridge (CloudWatch Events) rule that fires every 5 minutes and
invokes the Lambda via its function URL or API Gateway:

```bash
# 1. Create the rule
aws events put-rule \
  --name fraud-api-warmup \
  --schedule-expression "rate(5 minutes)" \
  --state ENABLED \
  --region eu-central-1

# 2. Create the HTTP target (requires an EventBridge Connection for auth if needed)
#    For a public function URL with no auth, use an API destination:
aws events put-targets \
  --rule fraud-api-warmup \
  --targets '[
    {
      "Id": "warmup-target",
      "Arn": "arn:aws:events:eu-central-1:<ACCOUNT_ID>:api-destination/fraud-warmup-dest",
      "HttpParameters": {
        "PathParameterValues": [],
        "HeaderParameters": {},
        "QueryStringParameters": {}
      }
    }
  ]' \
  --region eu-central-1
```

For simpler setups, use a second Lambda (a "pinger") triggered by the
EventBridge rule that calls the scoring Lambda's function URL directly:

```python
import urllib.request

def handler(event, context):
    req = urllib.request.Request("https://<FUNCTION_URL>/warmup")
    urllib.request.urlopen(req, timeout=5)
```

## Expected cost

| Parameter | Value |
|---|---|
| Pings per hour | 12 |
| Pings per day | 288 |
| Pings per month | ~8 640 |
| Lambda invocations (warmup) | ~8 640 / month |
| Duration per warmup | < 5 ms |
| GB-seconds per month | ~0.02 (negligible) |

With AWS Lambda's 1 000 000 free-tier invocations per month, warming costs
effectively **$0.00** in invocation charges. The EventBridge rule itself is
free under the 14 400 000 scheduler invocations/month free tier.

## Verifying warming is working

Check CloudWatch Logs for the Lambda function. A healthy warmup cycle produces
log lines like:

```
INFO  fraud_detection_api  GET /warmup → 200  2ms
```

To confirm no cold-starts are occurring during business hours, set a CloudWatch
metric filter on the log group for the string `Init Duration` (which Lambda
emits only on cold-starts) and alarm if count > 0 during peak hours.

```bash
aws logs put-metric-filter \
  --log-group-name /aws/lambda/fraud-detection-api \
  --filter-name cold-start-counter \
  --filter-pattern "Init Duration" \
  --metric-transformations \
      metricName=ColdStarts,metricNamespace=FraudAPI,metricValue=1

aws cloudwatch put-metric-alarm \
  --alarm-name fraud-api-cold-starts \
  --metric-name ColdStarts \
  --namespace FraudAPI \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:eu-central-1:<ACCOUNT_ID>:<SNS_TOPIC>
```
