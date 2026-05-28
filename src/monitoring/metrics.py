"""CloudWatch metric publisher for component failures."""

import logging
import os

log = logging.getLogger(__name__)


def publish_component_failure(component_name: str) -> None:
    """Put a ComponentFailure metric to CloudWatch. Never raises."""
    try:
        import boto3
        region = os.environ.get("AWS_REGION", "eu-central-1")
        cw = boto3.client("cloudwatch", region_name=region)
        cw.put_metric_data(
            Namespace="FraudDetection",
            MetricData=[{
                "MetricName": "ComponentFailure",
                "Dimensions": [{"Name": "ComponentName", "Value": component_name}],
                "Value": 1,
                "Unit": "Count",
            }],
        )
        log.debug("ComponentFailure metric published for %s", component_name)
    except Exception as exc:
        log.debug("ComponentFailure metric publish failed for %s: %s", component_name, exc)
