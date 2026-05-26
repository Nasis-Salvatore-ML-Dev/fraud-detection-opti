# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11 AS builder

# C-extension build dependencies for XGBoost
RUN yum install -y gcc gcc-c++ && yum clean all

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/packages -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.11

# Install pre-built packages from builder stage
COPY --from=builder /build/packages ${LAMBDA_TASK_ROOT}

# Application source
COPY src/    ${LAMBDA_TASK_ROOT}/src/
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models/

# Baseline data required at startup by DriftMonitor (shap_background.pkl excluded)
COPY data/baselines/training_baseline.json ${LAMBDA_TASK_ROOT}/data/baselines/

# Model card for informational endpoints
COPY model_card.json ${LAMBDA_TASK_ROOT}/

# Reports directory written to by BiasTestSuite (must exist inside container)
RUN mkdir -p ${LAMBDA_TASK_ROOT}/data/reports

# Runtime configuration
ENV MODEL_PATH=/tmp/xgboost_fraud_v1.pkl
ENV MODEL_VERSION=xgboost_fraud_v1
ENV AWS_DEFAULT_REGION=eu-central-1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Build-time metadata — injected by CI: docker build --build-arg GIT_SHA=$(git rev-parse HEAD)
ARG GIT_SHA=unknown
LABEL git.sha="${GIT_SHA}" \
      project="fraud-detection-api"

CMD ["src.api.app.handler"]
