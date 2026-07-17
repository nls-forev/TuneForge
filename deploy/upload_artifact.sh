#!/usr/bin/env bash
# Upload generate (phase A) outputs to S3; logs always ship, responses only on success.
set -uo pipefail

BUCKET="${EVAL_S3_BUCKET:-tuneforge-adapters-719201730313}"
RESPONSES="artifact/model_evaluation/responses.parquet"

if [ -f "$RESPONSES" ]; then
    aws s3 cp "$RESPONSES" "s3://${BUCKET}/evaluation/responses.parquet"
else
    echo "no responses.parquet — generate did not complete"
fi

if compgen -G "logs/*.log" >/dev/null; then
    aws s3 cp --recursive logs "s3://${BUCKET}/evaluation/logs/"
fi
