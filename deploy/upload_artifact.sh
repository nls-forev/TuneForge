#!/usr/bin/env bash
# Uploads the evaluation artifact to S3 after the eval stage finishes. Runs regardless of
# eval exit code so logs always ship; metrics.json only exists on success, which
# the watchdog uses as the done/ok signal.
set -uo pipefail

BUCKET="${EVAL_S3_BUCKET:-tuneforge-adapters-719201730313}"
METRICS="artifact/model_evaluation/metrics.json"

if [ -f "$METRICS" ]; then
    aws s3 cp "$METRICS" "s3://${BUCKET}/evaluation/metrics.json"
else
    echo "no metrics.json — eval did not complete"
fi

if compgen -G "logs/*.log" >/dev/null; then
    aws s3 cp --recursive logs "s3://${BUCKET}/evaluation/logs/"
fi
