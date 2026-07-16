#!/usr/bin/env bash
# Uploads the generate (phase A) artifact to S3 after the stage finishes. Runs
# regardless of exit code so logs always ship; responses.parquet only exists on
# success, which the watchdog uses as the done/ok signal. Phase B (judge) runs
# locally against this responses.parquet and produces metrics.json off-box.
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
