#!/bin/bash

set -ex

gunicorn \
    app:app \
    --timeout 600 \
    --workers "${NUM_WORKERS:=1}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --threads 1 \
    --access-logfile -