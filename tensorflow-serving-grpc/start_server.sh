#!/bin/bash
/usr/bin/tensorflow_model_server \
    --model_config_file=/mnt/models/models.config \
    --port=9000 \
    --rest_api_port=8080 \
    --grpc_max_threads=8 \
    --enable_batching=true \
    --enable_model_warmup=true
