#!/bin/bash
export UVICORN_HOST="127.0.0.1"
export UVICORN_PORT="8000"

export LOGHI_BASE_MODEL_DIR="/path/to/loghi-models/"
export LOGHI_MODEL_NAME="SOME_MODEL_NAME"

export LOGHI_BATCH_SIZE=200
export LOGHI_MAX_QUEUE_SIZE=50000
export LOGHI_PATIENCE=0.5

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

python3 -m src.api.experimental.app hypercorn --h2c
