#!/bin/bash 
export LOGHI_BASE_MODEL_DIR="/path/to/loghi-models/"
export LOGHI_MODEL_NAME="SOME_MODEL_NAME"
export LOGHI_OUTPUT_PATH="/path/to/output/dir/"

export LOGHI_BATCH_SIZE=200
export LOGHI_MAX_QUEUE_SIZE=50000
export LOGHI_PATIENCE=0.5

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

uvicorn app:app --host 0.0.0.0 --port 5000
