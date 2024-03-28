export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_ACCESSLOG='-'

export LOGHI_BATCH_SIZE=300
export LOGHI_MODEL_PATH="/home/tim/Downloads/new_model"
export LOGHI_OUTPUT_PATH="/home/tim/Documents/development/loghi-htr/output/"
export LOGHI_MAX_QUEUE_SIZE=50000
export LOGHI_PATIENCE=0.5

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

gunicorn -w 1 -b $GUNICORN_RUN_HOST \
    --access-logfile $GUNICORN_ACCESSLOG 'app:create_app()'
