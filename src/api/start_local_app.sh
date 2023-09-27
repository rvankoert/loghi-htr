export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

export LOGHI_MODEL_PATH="/home/tim/Downloads/new_model/"
export LOGHI_CHARLIST_PATH="/home/tim/Downloads/new_model/charlist.txt"
export LOGHI_MODEL_CHANNELS=4
export LOGHI_BATCH_SIZE=300
export LOGHI_OUTPUT_PATH="/home/tim/Documents/development/loghi-htr/output/"
export LOGHI_MAX_QUEUE_SIZE=50000

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

python3 gunicorn_app.py
