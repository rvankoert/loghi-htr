export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

export LOGHI_MODEL_PATH="/home/luke/ai_development/loghi-dev/loghi/loghi-htr/src/output/best_val/"
export LOGHI_CHARLIST_PATH="/home/luke/ai_development/loghi-dev/loghi/loghi-htr/src/output/best_val/charlist.txt"
export LOGHI_MODEL_CHANNELS=3
export LOGHI_BATCH_SIZE=64
export LOGHI_OUTPUT_PATH="/home/luke/ai_development/loghi-dev/loghi/loghi-htr/output/"
export LOGHI_MAX_QUEUE_SIZE=50000

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

python3 gunicorn_app.py
