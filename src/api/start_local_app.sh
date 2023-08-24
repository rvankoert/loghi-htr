export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

export LOGHI_MODEL_PATH="/home/tim/Documents/loghi-models/generic-2023-02-15/"
export LOGHI_CHARLIST_PATH="/home/tim/Documents/loghi-models/generic-2023-02-15/charlist.txt"
export LOGHI_MODEL_CHANNELS=1
export LOGHI_BATCH_SIZE=100
export LOGHI_OUTPUT_PATH="/home/tim/Documents/development/loghi-htr/output/"

export LOGGING_LEVEL="INFO"

python3 gunicorn_app.py
