export GUNICORN_RUN_HOST='0.0.0.0:5000'
export GUNICORN_WORKERS=1
export GUNICORN_THREADS=1
export GUNICORN_ACCESSLOG='-'

export LOGHI_BATCH_SIZE=300
export LOGHI_MODEL_PATH="/home/tim/Downloads/new_model"
export LOGHI_OUTPUT_PATH="/home/tim/Documents/development/loghi-htr/output/"
export LOGHI_MAX_QUEUE_SIZE=50000

export LOGGING_LEVEL="INFO"
export LOGHI_GPUS="0"

export SECURITY_ENABLED="True"
export API_KEY_USER_JSON_STRING='{"1234": "test user"}'

python3 gunicorn_app.py
