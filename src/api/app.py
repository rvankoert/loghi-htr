# This file serves as the main entry point for the STABLE API
# when running with a production server like Uvicorn.
#
# To run the stable API:
# uvicorn src.api.app:app --host 127.0.0.1 --port 5000
#
# To run the experimental API, use its own entry point:
# python -m src.api.experimental.app

# > Local imports
from .stable.app import create_app

# Create the FastAPI application instance for the stable API
app = create_app()
