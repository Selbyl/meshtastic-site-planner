"""
Signal Coverage Prediction API

Provides endpoints to predict radio signal coverage
using the ITM (Irregular Terrain Model) via SPLAT! (https://github.com/jmcmellen/splat).

Endpoints:
    - /predict: Accepts a signal coverage prediction request and starts a background task.
    - /status/{task_id}: Retrieves the status of a given prediction task.
    - /result/{task_id}: Retrieves the result (GeoTIFF file) of a given prediction task.
"""

import io
import logging
import os
from uuid import uuid4

import redis
from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.models.CoveragePredictionRequest import CoveragePredictionRequest
from app.services.splat import Splat
from app.services.terrain_engine import TerrainEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis client for binary data
redis_client = redis.StrictRedis(host="redis", port=6379, decode_responses=False)

propagation_engine = os.getenv("PROPAGATION_ENGINE", "splat").lower()

# Initialize propagation service
if propagation_engine == "terrain":
    splat_service = TerrainEngine(
        dem_1m_path=os.getenv("DEM_1M_PATH"),
        dem_10m_path=os.getenv("DEM_10M_PATH"),
        max_pixels=int(os.getenv("DEM_MAX_PIXELS", "2048")),
        azimuth_step=int(os.getenv("AZIMUTH_STEP", "1")),
        blocked_loss_db=float(os.getenv("BLOCKED_LOSS_DB", "20")),
    )
else:
    splat_service = Splat(splat_path="/app/splat")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*/", "http://site.meshtastic.org"],  # Replace '*' with specific origins like ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def run_splat(task_id: str, request: CoveragePredictionRequest):
    """
    Execute the SPLAT! coverage prediction and store the resulting GeoTIFF data in Redis.

    Args:
        task_id (str): UUID identifier for the task.
        request (CoveragePredictionRequest): The parameters for the SPLAT! prediction.

    Workflow:
        - Runs the SPLAT! coverage prediction.
        - Stores the resulting GeoTIFF data and the task status ("completed") in Redis.
        - On failure, stores the task status as "failed" and logs the error in Redis.

    Raises:
        Exception: If SPLAT! fails during execution.
    """
    try:
        logger.info(f"Starting SPLAT! coverage prediction for task {task_id}.")
        if hasattr(splat_service, "coverage_prediction_with_kmz"):
            geotiff_data, kmz_data = splat_service.coverage_prediction_with_kmz(request)
            redis_client.setex(f"{task_id}:kmz", 3600, kmz_data)
        else:
            geotiff_data = splat_service.coverage_prediction(request)

        # Log before storing in Redis
        logger.info(f"Storing result in Redis for task {task_id}")
        redis_client.setex(task_id, 3600, geotiff_data)
        redis_client.setex(f"{task_id}:status", 3600, "completed")
        logger.info(f"Task {task_id} marked as completed.")
    except Exception as e:
        logger.error(f"Error in SPLAT! task {task_id}: {e}")
        redis_client.setex(f"{task_id}:status", 3600, "failed")
        redis_client.setex(f"{task_id}:error", 3600, str(e))
        raise

@app.post("/predict")
async def predict(payload: CoveragePredictionRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Predict signal coverage using SPLAT!.
    Accepts a CoveragePredictionRequest and processes it in the background.

    - Generates a unique task ID.
    - Sets the initial task status to "processing" in Redis.
    - Adds the `run_splat` function to the background task queue.

    Args:
        payload (CoveragePredictionRequest): The parameters required for the SPLAT! coverage prediction.
        background_tasks (BackgroundTasks): FastAPI background tasks.

    Returns:
        JSONResponse: A response containing the unique task ID to track the prediction progress.
    """
    task_id = str(uuid4())
    redis_client.setex(f"{task_id}:status", 3600, "processing")
    background_tasks.add_task(run_splat, task_id, payload)
    return JSONResponse({"task_id": task_id})

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Retrieve the status of a given SPLAT! task.

    - Checks Redis for the task status.
    - Returns "processing", "completed", or "failed" based on the status.
    - Returns a 404 error if the task ID is not found.

    Args:
        task_id (str): The unique identifier for the task.

    Returns:
        JSONResponse: The task status or an error message if the task is not found.
    """
    status = redis_client.get(f"{task_id}:status")
    if not status:
        logger.warning(f"Task {task_id} not found in Redis.")
        return JSONResponse({"error": "Task not found"}, status_code=404)

    return JSONResponse({"task_id": task_id, "status": status.decode("utf-8")})

@app.get("/result/{task_id}")
async def get_result(task_id: str, format: str = Query("tif", pattern="^(tif|kmz)$")):
    """
    Retrieve SPLAT! task status or GeoTIFF/KMZ result.

    - Checks the task status in Redis.
    - If "completed," retrieves the GeoTIFF (default) or KMZ when format=kmz.
    - If "failed," returns the error message stored in Redis.
    - If "processing", indicate the same in the response.

    Args:
        task_id (str): The unique identifier for the task.

    Returns:
        JSONResponse: Task status if the task is still "processing" or "failed."
        StreamingResponse: A downloadable GeoTIFF file if the task is "completed."
    """
    status = redis_client.get(f"{task_id}:status")
    if not status:
        logger.warning(f"Task {task_id} not found in Redis.")
        return JSONResponse({"error": "Task not found"}, status_code=404)

    status = status.decode("utf-8")
    if status == "completed":
        if format == "kmz":
            kmz_data = redis_client.get(f"{task_id}:kmz")
            if not kmz_data:
                logger.error(f"No KMZ data found for completed task {task_id}.")
                return JSONResponse({"error": "No KMZ result found"}, status_code=500)
            kmz_file = io.BytesIO(kmz_data)
            return StreamingResponse(
                kmz_file,
                media_type="application/vnd.google-earth.kmz",
                headers={"Content-Disposition": f"attachment; filename={task_id}.kmz"},
            )

        geotiff_data = redis_client.get(task_id)
        if not geotiff_data:
            logger.error(f"No data found for completed task {task_id}.")
            return JSONResponse({"error": "No result found"}, status_code=500)

        geotiff_file = io.BytesIO(geotiff_data)
        return StreamingResponse(
            geotiff_file,
            media_type="image/tiff",
            headers={"Content-Disposition": f"attachment; filename={task_id}.tif"},
        )
    elif status == "failed":
        error = redis_client.get(f"{task_id}:error")
        return JSONResponse({"status": "failed", "error": error.decode("utf-8")})

    logger.info(f"Task {task_id} is still processing.")
    return JSONResponse({"status": "processing"})

app.mount("/", StaticFiles(directory="app/ui", html=True), name="ui")
