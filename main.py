# main.py

from fastapi import FastAPI, Request
import uvicorn
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/webhook")
async def receive_form(request: Request):
    """
    Receives JSON payload from Fillout form submissions,
    extracts the five answers by index, and logs them.
    """
    payload = await request.json()

    # Extract the list of question objects
    questions = payload.get("submission", {}).get("questions", [])

    # Safely get each answer by position
    solution       = questions[0]["value"] if len(questions) > 0 else None
    problem        = questions[1]["value"] if len(questions) > 1 else None
    unique_feature = questions[2]["value"] if len(questions) > 2 else None
    current_trl    = questions[3]["value"] if len(questions) > 3 else None
    revenue_model  = questions[4]["value"] if len(questions) > 4 else None

    # Log each one clearly
    logger.info("Solution      : %s", solution)
    logger.info("Problem       : %s", problem)
    logger.info("Unique Feature: %s", unique_feature)
    logger.info("Current TRL   : %s", current_trl)
    logger.info("Revenue Model : %s", revenue_model)

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
