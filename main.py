# main.py

from fastapi import FastAPI, Request
import uvicorn
import logging
from dotenv import load_dotenv
import os

# 1. Load environment variables from a .env file (we'll create this next)
load_dotenv()

# 2. Create the FastAPI app instance
app = FastAPI()

# 3. Configure basic logging to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 4. Define the webhook endpoint that Fillout will call
@app.post("/webhook")
async def receive_form(request: Request):
    """
    Receives JSON payload from Fillout form submissions.
    """
    payload = await request.json()              # Read the JSON data
    logger.info("Received form data: %s", payload)  # Print it to the console/logs
    # Here is where we'll later hand this data off to our LangChain agent.
    return {"status": "ok"}                     # Simple JSON response

# 5. If you run this file directly, start the Uvicorn server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
