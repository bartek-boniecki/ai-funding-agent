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
    payload = await request.json()

    # 1️⃣ Grab the list of question objects
    questions = payload.get("submission", {}).get("questions", [])

    # 2️⃣ Build a simple dict: { question_name: answer_value }
    answers = { q["name"]: q["value"] for q in questions }

    # 3️⃣ Log each answer clearly
    logger.info("Solution      : %s", answers.get("Describe your solution (<500 characters)What is the underlying tech, what it does, why it does what it does"))
    logger.info("Problem       : %s", answers.get("What pain/problem does it solve? (<500 characters)Why is it both urgent and important"))
    logger.info("Unique Feature: %s", answers.get("What unique features and value does it offer? (<500 characters)EIC is for tech innovations that add significant value"))
    logger.info("Current TRL   : %s", answers.get("What technology readiness level has already been completedFor AI solutions, see TRL definitions here: https://ai-watch.ec.europa.eu/publications/ai-watch-revisiting-technology-readiness-levels-relevant-artificial-intelligence-technologies_en"))
    logger.info("Revenue Model : %s", answers.get("What is your business and revenue model (<500 characters)How will this solution generate income"))

    # 4️⃣ (For now) just return OK
    return {"status": "ok"}

# 5. If you run this file directly, start the Uvicorn server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
