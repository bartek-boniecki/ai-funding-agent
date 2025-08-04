# main.py

from fastapi import FastAPI, Request
import uvicorn
import logging
import os
from dotenv import load_dotenv

# LangChain imports for web search and LLM-based query refinement
from langchain.utilities import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

# 1️⃣ Load environment variables from .env
load_dotenv()

# 2️⃣ Configure logging so we can see what’s happening in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3️⃣ Initialize FastAPI app
app = FastAPI()

# 4️⃣ Initialize the SerpAPI search wrapper
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))

# 5️⃣ Initialize OpenAI LLM for refining raw user text into concise search queries
llm = OpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 6️⃣ Define templates to turn long, varied user input into focused search queries

# a) Scale & urgency of the problem
template_a = PromptTemplate(
    input_variables=["problem_text"],
    template=(
        "You are a search query generator. Given a user-provided problem description, "
        "produce a keyword-focused search query of no more than 10 words to reliably find publicly available "
        "statistics illustrating the scale and urgency of the problem.\n\n"
        "Problem Description:\n{problem_text}\n\n"
        "Search Query:"
    )
)
chain_a = LLMChain(llm=llm, prompt=template_a)

# b) Market size & trends
template_b = PromptTemplate(
    input_variables=["solution_text"],
    template=(
        "You are a search query generator. Given a description of a technology or product, "
        "produce a keyword-focused search query of no more than 10 words to reliably find the market that it addresses, including market size "
        "and key trends.\n\n"
        "Product/Solution Description:\n{solution_text}\n\n"
        "Search Query:"
    )
)
chain_b = LLMChain(llm=llm, prompt=template_b)

# c) Competitors & novelty
template_c = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "You are a search query generator. Given a list of unique features for a product, "
        "produce a keyword-focused search query of no more than 10 words to reliably find companies offering similar features and to assess the degree "
        "of novelty compared to existing solutions.\n\n"
        "Unique Features:\n{features_text}\n\n"
        "Search Query:"
    )
)
chain_c = LLMChain(llm=llm, prompt=template_c)

# d) Competitor revenue streams
template_d = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "You are a search query generator. Given a list of unique features for a product, "
        "produce a keyword-focused search query of no more than 10 words to reliably find what revenue streams competitors generate from products "
        "with similar features.\n\n"
        "Unique Features:\n{features_text}\n\n"
        "Search Query:"
    )
)
chain_d = LLMChain(llm=llm, prompt=template_d)

@app.post("/webhook")
async def receive_form(request: Request):
    """
    Receives the form submission JSON, extracts the five user answers,
    uses the LLM to refine each into a focused search query, then
    runs SerpAPI searches and logs the results.
    """
    payload = await request.json()

    # 7️⃣ Extract the list of questions from the payload
    questions = payload.get("submission", {}).get("questions", [])

    # 8️⃣ Safely extract each answer by its position in the form
    solution       = questions[0]["value"] if len(questions) > 0 else ""
    problem        = questions[1]["value"] if len(questions) > 1 else ""
    unique_feature = questions[2]["value"] if len(questions) > 2 else ""
    current_trl    = questions[3]["value"] if len(questions) > 3 else ""
    revenue_model  = questions[4]["value"] if len(questions) > 4 else ""

    # 9️⃣ Log the raw user inputs
    logger.info("Solution      : %s", solution)
    logger.info("Problem       : %s", problem)
    logger.info("Unique Feature: %s", unique_feature)
    logger.info("Current TRL   : %s", current_trl)
    logger.info("Revenue Model : %s", revenue_model)

    # 🔍 a) Scale & urgency of the problem
    if problem:
        # Use the LLM chain to turn the long description into a search query
        query_a = chain_a.run(problem_text=problem)
        logger.info("Refined query a): %s", query_a)
        results_a = serp.run(query_a)
        logger.info("Results a): %s", results_a)

    # 🔍 b) Market addressed, size & key trends
    if solution:
        query_b = chain_b.run(solution_text=solution)
        logger.info("Refined query b): %s", query_b)
        results_b = serp.run(query_b)
        logger.info("Results b): %s", results_b)

    # 🔍 c) Competitors & degree of novelty
    if unique_feature:
        query_c = chain_c.run(features_text=unique_feature)
        logger.info("Refined query c): %s", query_c)
        results_c = serp.run(query_c)
        logger.info("Results c): %s", results_c)

    # 🔍 d) Competitor revenue streams
    if unique_feature:
        query_d = chain_d.run(features_text=unique_feature)
        logger.info("Refined query d): %s", query_d)
        results_d = serp.run(query_d)
        logger.info("Results d): %s", results_d)

    # 1️⃣0️⃣ Return a simple acknowledgment
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
