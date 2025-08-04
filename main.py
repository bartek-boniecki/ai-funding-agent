# main.py

from fastapi import FastAPI, Request
import uvicorn
import logging
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.utilities import SerpAPIWrapper
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

# 1️⃣ Load environment variables
load_dotenv()

# 2️⃣ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3️⃣ Initialize FastAPI
app = FastAPI()

# 4️⃣ Initialize SerpAPI and OpenAI LLM
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 5️⃣ Query‐refinement chains for the four research questions

# a) Scale & urgency of the problem
template_a = PromptTemplate(
    input_variables=["problem_text"],
    template=(
        "You are a search query generator. Given a user‐provided problem description, "
        "produce a keyword-focused search query of no more than 10 words to reliably find publicly available "
        "statistics illustrating the scale and urgency of the problem.\n\n"
        "Problem Description:\n{problem_text}\n\n"
        "Search Query:"
    )
)
chain_a = LLMChain(llm=llm, prompt=template_a)

# b) Market addressed, size & key trends
template_b = PromptTemplate(
    input_variables=["solution_text"],
    template=(
        "You are a search query generator. Given a description of a technology or product, "
        "produce a keyword-focused search query of no more than 10 words to reliably find the market it addresses, including market size "
        "and key trends.\n\n"
        "Product/Solution Description:\n{solution_text}\n\n"
        "Search Query:"
    )
)
chain_b = LLMChain(llm=llm, prompt=template_b)

# c) Competitors & degree of novelty
template_c = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "You are a search query generator. Given a list of unique features for a product, "
        "produce a search query of no more than 10 words to reliably find companies offering similar features and assess the degree "
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
        "produce a search query of no more than 10 words to reliably find what revenue streams competitors generate from products "
        "with similar features.\n\n"
        "Unique Features:\n{features_text}\n\n"
        "Search Query:"
    )
)
chain_d = LLMChain(llm=llm, prompt=template_d)

# 6️⃣ Analysis chain for "The problem/market opportunity"
analysis_template_a = PromptTemplate(
    input_variables=["problem_text", "search_results"],
    template=(
        "You are an expert grant writer. Draft a 1500–2000 character analysis titled "
        "'The problem/market opportunity'.\n\n"
        "Use the user’s problem description:\n"
        "{problem_text}\n\n"
        "And these search results (statistics/snippets):\n"
        "{search_results}\n\n"
        "Substantiate the analysis with concrete statistics illustrating the scale and urgency of the problem."
    )
)
analysis_chain_a = LLMChain(llm=llm, prompt=analysis_template_a)

@app.post("/webhook")
async def receive_form(request: Request):
    # 7️⃣ Receive and parse payload
    payload = await request.json()
    questions = payload.get("submission", {}).get("questions", [])

    # 8️⃣ Extract answers by index (safe defaults)
    solution       = questions[0]["value"] if len(questions) > 0 else ""
    problem        = questions[1]["value"] if len(questions) > 1 else ""
    unique_feature = questions[2]["value"] if len(questions) > 2 else ""
    current_trl    = questions[3]["value"] if len(questions) > 3 else ""
    revenue_model  = questions[4]["value"] if len(questions) > 4 else ""

    # 9️⃣ Log raw inputs
    logger.info("Solution      : %s", solution)
    logger.info("Problem       : %s", problem)
    logger.info("Unique Feature: %s", unique_feature)
    logger.info("Current TRL   : %s", current_trl)
    logger.info("Revenue Model : %s", revenue_model)

    # 🔍 a) Scale & urgency research
    if problem:
        # 1. Refine user text into a search query
        query_a = chain_a.run(problem_text=problem)
        logger.info("Refined query a): %s", query_a)
        # 2. Perform web search
        results_a = serp.run(query_a)
        logger.info("Results a): %s", results_a)
        # 3. Draft the analysis using search results
        analysis_a = analysis_chain_a.run(
            problem_text=problem,
            search_results=results_a
        )
        logger.info("Analysis a): %s", analysis_a)

    # (Later: research b), c), d) and analyses b), c), etc.)

    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
