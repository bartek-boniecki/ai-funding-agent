# main.py

from fastapi import FastAPI, Request
import uvicorn
import logging
import os
from dotenv import load_dotenv

# LangChain imports
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Document generation
from docx import Document

# 1️⃣ Load environment variables
load_dotenv()

# 2️⃣ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3️⃣ Initialize FastAPI
app = FastAPI()

# 4️⃣ Initialize SerpAPI and a chat-based LLM with high token limit
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2048
)

# 5️⃣ Query-refinement chains
# a) Scale & urgency of the problem
template_a = PromptTemplate(
    input_variables=["problem_text"],
    template=(
        "You are a search query generator. Given a user-provided problem description, "
        "produce a search query of up to 10 words to reliably find to find statistics illustrating scale and urgency.\n\n"
        "Problem:\n{problem_text}\n\nSearch Query:"
    )
)
chain_a = LLMChain(llm=llm, prompt=template_a)

# b) Market size & trends
template_b = PromptTemplate(
    input_variables=["solution_text"],
    template=(
        "You are a search query generator. Given a product description, "
        "produce a search query of up to 10 words to reliably find market size and key trends.\n\n"
        "Product:\n{solution_text}\n\nSearch Query:"
    )
)
chain_b = LLMChain(llm=llm, prompt=template_b)

# c) Competitors & novelty
template_c = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "You are a search query generator. Given unique product features, "
        "produce a search query of up to 10 words to reliably find similar offerings and assess novelty.\n\n"
        "Features:\n{features_text}\n\nSearch Query:"
    )
)
chain_c = LLMChain(llm=llm, prompt=template_c)

# d) Competitor revenue streams
template_d = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "You are a search query generator. Given unique product features, "
        "produce a search query of up to 10 words to reliably find competitor revenue streams.\n\n"
        "Features:\n{features_text}\n\nSearch Query:"
    )
)
chain_d = LLMChain(llm=llm, prompt=template_d)

# 6️⃣ Analysis chains
# a) "The problem/market opportunity"
analysis_template_a = PromptTemplate(
    input_variables=["problem_text","search_results"],
    template=(
        "You are an expert grant writer. Draft a 1500–2000 character analysis titled 'The problem/market opportunity'.\n\n"
        "User problem:\n{problem_text}\n\nSearch snippets:\n{search_results}\n\n"
        "Include concrete statistics and urgency language. Do NOT truncate."
    )
)
analysis_chain_a = LLMChain(llm=llm, prompt=analysis_template_a)

# b) "The innovation: Solution/Product or Services (USP)"
analysis_template_b = PromptTemplate(
    input_variables=["solution_text","search_results","trl","features_text"],
    template=(
        "Draft a 1500–2000 character analysis titled 'The innovation: Solution/Product or Services (USP)'.\n\n"
        "1. Why the product is better than existing solutions (use search results).\n"
        "2. Current TRL: {trl} and any validation/certification details provided.\n"
        "3. Why now is the right time to bring the product to market.\n\nDo NOT truncate."
    )
)
analysis_chain_b = LLMChain(llm=llm, prompt=analysis_template_b)

# c) "Market and Competition analysis"
analysis_template_c = PromptTemplate(
    input_variables=["solution_text","search_results","revenue_model","features_text"],
    template=(
        "Draft a 1500–2000 character analysis titled 'Market and Competition analysis'.\n\n"
        "1. Market size and key trends (from search results).\n"
        "2. Product potential to transform or create a market.\n"
        "3. Business model and envisioned revenue streams: {revenue_model}.\n"
        "4. Why features {features_text} will convince clients to buy.\n"
        "5. Advantages and disadvantages and likelihood of success.\n\nDo NOT truncate."
    )
)
analysis_chain_c = LLMChain(llm=llm, prompt=analysis_template_c)

# d) "Broad impacts"
analysis_template_d = PromptTemplate(
    input_variables=["solution_text","search_results"],
    template=(
        "Draft a 1500–2000 character analysis titled 'Broad impacts'.\n\n"
        "Discuss the product’s potential societal, environmental, or climate impacts and estimate job creation. Use search results.\n\nDo NOT truncate."
    )
)
analysis_chain_d = LLMChain(llm=llm, prompt=analysis_template_d)

# e) "Funding rationale and MVP"
analysis_template_e = PromptTemplate(
    input_variables=["solution_text","revenue_model","trl"],
    template=(
        "Draft a 1500–2000 character analysis titled 'Funding rationale and MVP'.\n\n"
        "Explain why raising funding is nearly impossible at TRL {trl}, why building an MVP is crucial, and include any funding history if provided. Use revenue model {revenue_model}.\n\nDo NOT truncate."
    )
)
analysis_chain_e = LLMChain(llm=llm, prompt=analysis_template_e)

@app.post("/webhook")
async def receive_form(request: Request):
    payload = await request.json()
    questions = payload.get("submission", {}).get("questions", [])

    # Extract inputs safely by index
    solution       = questions[0]["value"] if len(questions) > 0 else ""
    problem        = questions[1]["value"] if len(questions) > 1 else ""
    unique_feature = questions[2]["value"] if len(questions) > 2 else ""
    current_trl    = questions[3]["value"] if len(questions) > 3 else ""
    revenue_model  = questions[4]["value"] if len(questions) > 4 else ""

    logger.info("Inputs: Solution=%s, Problem=%s, Features=%s, TRL=%s, Revenue=%s", solution, problem, unique_feature, current_trl, revenue_model)

    # Generate and refine queries
    query_a = chain_a.run(problem_text=problem)
    results_a = serp.run(query_a)
    snippets_a = "\n".join(results_a[:5]) if isinstance(results_a, list) else str(results_a)

    query_b = chain_b.run(solution_text=solution)
    results_b = serp.run(query_b)
    snippets_b = "\n".join(results_b[:5]) if isinstance(results_b, list) else str(results_b)

    query_c = chain_c.run(features_text=unique_feature)
    results_c = serp.run(query_c)
    snippets_c = "\n".join(results_c[:5]) if isinstance(results_c, list) else str(results_c)

    query_d = chain_d.run(features_text=unique_feature)
    results_d = serp.run(query_d)
    snippets_d = "\n".join(results_d[:5]) if isinstance(results_d, list) else str(results_d)

    # Generate analyses (a–e)
    analysis_a = analysis_chain_a.run(problem_text=problem, search_results=snippets_a)
    analysis_b = analysis_chain_b.run(solution_text=solution, search_results=snippets_b, trl=current_trl, features_text=unique_feature)
    analysis_c = analysis_chain_c.run(solution_text=solution, search_results=snippets_b, revenue_model=revenue_model, features_text=unique_feature)
    analysis_d = analysis_chain_d.run(solution_text=solution, search_results=snippets_d)
    analysis_e = analysis_chain_e.run(solution_text=solution, revenue_model=revenue_model, trl=current_trl)

    # 7️⃣ Compile analyses into a Word document
    doc = Document()
    sections = [
        ("The problem/market opportunity", analysis_a),
        ("The innovation: Solution/Product or Services (USP)", analysis_b),
        ("Market and Competition analysis", analysis_c),
        ("Broad impacts", analysis_d),
        ("Funding rationale and MVP", analysis_e)
    ]
    for title, text in sections:
        doc.add_heading(title, level=1)
        doc.add_paragraph(text)

    output_path = "/mnt/data/analyses.docx"
    doc.save(output_path)
    logger.info("Document saved to %s", output_path)

    return {"status": "ok", "document_path": output_path}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
