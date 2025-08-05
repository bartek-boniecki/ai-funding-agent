# main.py

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
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

# Email sending via SendGrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

# 1️⃣ Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_SENDER = os.getenv("SENDGRID_SENDER")
if not SENDGRID_API_KEY or not SENDGRID_SENDER:
    raise RuntimeError("SENDGRID_API_KEY and SENDGRID_SENDER must be set in .env")

# 2️⃣ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 3️⃣ Initialize FastAPI
app = FastAPI()

# 4️⃣ Initialize tools
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2048
)

# Utility to create chains
def make_chain(template: PromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=template)

# 5️⃣ PromptTemplates & Chains
chain_a = make_chain(PromptTemplate(
    input_variables=["problem_text"],
    template="""
Generate a concise search query to find publicly available statistics illustrating the scale and urgency of this problem.

Problem: {problem_text}

Search Query:
"""
))
chain_b = make_chain(PromptTemplate(
    input_variables=["solution_text"],
    template="""
Generate a concise search query to find the market addressed by this product, including market size and key trends.

Product: {solution_text}

Search Query:
"""
))
chain_c = make_chain(PromptTemplate(
    input_variables=["features_text"],
    template="""
Generate a concise search query to identify companies offering similar features and assess the degree of novelty.

Features: {features_text}

Search Query:
"""
))
chain_d = make_chain(PromptTemplate(
    input_variables=["features_text"],
    template="""
Generate a concise search query to find revenue streams competitors derive from products with similar features.

Features: {features_text}

Search Query:
"""
))

analysis_a_chain = make_chain(PromptTemplate(
    input_variables=["problem_text","search_results"],
    template="""
Draft a 1500–2000 character analysis titled "The problem/market opportunity".

Use the user’s problem description:
{problem_text}

Incorporate these statistics/snippets:
{search_results}

Substantiate with concrete data illustrating scale and urgency. Do NOT truncate mid-sentence.
"""
))
analysis_b_chain = make_chain(PromptTemplate(
    input_variables=["solution_text","search_results","trl","features_text"],
    template="""
Draft a 1500–2000 character analysis titled "The innovation: Solution/Product or Services (USP)".

1. Why this product is better than existing solutions (use: {search_results}).
2. Current TRL: {trl}, include validation/certification details.
3. Why now is the right time to bring this product to market.

Unique features: {features_text}
Do NOT truncate mid-sentence.
"""
))
analysis_c_chain = make_chain(PromptTemplate(
    input_variables=["solution_text","search_results","revenue_model","features_text"],
    template="""
Draft a 1500–2000 character analysis titled "Market and Competition analysis".

1. Market size & key trends (use: {search_results}).
2. Product potential to transform/create market.
3. Business model & revenue streams: {revenue_model}.
4. Why features {features_text} will drive adoption.
5. Advantages/disadvantages & success factors.
Do NOT truncate mid-sentence.
"""
))
analysis_d_chain = make_chain(PromptTemplate(
    input_variables=["solution_text","search_results"],
    template="""
Draft a 1500–2000 character analysis titled "Broad impacts".

Discuss societal, environmental, or climate impacts and estimate job creation. Use:
{search_results}
Do NOT truncate mid-sentence.
"""
))
analysis_e_chain = make_chain(PromptTemplate(
    input_variables=["solution_text","revenue_model","trl"],
    template="""
Draft a 1500–2000 character analysis titled "Funding rationale and MVP".

Explain why raising funding is nearly impossible at TRL {trl}, why an MVP is crucial, and include any funding history if provided.
Use: {revenue_model}
Do NOT truncate mid-sentence.
"""
))

@app.post("/webhook")
async def receive_form(request: Request):
    payload = await request.json()
    qs = payload.get("submission", {}).get("questions", [])
    # Map question names to values
    answers = {q.get("name"): q.get("value") for q in qs}

    # Required fields by label
    try:
        solution = answers["Describe your solution"]
        problem = answers["What problem does it solve?"]
        features = answers["Unique features of your solution"]
        trl = answers["Current Technology Readiness Level (TRL)"]
        revenue_model = answers["Envisioned revenue streams"]
        user_email = answers["Your email address"]
        consent = answers.get("I consent to data processing in compliance with GDPR")
    except KeyError as e:
        return JSONResponse(status_code=400, content={"detail": f"Missing field: {e}"})

    # Validate consent
    if consent not in [True, "true", "True", "on", "yes", "Yes"]:
        return JSONResponse(status_code=400, content={"detail": "Consent is required to proceed."})

    logger.info(f"Processing for email: {user_email}")

    # Generate snippets
    def snippet(chain, text):
        query = chain.run(**{chain.prompt.input_variables[0]: text})
        results = serp.run(query)
        return "\n".join(results[:5]) if isinstance(results, list) else str(results)

    s_a, s_b, s_c, s_d = snippet(chain_a, problem), snippet(chain_b, solution), snippet(chain_c, features), snippet(chain_d, features)

    # Generate analyses
    a = analysis_a_chain.run(problem_text=problem, search_results=s_a)
    b = analysis_b_chain.run(solution_text=solution, search_results=s_b, trl=trl, features_text=features)
    c = analysis_c_chain.run(solution_text=solution, search_results=s_b, revenue_model=revenue_model, features_text=features)
    d = analysis_d_chain.run(solution_text=solution, search_results=s_d)
    e = analysis_e_chain.run(solution_text=solution, revenue_model=revenue_model, trl=trl)

    # Compile document
    doc = Document()
    for title, txt in [
        ("The problem/market opportunity", a),
        ("The innovation: Solution/Product or Services (USP)", b),
        ("Market and Competition analysis", c),
        ("Broad impacts", d),
        ("Funding rationale and MVP", e)
    ]:
        doc.add_heading(title, level=1)
        doc.add_paragraph(txt)
    filename = "analyses.docx"
    doc.save(filename)
    logger.info(f"Saved document {filename}")

    # Email document
    with open(filename, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    message = Mail(
        from_email=SENDGRID_SENDER,
        to_emails=user_email,
        subject="Your AI-Generated Analyses",
        html_content="<p>Please find the attached analyses document.</p>"
    )
    attach = Attachment(
        FileContent(encoded),
        FileName(filename),
        FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        Disposition("attachment")
    )
    message.attachment = attach
    resp = SendGridAPIClient(SENDGRID_API_KEY).send(message)
    logger.info(f"Email sent status {resp.status_code}")

    download_url = request.url._url.rstrip(request.url.path) + "/download"
    return {"status": "ok", "email_status": resp.status_code, "download_url": download_url}

@app.get("/download")
async def download_doc():
    path = os.path.join(os.getcwd(), "analyses.docx")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"detail": "Document not found"})
    return FileResponse(path=path, filename="analyses.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
