# main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import logging
import os
from dotenv import load_dotenv

# Updated LangChain imports
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Word document generation
from docx import Document

# SendGrid for email
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition
)
import base64

# 1️⃣ Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_SENDER  = os.getenv("SENDGRID_SENDER")
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

# 4️⃣ Initialize SerpAPI & the Chat LLM
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2048
)

# Helper to build LLMChains
def make_chain(template: PromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=template)

# 5️⃣ Query-refinement chains
chain_a = make_chain(PromptTemplate(
    input_variables=["text"],
    template="""
Generate a concise search query to find publicly available statistics illustrating
the scale and urgency of this problem.

Problem: {text}

Search Query:
"""
))
chain_b = make_chain(PromptTemplate(
    input_variables=["text"],
    template="""
Generate a concise search query to find the market addressed by this product,
including market size and key trends.

Product: {text}

Search Query:
"""
))
chain_c = make_chain(PromptTemplate(
    input_variables=["text"],
    template="""
Generate a concise search query to identify companies offering similar features
and assess the degree of novelty.

Features: {text}

Search Query:
"""
))
chain_d = make_chain(PromptTemplate(
    input_variables=["text"],
    template="""
Generate a concise search query to find revenue streams competitors derive
from products with similar features.

Features: {text}

Search Query:
"""
))

# 6️⃣ Analysis chains
analysis_a = make_chain(PromptTemplate(
    input_variables=["problem", "stats"],
    template="""
Draft a 1500–2000 character analysis titled "The problem/market opportunity".

Problem description:
{problem}

Statistics/snippets:
{stats}

Substantiate with concrete data illustrating scale and urgency. Do NOT truncate mid-sentence.
"""
))
analysis_b = make_chain(PromptTemplate(
    input_variables=["solution", "stats", "trl", "features"],
    template="""
Draft a 1500–2000 character analysis titled "The innovation: Solution/Product or Services (USP)".

1. Why this product is better than existing solutions (use: {stats}).
2. Current Technology Readiness Level: {trl}, including any validation/certification details.
3. Why now is the right time to bring this product to market.

Unique features: {features}
Do NOT truncate mid-sentence.
"""
))
analysis_c = make_chain(PromptTemplate(
    input_variables=["solution", "stats", "revenue", "features"],
    template="""
Draft a 1500–2000 character analysis titled "Market and Competition analysis".

1. Market size & key trends (use: {stats}).
2. Product potential to transform/create market.
3. Business model & revenue streams: {revenue}.
4. Why features {features} will drive adoption.
5. Advantages/disadvantages & success factors.

Do NOT truncate mid-sentence.
"""
))
analysis_d = make_chain(PromptTemplate(
    input_variables=["solution", "stats"],
    template="""
Draft a 1500–2000 character analysis titled "Broad impacts".

Discuss potential societal, environmental, or climate impacts and estimate job creation
(use: {stats}). Do NOT truncate mid-sentence.
"""
))
analysis_e = make_chain(PromptTemplate(
    input_variables=["solution", "revenue", "trl"],
    template="""
Draft a 1500–2000 character analysis titled "Funding rationale and MVP".

Explain why raising funding is nearly impossible at TRL {trl}, why an MVP is crucial,
and include any funding history if provided.

Revenue model: {revenue}
Do NOT truncate mid-sentence.
"""
))

@app.post("/webhook")
async def receive_form(request: Request):
    payload = await request.json()
    qs = payload.get("submission", {}).get("questions", [])
    if len(qs) < 6:
        raise HTTPException(status_code=400, detail="Form must include 5 answers plus an email field")

    # 7️⃣ Extract inputs
    solution = qs[0].get("value", "").strip()
    problem  = qs[1].get("value", "").strip()
    features = qs[2].get("value", "").strip()
    trl      = qs[3].get("value", "").strip()
    revenue  = qs[4].get("value", "").strip()
    email    = qs[5].get("value", "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email address is required")

    logger.info(f"Processing analyses for {email}")

    # Helper to get top-5 snippets, swallowing SerpAPI errors
    def get_snippets(chain: LLMChain, text: str) -> str:
        try:
            query = chain.invoke(**{chain.prompt.input_variables[0]: text})
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return ""
        try:
            results = serp.run(query)
            return "\n".join(results[:5]) if isinstance(results, list) else str(results)
        except Exception as e:
            logger.warning(f"SerpAPI search failed for '{query}': {e}")
            return ""

    # 8️⃣ Gather snippets
    stats_a = get_snippets(chain_a, problem)
    stats_b = get_snippets(chain_b, solution)
    stats_c = get_snippets(chain_c, features)
    stats_d = get_snippets(chain_d, features)

    # 9️⃣ Generate analyses
    a_txt = analysis_a.invoke(problem=problem, stats=stats_a)
    b_txt = analysis_b.invoke(solution=solution, stats=stats_b, trl=trl, features=features)
    c_txt = analysis_c.invoke(solution=solution, stats=stats_b, revenue=revenue, features=features)
    d_txt = analysis_d.invoke(solution=solution, stats=stats_d)
    e_txt = analysis_e.invoke(solution=solution, revenue=revenue, trl=trl)

    # 10️⃣ Compile into a Word document
    doc = Document()
    for title, content in [
        ("The problem/market opportunity", a_txt),
        ("The innovation: Solution/Product or Services (USP)", b_txt),
        ("Market and Competition analysis", c_txt),
        ("Broad impacts", d_txt),
        ("Funding rationale and MVP", e_txt),
    ]:
        doc.add_heading(title, level=1)
        doc.add_paragraph(content)
    filename = "analyses.docx"
    doc.save(filename)
    logger.info(f"Saved document: {filename}")

    # 11️⃣ Email as attachment
    with open(filename, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode()
    message = Mail(
        from_email=SENDGRID_SENDER,
        to_emails=email,
        subject="Your AI-Generated Analyses",
        html_content="<p>Please find attached your analyses document.</p>"
    )
    attachment = Attachment(
        FileContent(data_b64),
        FileName(filename),
        FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        Disposition("attachment")
    )
    message.attachment = attachment
    resp = SendGridAPIClient(SENDGRID_API_KEY).send(message)
    logger.info(f"Email sent; status code: {resp.status_code}")

    # 12️⃣ Return success
    download_url = request.url._url.rstrip(request.url.path) + "/download"
    return {"status": "ok", "email_status": resp.status_code, "download_url": download_url}

@app.get("/download")
async def download():
    path = os.path.join(os.getcwd(), "analyses.docx")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(
        path=path,
        filename="analyses.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
