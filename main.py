# main.py

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import logging
import os
from dotenv import load_dotenv

# SerpAPI wrapper from community package
from langchain_community.utilities import SerpAPIWrapper
# ChatOpenAI from langchain-openai package
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from docx import Document
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

# Utility to create LLMChains
def make_chain(prompt: PromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt)

# 5️⃣ Define query-refinement chains
a_query = make_chain(PromptTemplate(
    input_variables=["text"],
    template="Generate a concise search query to find statistics illustrating the scale and urgency of this problem.\n\n{text}"))
b_query = make_chain(PromptTemplate(
    input_variables=["text"],
    template="Generate a search query to find market size and key trends for this product.\n\n{text}"))
c_query = make_chain(PromptTemplate(
    input_variables=["text"],
    template="Generate a search query to identify competitors with similar features {text} and assess novelty."))
d_query = make_chain(PromptTemplate(
    input_variables=["text"],
    template="Generate a search query to find competitor revenue streams for products with features {text}."))

# 6️⃣ Define analysis chains
a_chain = make_chain(PromptTemplate(
    input_variables=["problem","stats"],
    template="""
Draft a 1500–2000 character analysis titled "The problem/market opportunity".
Use problem: {problem} and stats: {stats}. Do NOT truncate.
"""))
b_chain = make_chain(PromptTemplate(
    input_variables=["solution","stats","trl","features"],
    template="""
Draft a 1500–2000 character analysis titled "The innovation: Solution/Product or Services (USP)".
Use solution: {solution}, stats: {stats}, TRL: {trl}, features: {features}. Do NOT truncate.
"""))
c_chain = make_chain(PromptTemplate(
    input_variables=["solution","stats","revenue","features"],
    template="""
Draft a 1500–2000 character analysis titled "Market and Competition analysis".
Use solution: {solution}, stats: {stats}, revenue: {revenue}, features: {features}. Do NOT truncate.
"""))
d_chain = make_chain(PromptTemplate(
    input_variables=["solution","stats"],
    template="""
Draft a 1500–2000 character analysis titled "Broad impacts".
Use solution: {solution}, stats: {stats}. Do NOT truncate.
"""))
e_chain = make_chain(PromptTemplate(
    input_variables=["solution","revenue","trl"],
    template="""
Draft a 1500–2000 character analysis titled "Funding rationale and MVP".
Use solution: {solution}, revenue: {revenue}, TRL: {trl}. Do NOT truncate.
"""))

@app.post("/webhook")
async def receive_form(request: Request):
    data = await request.json()
    questions = data.get("submission", {}).get("questions", [])
    if not questions:
        return JSONResponse(status_code=400, content={"detail": "No form data received."})

    # Map question names to values
    field_map = {q.get("name", "").lower(): q.get("value", "").strip() for q in questions}
    def get_field(keyword: str):
        for k, v in field_map.items():
            if keyword in k:
                return v
        return ""

    solution = get_field("solution")
    problem = get_field("problem")
    features = get_field("unique")
    trl = get_field("trl")
    revenue = get_field("revenue")
    email = get_field("email")

    missing = [name for name, val in [("solution", solution), ("problem", problem), ("features", features), ("trl", trl), ("revenue", revenue), ("email", email)] if not val]
    if missing:
        return JSONResponse(status_code=400, content={"detail": f"Missing fields: {', '.join(missing)}"})

    logger.info(f"Fields OK; processing for email: {email}")

    # Build search snippets
    def snippet(chain, text):
        query = chain.run(text=text)
        res = serp.run(query)
        return "\n".join(res[:5]) if isinstance(res, list) else str(res)

    stats_a = snippet(a_query, problem)
    stats_b = snippet(b_query, solution)
    stats_c = snippet(c_query, features)
    stats_d = snippet(d_query, features)

    # Generate analyses
    a = a_chain.run(problem=problem, stats=stats_a)
    b = b_chain.run(solution=solution, stats=stats_b, trl=trl, features=features)
    c = c_chain.run(solution=solution, stats=stats_b, revenue=revenue, features=features)
    d = d_chain.run(solution=solution, stats=stats_d)
    e = e_chain.run(solution=solution, revenue=revenue, trl=trl)

    # Compile Word document
    doc = Document()
    for title, content in [
        ("The problem/market opportunity", a),
        ("The innovation: Solution/Product or Services (USP)", b),
        ("Market and Competition analysis", c),
        ("Broad impacts", d),
        ("Funding rationale and MVP", e)
    ]:
        doc.add_heading(title, level=1)
        doc.add_paragraph(content)
    filename = "analyses.docx"
    doc.save(filename)
    logger.info("Document saved: %s", filename)

    # Email attachment via SendGrid
    with open(filename, "rb") as f:
        data_enc = base64.b64encode(f.read()).decode()
    mail = Mail(
        from_email=SENDGRID_SENDER,
        to_emails=email,
        subject="Your AI-Generated Analyses",
        html_content="<p>Your analyses are attached.</p>"
    )
    attach = Attachment(
        FileContent(data_enc),
        FileName(filename),
        FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        Disposition("attachment")
    )
    mail.attachment = attach
    resp = SendGridAPIClient(SENDGRID_API_KEY).send(mail)
    logger.info(f"Email sent; status {resp.status_code}")

    # Return response
    download_url = request.url._url.rstrip(request.url.path) + "/download"
    return {"status": "ok", "email_status": resp.status_code, "download_url": download_url}

@app.get("/download")
async def download_doc():
    path = os.path.join(os.getcwd(), "analyses.docx")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"detail": "Document not found"})
    return FileResponse(
        path=path,
        filename="analyses.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")