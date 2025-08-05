# main.py

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import logging
import os
from dotenv import load_dotenv

# LangChain imports (community versions)
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 3️⃣ Initialize FastAPI
app = FastAPI()

# 4️⃣ Initialize SerpAPI and chat LLM
serp = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2048
)

# 5️⃣ Define PromptTemplates and LLMChains
def make_chain(prompt_template: PromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt_template)

# a) Query for problem scale & urgency
template_a = PromptTemplate(
    input_variables=["problem_text"],
    template=(
        "Generate a concise search query to find publicly available statistics illustrating "
        "the scale and urgency of this problem.\n\n"
        "Problem: {problem_text}\n\nSearch Query:"
    )
)
chain_a = make_chain(template_a)

# b) Query for market size & trends
template_b = PromptTemplate(
    input_variables=["solution_text"],
    template=(
        "Generate a concise search query to find the market addressed by this product, "
        "including market size and key trends.\n\n"
        "Product: {solution_text}\n\nSearch Query:"
    )
)
chain_b = make_chain(template_b)

# c) Query for competitors & novelty
template_c = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "Generate a concise search query to identify companies offering similar features "
        "and assess the degree of novelty.\n\n"
        "Features: {features_text}\n\nSearch Query:"
    )
)
chain_c = make_chain(template_c)

# d) Query for competitor revenue streams
template_d = PromptTemplate(
    input_variables=["features_text"],
    template=(
        "Generate a concise search query to find revenue streams competitors derive "
        "from products with similar features.\n\n"
        "Features: {features_text}\n\nSearch Query:"
    )
)
chain_d = make_chain(template_d)

# 6️⃣ Analysis templates (all corrected to single triple-quoted strings)
template_analysis_a = PromptTemplate(
    input_variables=["problem_text", "search_results"],
    template="""
Draft a 1500–2000 character analysis titled "The problem/market opportunity".

Use the user’s problem description:
{problem_text}

Incorporate these statistics/snippets:
{search_results}

Substantiate the analysis with concrete data illustrating the scale and urgency of the problem.
Do NOT truncate mid-sentence.
"""
)
analysis_a_chain = make_chain(template_analysis_a)

template_analysis_b = PromptTemplate(
    input_variables=["solution_text", "search_results", "trl", "features_text"],
    template="""
Draft a 1500–2000 character analysis titled "The innovation: Solution/Product or Services (USP)".

1. Explain why this product is better than existing solutions, using:
{search_results}

2. State the current Technology Readiness Level (TRL) as {trl}, including any validation/certification details.

3. Explain why now is the right time to bring this product to market.

Unique features to emphasize: {features_text}

Do NOT truncate mid-sentence.
"""
)
analysis_b_chain = make_chain(template_analysis_b)

template_analysis_c = PromptTemplate(
    input_variables=["solution_text", "search_results", "revenue_model", "features_text"],
    template="""
Draft a 1500–2000 character analysis titled "Market and Competition analysis".

1. Describe market size and key trends, using:
{search_results}

2. Explain the product’s potential to transform this market or create a new one.

3. Detail the business model and envisioned revenue streams: {revenue_model}

4. Show why features {features_text} will convince clients to buy.

5. List advantages and disadvantages, and argue why this product is likely to succeed.

Do NOT truncate mid-sentence.
"""
)
analysis_c_chain = make_chain(template_analysis_c)

template_analysis_d = PromptTemplate(
    input_variables=["solution_text", "search_results"],
    template="""
Draft a 1500–2000 character analysis titled "Broad impacts".

Discuss the potential societal, environmental, or climate impacts of the product, and estimate job creation, using:
{search_results}

Do NOT truncate mid-sentence.
"""
)
analysis_d_chain = make_chain(template_analysis_d)

template_analysis_e = PromptTemplate(
    input_variables=["solution_text", "revenue_model", "trl"],
    template="""
Draft a 1500–2000 character analysis titled "Funding rationale and MVP".

Explain why raising funding is nearly impossible at TRL {trl}, why building a minimum viable product (MVP) is crucial, and include any funding history if provided. Use:
Revenue model: {revenue_model}

Do NOT truncate mid-sentence.
"""
)
analysis_e_chain = make_chain(template_analysis_e)

@app.post("/webhook")
async def receive_form(request: Request):
    data = await request.json()
    questions = data.get("submission", {}).get("questions", [])
    if len(questions) < 6:
        return JSONResponse(status_code=400, content={"detail": "Expected 6 form fields (5 inputs + email)"})

    # 7️⃣ Extract inputs and email
    try:
        solution = questions[0]["value"].strip()
        problem = questions[1]["value"].strip()
        features = questions[2]["value"].strip()
        trl = questions[3]["value"].strip()
        revenue = questions[4]["value"].strip()
        email = questions[5]["value"].strip()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Form fields missing or misordered"})

    if not email:
        return JSONResponse(status_code=400, content={"detail": "Email address is required"})

    logger.info(f"Inputs received; will email to {email}")

    # 8️⃣ Perform search queries and build snippets
    def snippets_for(chain: LLMChain, text: str) -> str:
        query = chain.run(**{chain.prompt.input_variables[0]: text})
        results = serp.run(query)
        if isinstance(results, list):
            return "\n".join(results[:5])
        return str(results)

    try:
        s_a = snippets_for(chain_a, problem)
        s_b = snippets_for(chain_b, solution)
        s_c = snippets_for(chain_c, features)
        s_d = snippets_for(chain_d, features)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return JSONResponse(status_code=500, content={"detail": "Search step failed"})

    # 9️⃣ Generate each of the five analyses
    try:
        a = analysis_a_chain.run(problem_text=problem, search_results=s_a)
        b = analysis_b_chain.run(solution_text=solution, search_results=s_b, trl=trl, features_text=features)
        c = analysis_c_chain.run(solution_text=solution, search_results=s_b, revenue_model=revenue, features_text=features)
        d = analysis_d_chain.run(solution_text=solution, search_results=s_d)
        e = analysis_e_chain.run(solution_text=solution, revenue_model=revenue, trl=trl)
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}")
        return JSONResponse(status_code=500, content={"detail": "Analysis generation failed"})

    # 10️⃣ Compile all analyses into a Word document
    doc = Document()
    for title, content in [
        ("The problem/market opportunity", a),
        ("The innovation: Solution/Product or Services (USP)", b),
        ("Market and Competition analysis", c),
        ("Broad impacts", d),
        ("Funding rationale and MVP", e),
    ]:
        doc.add_heading(title, level=1)
        doc.add_paragraph(content)

    filename = "analyses.docx"
    doc.save(filename)
    logger.info(f"Saved document: {filename}")

    # 11️⃣ Email the document as attachment
    try:
        with open(filename, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        message = Mail(
            from_email=SENDGRID_SENDER,
            to_emails=email,
            subject="Your AI-Generated Analyses",
            html_content="<p>Your requested analyses are attached.</p>"
        )
        attach = Attachment(
            FileContent(encoded),
            FileName(filename),
            FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            Disposition("attachment")
        )
        message.attachment = attach
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        resp = sg.send(message)
        logger.info(f"Email sent; status {resp.status_code}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Email send failed: {e}"})

    # 12️⃣ Return success
    download_url = request.url._url.rstrip(request.url.path) + "/download"
    return {"status": "ok", "email_status": resp.status_code, "download_url": download_url}


@app.get("/download")
async def download():
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
