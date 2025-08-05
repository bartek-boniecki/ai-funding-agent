import os
import re
import uuid
import asyncio
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import base64
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.utilities import SerpAPIWrapper

from docx import Document

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

# -------------------------------------------------------------------
# Bootstrap
# -------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("grant-agent")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_SENDER = os.getenv("SENDGRID_SENDER")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # optional
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

# -------------------------------------------------------------------
# LLM + prompts (modern LangChain style)
# -------------------------------------------------------------------
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, timeout=120, max_retries=2)
parser = StrOutputParser()

QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You craft precise web search queries for professional market & problem research. Return ONLY the query string."),
    ("human", "Task: {task}\nContext: {context}\nReturn one concise search query.")
])

RESEARCH_TASKS = {
    "problem": "Find publicly available statistics quantifying the SCALE and URGENCY of the problem.",
    "market": "Identify the addressed market, market SIZE (global/regional), and all KEY TRENDS.",
    "novelty": "Find competitive products with similar features and assess degree of NOVELTY vs existing solutions.",
    "revenues": "Identify typical REVENUE STREAMS key players and startups generate from similar products."
}

ANALYSIS_PROMPTS = {
    "problem": ChatPromptTemplate.from_messages([
        ("system", "Write a 1,500–2,000 character, in-depth analysis fine-tuned to the standards of VC investors showing the scale and urgency of the problem with inline numeric evidence and citations [n]. Use only provided notes + user's text."),
        ("human",
         "Title: The problem/market opportunity\n\n"
         "User problem description:\n{problem}\n\n"
         "Research notes:\n{notes}\n\n"
         "Write 1500–2000 characters. Include concrete figures and cite them inline as [n].")
    ]),
    "innovation": ChatPromptTemplate.from_messages([
        ("system", "Write a 1,500–2,000 character, in-depth, and unbiased analysis of the degree of novelty that is fine-tuned to the standards of VC investors with citations [n]. Be precise and non-generic."),
        ("human",
         "Title: The innovation: Solution/Product or Services (USP)\n\n"
         "User solution:\n{solution}\n\n"
         "Unique features:\n{features}\n\n"
         "User-stated TRL: {trl}\n\n"
         "Research notes (competitors/validation):\n{notes}\n\n"
         "Cover: (1) why the features of the innovation make it much better than existing solutions and bring sufficient added value to trigger demand from potential customers; (2) EC TRL and any validation; (3) what latest market, societal or technological trends show now is the right timing to launch the innovation. Cite [n]. 1500–2000 chars.")
    ]),
    "market": ChatPromptTemplate.from_messages([
        ("system", "Write a 1,500–2,000 character unbiased market analysis fine-tuned to the standards of VC investors with citations [n]."),
        ("human",
         "Title: Market and Competition analysis\n\n"
         "User solution:\n{solution}\n\n"
         "Unique features:\n{features}\n\n"
         "User revenue ideas:\n{revenue}\n\n"
         "Research notes (market size & trends):\n{market_notes}\n\n"
         "Research notes (competitors & novelty):\n{novelty_notes}\n\n"
         "Research notes (competitor revenues):\n{rev_notes}\n\n"
         "Cover: (i) market size & trends; (ii) potential of the innovation to transform the market; (iii) optimal business model & revenue model; (iv) why the innovative features of the innovation drive adoption; (v) pros/cons of the innovation & why it is likely to succeed. Cite [n]. 1500–2000 chars.")
    ]),
    "impact": ChatPromptTemplate.from_messages([
        ("system", "Write a 1,500–2,000 character impacts analysis with citations [n]."),
        ("human",
         "Title: Broad impacts\n\n"
         "User solution:\n{solution}\n\n"
         "Research notes (jobs, social/environmental/climate):\n{notes}\n\n"
         "Discuss potential societal/environmental/climate impacts and estimate jobs created. Cite [n]. 1500–2000 chars.")
    ]),
    "funding": ChatPromptTemplate.from_messages([
        ("system", "Write a candid 1,500–2,000 character funding analysis."),
        ("human",
         "Title: Funding rationale and MVP\n\n"
         "User solution:\n{solution}\n\n"
         "User-stated TRL: {trl}\n\n"
         "User revenue ideas:\n{revenue}\n\n"
         "Explain why funding is hard at this TRL per EC definitions, why MVP is crucial now, and why it will enable successful fundraising . 1500–2000 chars.")
    ])
}

TRL_MAP = {
    "1": "TRL 1 – basic principles observed",
    "2": "TRL 2 – technology concept formulated",
    "3": "TRL 3 – experimental proof of concept",
    "4": "TRL 4 – technology validated in lab",
    "5": "TRL 5 – technology validated in relevant environment",
    "6": "TRL 6 – technology demonstrated in relevant environment",
    "7": "TRL 7 – system prototype demonstration in operational environment",
    "8": "TRL 8 – system complete and qualified",
    "9": "TRL 9 – actual system proven in operational environment",
}
EMAIL_RE = re.compile(r"[^@]+@[^@]+\.[^@]+")

# -------------------------------------------------------------------
# Minimal, robust Fillout extraction (no reliance on names)
# -------------------------------------------------------------------
def extract_answers(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Fillout webhook = { submission: { questions: [ {id,name,type,value}, ... ] } }
    Some workspaces hide 'name'. We:
      1) detect email by value regex or type containing 'Email'
      2) take remaining text-like answers in order:
         0->solution, 1->problem, 2->features, 3->trl, 4->revenue
    """
    sub = (payload or {}).get("submission", {})
    qs = sub.get("questions", []) or []
    vals: List[str] = []
    email = ""
    for q in qs:
        val = q.get("value")
        qtype = (q.get("type") or "").lower()
        if isinstance(val, str) and EMAIL_RE.fullmatch(val.strip()):
            email = val.strip()
            continue
        if "email" in qtype and isinstance(val, str) and val.strip():
            email = val.strip()
            continue
        if isinstance(val, str) and val.strip():
            vals.append(val.strip())

    # Map by position for the five business answers
    labels = ["solution", "problem", "features", "trl", "revenue"]
    out = {k: (vals[i] if i < len(vals) else "") for i, k in enumerate(labels)}
    out["email"] = email or "no-reply@example.com"
    return out

def normalize_trl(text: str) -> str:
    if not text:
        return "Unspecified"
    digits = re.findall(r"\d", text)
    if digits and digits[0] in TRL_MAP:
        return TRL_MAP[digits[0]]
    return text.strip()

# -------------------------------------------------------------------
# Search helpers
# -------------------------------------------------------------------
def init_serp() -> Optional[SerpAPIWrapper]:
    if not SERPAPI_API_KEY:
        return None
    return SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

def search_serp_sync(serp: SerpAPIWrapper, query: str, k: int = 5) -> List[Dict[str, str]]:
    try:
        raw = serp.results(query)  # dict (per LangChain SerpAPIWrapper docs)
        items = (raw or {}).get("organic_results", [])[:k]
        return [{
            "title": it.get("title", "") or "",
            "link": it.get("link", "") or "",
            "snippet": it.get("snippet", "") or ""
        } for it in items]
    except Exception as e:
        log.warning(f"SerpAPI error: {e}")
        return []

def notes_and_refs(results: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    links: List[str] = []
    lines: List[str] = []
    for it in results:
        url = it.get("link")
        if not url:
            continue
        if url not in links:
            links.append(url)
        idx = links.index(url) + 1
        title = it.get("title", "").strip()
        snip = it.get("snippet", "").strip()
        lines.append(f"- {title}: {snip} [{idx}]")
    return ("\n".join(lines), links)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/webhook")
async def webhook(request: Request):
    """
    Always return 200 so Fillout doesn't retry. Errors are in JSON 'status'.
    """
    try:
        payload = await request.json()
    except Exception as e:
        log.error(f"Invalid JSON: {e}")
        return JSONResponse({"status": "error", "message": f"Invalid JSON: {e}"})

    # 1) Extract inputs (robust to missing field names)
    a = extract_answers(payload)
    sol, prob, feats, trl_in, rev = a["solution"], a["problem"], a["features"], a["trl"], a["revenue"]
    email = a["email"]
    trl = normalize_trl(trl_in)

    # 2) Refine 4 search queries (async, concise)
    async def refine(task: str, ctx: str) -> str:
        return await (QUERY_PROMPT | llm | parser).ainvoke({"task": task, "context": ctx})

    try:
        q_problem, q_market, q_novelty, q_revenues = await asyncio.gather(
            refine(RESEARCH_TASKS["problem"], prob),
            refine(RESEARCH_TASKS["market"], sol),
            refine(RESEARCH_TASKS["novelty"], feats),
            refine(RESEARCH_TASKS["revenues"], feats),
        )
    except Exception as e:
        log.error(f"Query refinement failed: {e}")
        return JSONResponse({"status": "error", "message": f"Query refinement failed: {e}"} )

    # 3) SerpAPI searches in parallel (if key present)
    serp = init_serp()
    research = {"problem": ("", []), "market": ("", []), "novelty": ("", []), "revenues": ("", [])}
    all_refs: List[str] = []

    try:
        if serp:
            loop = asyncio.get_running_loop()
            p, m, n, r = await asyncio.gather(
                loop.run_in_executor(None, search_serp_sync, serp, q_problem),
                loop.run_in_executor(None, search_serp_sync, serp, q_market),
                loop.run_in_executor(None, search_serp_sync, serp, q_novelty),
                loop.run_in_executor(None, search_serp_sync, serp, q_revenues),
            )
            p_notes, p_links = notes_and_refs(p)
            m_notes, m_links = notes_and_refs(m)
            n_notes, n_links = notes_and_refs(n)
            r_notes, r_links = notes_and_refs(r)
            research = {
                "problem": (p_notes, p_links),
                "market": (m_notes, m_links),
                "novelty": (n_notes, n_links),
                "revenues": (r_notes, r_links),
            }
            for url in p_links + m_links + n_links + r_links:
                if url not in all_refs:
                    all_refs.append(url)
        else:
            # No SerpAPI key—keep queries as notes
            research = {
                "problem": (f"Query: {q_problem}", []),
                "market": (f"Query: {q_market}", []),
                "novelty": (f"Query: {q_novelty}", []),
                "revenues": (f"Query: {q_revenues}", []),
            }
    except Exception as e:
        log.error(f"Search error: {e}")
        research = {
            "problem": (f"Query: {q_problem}", []),
            "market": (f"Query: {q_market}", []),
            "novelty": (f"Query: {q_novelty}", []),
            "revenues": (f"Query: {q_revenues}", []),
        }

    # 4) Generate 5 analyses (async)
    async def compose(prompt: ChatPromptTemplate, vars: Dict[str, Any]) -> str:
        return await (prompt | llm | parser).ainvoke(vars)

    try:
        a_problem, a_innov, a_market, a_impact, a_funding = await asyncio.gather(
            compose(ANALYSIS_PROMPTS["problem"], {"problem": prob, "notes": research["problem"][0]}),
            compose(ANALYSIS_PROMPTS["innovation"], {
                "solution": sol, "features": feats, "trl": trl,
                "notes": research["novelty"][0] or research["market"][0]
            }),
            compose(ANALYSIS_PROMPTS["market"], {
                "solution": sol, "features": feats, "revenue": rev,
                "market_notes": research["market"][0],
                "novelty_notes": research["novelty"][0],
                "rev_notes": research["revenues"][0],
            }),
            compose(ANALYSIS_PROMPTS["impact"], {"solution": sol, "notes": research["revenues"][0] or research["market"][0]}),
            compose(ANALYSIS_PROMPTS["funding"], {"solution": sol, "trl": trl, "revenue": rev}),
        )
    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        return JSONResponse({"status": "error", "message": f"LLM generation failed: {e}"})

    # 5) Build .docx
    try:
        fname = f"analyses_{uuid.uuid4().hex[:8]}.docx"
        doc = Document()
        doc.add_heading("The problem/market opportunity", level=1); doc.add_paragraph(a_problem)
        doc.add_heading("The innovation: Solution/Product or Services (USP)", level=1); doc.add_paragraph(a_innov)
        doc.add_heading("Market and Competition analysis", level=1); doc.add_paragraph(a_market)
        doc.add_heading("Broad impacts", level=1); doc.add_paragraph(a_impact)
        doc.add_heading("Funding rationale and MVP", level=1); doc.add_paragraph(a_funding)

        doc.add_heading("Appendix: Research notes", level=1)
        for title, key in [("Problem scale & urgency", "problem"), ("Market size & trends", "market"),
                           ("Competitors & novelty", "novelty"), ("Competitor revenue streams", "revenues")]:
            doc.add_heading(title, level=2)
            for line in (research[key][0] or "(No results)").split("\n"):
                if line.strip(): doc.add_paragraph(line)

        if all_refs:
            doc.add_heading("References", level=1)
            for i, url in enumerate(all_refs, 1):
                doc.add_paragraph(f"[{i}] {url}")

        doc.save(fname)
        log.info(f"Saved {fname}")
    except Exception as e:
        log.error(f"Document generation failed: {e}")
        return JSONResponse({"status": "error", "message": f"Document generation failed: {e}"})

    # 6) Email (optional)
    email_status = None
    try:
        if SENDGRID_API_KEY and SENDGRID_SENDER and email:
            with open(fname, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode()
            message = Mail(
                from_email=SENDGRID_SENDER,
                to_emails=email,
                subject="Your grant analyses (AI-generated with sources)",
                html_content="<p>Your analyses are attached. Appendix + References included.</p>"
            )
            message.attachment = Attachment(
                FileContent(data_b64),
                FileName(fname),
                FileType("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                Disposition("attachment")
            )
            resp = SendGridAPIClient(SENDGRID_API_KEY).send(message)
            email_status = resp.status_code
            log.info(f"Emailed {email} status={email_status}")
    except Exception as e:
        log.warning(f"Email failed: {e}")

    # 7) Response (always 200 OK)
    base = str(request.url).replace("/webhook", "")
    return JSONResponse({
        "status": "ok",
        "download_url": f"{base}/download/{fname}",
        "email_status": email_status,
    })

@app.get("/download/{fname}")
async def download(fname: str):
    path = os.path.join(os.getcwd(), fname)
    if not os.path.exists(path):
        raise HTTPException(404, detail="Document not found")
    return FileResponse(
        path=path,
        filename=fname,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
