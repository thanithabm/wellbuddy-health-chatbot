"""
WellBuddy - a simple, SAFE health-information chatbot (RAG project).

What it does:
    You ask a general health question -> we find the matching part of our health
    notes -> an AI answers ONLY from those notes -> you get a JSON answer that
    includes an "urgency" level (self_care / see_a_doctor / emergency).

Works with OR without an API key (see providers.py):
    Gemini key  -> best answers
    Ollama      -> free local AI (e.g. Qwen), no key
    nothing     -> "notes-only" mode: shows the best-matching section of the notes

Why "safe"? Health is risky, so there are 3 safety layers:
    1. SAFETY RULES run BEFORE any AI: emergencies, crisis words, diagnosis and
       medicine-dose requests get a fixed, careful reply.
    2. GROUNDING: the AI may only use our notes, and must say "not covered" otherwise.
    3. A DISCLAIMER is always added by our code (not by the AI).

This is a learning project, NOT medical advice.
Run it with:   python -m uvicorn app:app --reload
"""
import re
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import providers

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"

DISCLAIMER = (
    "General information only, not medical advice. This assistant cannot diagnose "
    "or prescribe. In an emergency call 112 or go to the nearest hospital."
)

# ==========================================================================
# SAFETY LAYER 1: simple rules that run BEFORE any AI is called.
# (Real products use trained classifiers; keywords keep this project easy to read.
#  They may trigger on informational questions too - that is the safe direction.)
# ==========================================================================
EMERGENCY_PATTERNS = [
    r"chest pain", r"can'?t breathe", r"cannot breathe", r"not breathing",
    r"difficulty breathing", r"unconscious", r"not responding", r"seizure",
    r"overdose", r"swallowed (poison|bleach|pesticide)", r"heavy bleeding",
    r"bleeding (a lot|heavily|won'?t stop)", r"face is drooping", r"slurred speech",
]
CRISIS_PATTERNS = [
    r"suicid", r"kill myself", r"end my life", r"want to die", r"wanna die",
    r"hurt myself", r"self[- ]?harm", r"no reason to live",
]
# Questions we must not answer: diagnosis, medicine names, doses.
MEDICAL_ADVICE_PATTERNS = [
    r"\bdoses?\b", r"\bdosage\b", r"how (many|much) (mg|ml|tablets?|pills?|capsules?)", r"\b\d+\s?(mg|ml)\b",
    r"(which|what) (medicine|medication|tablet|drug|antibiotic)s?\b", r"\bprescri(be|ption)\b",
    r"\bdo i have (diabetes|dengue|malaria|typhoid|cancer|covid|asthma|thyroid|hiv|tb|tuberculosis)\b",
    r"\bdiagnos(e|is) me\b", r"what'?s wrong with me",
]

EMERGENCY_REPLY = (
    "This could be a medical emergency. Please do not wait for a chatbot. "
    "Call 112 (India's emergency number) or go to the nearest hospital emergency "
    "department right now. If someone is with you, ask them to stay with you and help."
)
CRISIS_REPLY = (
    "I'm really sorry you're going through this. You don't have to handle it alone, and help is "
    "available right now. If you might act on these thoughts or are in immediate danger, call 112 "
    "or go to the nearest hospital. You can also talk to a trained counsellor, free and confidential, "
    "24x7, on Tele-MANAS: 14416 or 1-800-891-4416. If you can, tell a family member or friend you "
    "trust and ask them to stay with you."
)
NO_ADVICE_REPLY = (
    "I can't diagnose conditions or recommend medicines or doses. A doctor or pharmacist can do that "
    "safely after asking about your age, health history and other medicines. I can share general "
    "self-care tips and warning signs for common symptoms if you ask about those."
)


def safety_check(question: str) -> dict | None:
    """Return a ready-made reply if the question needs special handling, else None."""
    q = question.lower()
    if any(re.search(p, q) for p in CRISIS_PATTERNS):
        return {"answer": CRISIS_REPLY, "urgency": "emergency", "found_in_docs": True,
                "sections_used": ["Safety rule: crisis support"]}
    if any(re.search(p, q) for p in EMERGENCY_PATTERNS):
        return {"answer": EMERGENCY_REPLY, "urgency": "emergency", "found_in_docs": True,
                "sections_used": ["Safety rule: emergency"]}
    if any(re.search(p, q) for p in MEDICAL_ADVICE_PATTERNS):
        return {"answer": NO_ADVICE_REPLY, "urgency": "see_a_doctor", "found_in_docs": False,
                "sections_used": ["Safety rule: no diagnosis or medicines"]}
    return None


# ==========================================================================
# STEP 1: LOAD + SPLIT the health notes into chunks (one chunk per "## Heading").
# ==========================================================================
def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for part in re.split(r"\n(?=## )", text):
            if part.startswith("## "):
                title = part.split("\n", 1)[0][3:].strip()
                chunks.append({"section": title, "file": path.name, "text": part.strip()})
    return chunks


# ==========================================================================
# STEP 2 + 3: EMBED the chunks (turn text into numbers) and STORE them in ChromaDB.
# The index lives in memory and is rebuilt every time the app starts.
# ==========================================================================
db = chromadb.Client()
collection = db.get_or_create_collection("health_notes", metadata={"hnsw:space": "cosine"})


def build_index() -> int:
    if collection.count() == 0:
        chunks = load_chunks()
        collection.add(
            ids=[f"chunk{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            embeddings=providers.embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT"),
            metadatas=[{"section": c["section"], "file": c["file"]} for c in chunks],
        )
    return collection.count()


# ==========================================================================
# STEP 4: RETRIEVE - find the 3 chunks closest in meaning to the question.
# ==========================================================================
def retrieve(question: str, k: int = 3) -> list[dict]:
    result = collection.query(query_embeddings=providers.embed([question], "RETRIEVAL_QUERY"), n_results=k)
    return [
        {"section": m["section"], "file": m["file"], "text": d, "score": round(1 - dist, 3)}
        for d, m, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


# ==========================================================================
# STEP 5: GENERATE - the AI answers from those chunks, in a fixed JSON shape.
# ==========================================================================
class Answer(BaseModel):
    answer: str = Field(description="Short, clear, friendly answer in simple language")
    urgency: str = Field(description="One of: self_care, see_a_doctor, emergency")
    found_in_docs: bool = Field(description="False if the context does not contain the answer")
    sections_used: list[str] = Field(description="Headings of the context sections used")


SYSTEM_PROMPT = """You are WellBuddy, a careful assistant that shares GENERAL health information.
Rules:
1. Use ONLY the CONTEXT. Never add medical facts that are not in the CONTEXT.
2. NEVER diagnose ("you have..."), never name medicines or give doses, never promise outcomes.
   If asked for a diagnosis, a medicine or a dose, set found_in_docs=false, explain you cannot
   help with that, and advise talking to a doctor or pharmacist.
3. If the CONTEXT does not answer the question, set found_in_docs=false and say it is not covered here.
4. If the CONTEXT lists warning signs or when to see a doctor, include them briefly.
5. Use simple language, 2-5 short sentences. Do not use scary words.
6. urgency: "self_care" = home care is fine; "see_a_doctor" = the person should consult a doctor
   (also use this when the answer is not covered); "emergency" = needs emergency help now, or the
   answer is about emergency warning signs (stroke, heart attack, severe allergic reaction).
7. In sections_used, list only the headings you actually used.
8. Never answer with only a few words. Give the key advice from the CONTEXT in 1-3 full sentences."""

VALID_URGENCY = {"self_care", "see_a_doctor", "emergency"}


def notes_only_answer(chunks: list[dict]) -> dict:
    """No-AI mode: show the best-matching note, or say 'not covered' if nothing matches well."""
    best = chunks[0]
    if best["score"] < providers.min_score():
        return {"answer": "I don't have that in my notes. Please ask a doctor or pharmacist, "
                          "or try asking in a different way.",
                "urgency": "see_a_doctor", "found_in_docs": False, "sections_used": []}
    body = best["text"].split("\n", 1)[1].strip()
    if best["file"] == "emergency_signs.md":
        urgency = "emergency"
    elif re.search(r"see a doctor|medical help|medical attention|get help|seek", body, re.I):
        urgency = "see_a_doctor"
    else:
        urgency = "self_care"
    return {"answer": f"From my notes on '{best['section']}': {body}",
            "urgency": urgency, "found_in_docs": True, "sections_used": [best["section"]]}


def generate(question: str, chunks: list[dict]) -> dict:
    if providers.llm_backend() == "none":
        return notes_only_answer(chunks)
    context = "\n\n".join(c["text"] for c in chunks)
    try:
        text = providers.generate_json(SYSTEM_PROMPT, f"CONTEXT:\n{context}\n\nQUESTION: {question}", Answer)
        data = Answer.model_validate_json(text).model_dump()
    except Exception as e:  # never crash the chat: fall back to notes-only and tell the developer
        print(f"WARNING: LLM call failed ({type(e).__name__}: {e}). Falling back to notes-only mode.")
        return notes_only_answer(chunks)
    if data["urgency"] not in VALID_URGENCY:  # never trust AI output blindly
        data["urgency"] = "see_a_doctor"
    if not data["found_in_docs"]:  # not covered -> never label it "self_care"
        data["urgency"] = "see_a_doctor"
    return data


def mode_text() -> str:
    return f"embeddings={providers.embedder_name()}, llm={providers.llm_backend()}"


def answer_question(question: str) -> dict:
    """Whole pipeline: safety rules -> retrieve -> generate -> add disclaimer."""
    rule_reply = safety_check(question)
    if rule_reply:  # handled by our rules, no AI call at all
        return {"question": question, **rule_reply, "retrieved_sections": [], "top_score": None,
                "handled_by": "safety_rule", "mode": mode_text(), "disclaimer": DISCLAIMER}

    chunks = retrieve(question)
    result = generate(question, chunks)
    return {"question": question, **result,
            "retrieved_sections": [c["section"] for c in chunks], "top_score": chunks[0]["score"],
            "handled_by": "rag", "mode": mode_text(), "disclaimer": DISCLAIMER}


# ==========================================================================
# STEP 6: SERVE as a web API with FastAPI.  POST /ask   GET /   GET /docs
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    n = build_index()
    print(f"Indexed {n} health-note sections. Mode: {mode_text()}")
    if providers.llm_backend() == "none":
        print("NOTE: no AI model is active, so answers show the best-matching note (notes-only mode).")
    yield


app = FastAPI(title="WellBuddy - Health Information Assistant", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=300, examples=["How do I treat a minor burn?"])


@app.post("/ask")
def ask(body: Question):
    return answer_question(body.question)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count(), "mode": mode_text()}


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")
