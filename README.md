# WellBuddy: A Safe Health-Information Chatbot (RAG)

A small AI chatbot that answers **general health questions** (cold, fever, first aid, sleep, nutrition...) using only a set of health notes, and **knows when NOT to answer**: emergencies, diagnosis requests and medicine doses.

Built with **Python, FastAPI, ChromaDB (vector database) and the Gemini API**. It returns **structured JSON** with an urgency level.

> Learning project. The notes in `data/docs/` are general wellness information written for this demo. It is **not medical advice** and not a medical device.

---

## What problem does it solve?

People search health questions online all the time, and a normal chatbot may answer confidently and wrongly. In health, that can be harmful. So this project follows one rule: **answer only from trusted notes, and send the person to a doctor or emergency help when needed.**

## How it works (simple version)

```
          your question
               │
   ┌───────────▼────────────┐   emergency / crisis words?
   │ 1. SAFETY RULES (code) │───────────────► fixed careful reply (112 / helpline), AI is NOT called
   └───────────┬────────────┘
               │ normal question
   ┌───────────▼────────────┐
   │ 2. SEARCH the notes    │  question → numbers (embedding) → ChromaDB finds the 3 closest sections
   └───────────┬────────────┘
   ┌───────────▼────────────┐
   │ 3. AI writes answer    │  Gemini or Ollama sees ONLY those 3 sections + strict rules
   │                        │  (or notes-only mode: show the best section, no AI)
   └───────────┬────────────┘
               ▼
   JSON: answer, urgency, found_in_docs, sections_used, disclaimer
```

**Open-book exam analogy:** the AI is a student who must answer using only the open book (our notes) and must say "not in the book" if it can't find it.

## The 3 safety layers (the best part to talk about in interviews)

| Layer | What it does | Where |
|---|---|---|
| 1. Safety rules | "Chest pain" or "want to end my life" get a fixed, careful reply with 112 / Tele-MANAS (14416). Diagnosis and medicine/dose requests get a fixed refusal. All **before** any AI is used | `safety_check()` |
| 2. Grounding | AI may use only the retrieved notes; otherwise it must say `found_in_docs: false` | `SYSTEM_PROMPT` |
| 3. No diagnosis or doses | Prompt forbids "you have..." and medicine names/doses; the disclaimer is added by our code, not by the AI | `SYSTEM_PROMPT`, `DISCLAIMER` |

Also: the urgency value from the AI is validated in code, because we never trust AI output blindly.

## Run it (no API key needed)

```bash
python -m venv .venv
.venv\Scripts\activate              # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** (chat page) or **http://127.0.0.1:8000/docs** (API playground).

The app picks the best option it can find. The startup line shows which one: `Mode: embeddings=..., llm=...`.

| Mode | What you need | What you get |
|---|---|---|
| **A. Gemini** | Free key in `.env`: `GEMINI_API_KEY=...` | Best answers, written by Gemini |
| **B. Ollama** (open-source model on your PC) | Install [Ollama](https://ollama.com), then `ollama pull qwen2.5:3b` (about 2 GB; use `qwen2.5:1.5b` if your PC has little RAM) | Real AI-written answers, no key, offline |
| **C. Notes-only** | Nothing | Shows the best-matching section of the notes (no AI-written text) |

Search (embeddings) works without a key too: ChromaDB downloads a small free model (about 80 MB) the first time. If that download fails, a tiny built-in fallback is used.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I treat a minor burn?"}'
```

Example response (shape):

```json
{
  "question": "How do I treat a minor burn?",
  "answer": "Cool the burn under cool running water for 20 minutes. Do not use ice, butter or toothpaste, and do not burst blisters. Get medical help for large burns or burns on the face, hands or feet.",
  "urgency": "self_care",
  "found_in_docs": true,
  "sections_used": ["Minor burns"],
  "retrieved_sections": ["Minor burns", "Minor cuts and scrapes", "Fainting"],
  "top_score": 0.71,
  "handled_by": "rag",
  "mode": "embeddings=chroma, llm=ollama",
  "disclaimer": "General information only, not medical advice..."
}
```

## Check how good it is

```bash
python check_answers.py
```

Runs 15 test questions: 10 answerable, 3 that must be refused (dose, diagnosis, off-topic) and 2 safety-rule cases. Prints a score and the mode used. Scores differ by mode, so write down which mode you tested. Write your score here after running:

| Result | Value |
|---|---|
| Mode tested | _e.g. embeddings=chroma, llm=ollama_ |
| Score | _run it and fill in_ |

## Files

```
app.py              the whole app, in 6 commented steps
providers.py        the engines: Gemini / Ollama / notes-only, and the embedders
data/docs/*.md      the health notes (the "open book")
static/index.html   simple chat page
check_answers.py    mini exam / evaluation
```

## Limitations (be honest about these)

- Keyword safety rules are crude: they miss some phrasings and may trigger on informational questions. Real products use trained classifiers and human review.
- The notes are small and written for the demo; they must be reviewed by a medical professional before any real use.
- No conversation memory, no user accounts, no storage of personal data.
- Emergency and helpline numbers are for India; change them for other countries.

## Ideas to extend

1. Chat history so follow-up questions work.
2. Add more notes (or PDFs from trusted sources such as WHO or government health sites) and re-measure the score.
3. Kannada / Hindi questions.
4. Better safety: an AI classifier for emergencies, plus logging of refused questions.
5. Deploy it (Render, Railway or AWS).
