# SmartBuyer AI — AI Extraction Backend

## What This Is
Python/FastAPI service that handles document processing, AI-powered extraction, and audit reasoning. This is where Gemini, Claude, and Groq are called.

**Patent:** US 2025/0348890 A1

## Architecture Layers
This repo contains **Layer 1 (OCR/Extraction)** and **Layer 2 (Audit Reasoning)**:

1. **Gemini OCR (Layer 1)** → 2. **Claude reasoning (Layer 2)** → 3. TypeScript scoring (backend repo) → 4. OpenAI narrative (backend repo) → 5. Replay/storage (backend repo)

## Governing Rules — NEVER VIOLATE
1. **Observe, Validate, Report.** Never mutate without approval.
2. **No credentials in chat output.** Reference .env by path only.
3. **Layer separation is sacred:**
   - Gemini = OCR and document extraction ONLY (Layer 1)
   - Claude = audit reasoning and flag identification ONLY (Layer 2)
   - This repo MUST NOT compute the SmartBuyer Score — that is the backend's job (Layer 3)
   - This repo MUST NOT generate consumer narratives — that is OpenAI's job (Layer 4)
4. **Deterministic scoring lives in the backend, not here.** This service outputs structured data that the backend scores deterministically.

## Tech Stack
- Python 3, FastAPI
- Google Document AI (GCP) for OCR
- Gemini for extraction
- Claude (Anthropic) for audit reasoning
- Groq for supplementary processing
- OpenAI for specific extraction tasks

## Key Structure
```
App/
  core/config.py                — Settings (GCP, Groq, API keys)
  services/
    extraction/                 — Document upload + OCR + extraction
      gemini_extractor.py       — Gemini-based extraction
      ocr_extractor.py          — GCP Document AI OCR
      document_extract.py       — Document extraction pipeline
    contract/                   — Multi-image contract analysis
    lease/                      — Lease document analysis
    rate_helper/                — Audit classification, flags, scoring helpers
      audit_classifier.py       — Audit type classification
      audit_flags.py            — Flag identification
      scoring_engine.py         — LEGACY Python scoring (superseded by backend TypeScript)
      rules/                    — Flag registry, pricing caps, suppression pairs
    rating/                     — Deal rating service
    chatbot/                    — AI chatbot service
    quiz/                       — Academy quiz generation
main.py                        — FastAPI app entry point
requirements.txt               — Python dependencies
```

## Important Notes
- `App/services/rate_helper/scoring_engine.py` is a LEGACY Python scoring engine. The authoritative scoring engine is now in the backend repo (`src/app/scoring/scoringEngine.ts`). This Python version should NOT be used for production scoring.
- The `rate_helper/rules/` directory contains an older set of rules. The authoritative rules are in the backend repo.
- There are 3 copies of this service on the VPS: `therubberbandit-Ai`, `therubberbandit-Ai2`, `therubberbandit-Ai2-2`. Only `therubberbandit-Ai2-2` is the active production instance.

## Environment Variables (in .env)
- `GCP_PROJECT_ID`, `GCP_LOCATION`, `GCP_PROCESSOR_ID` — Google Document AI
- `GROQ_URL`, `GROQ_MODEL`, `GROQ_API_KEY` — Groq API
- `OPENAI_API_KEY` — OpenAI
- `GEMINI_API_KEY`, `GEMINI_MODEL` — Google Gemini
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` — Anthropic Claude

## Deployment
```bash
ssh root@86.38.218.159
cd /var/www/therubberbandit-Ai2-2
git pull origin main
docker compose down
docker compose build --no-cache
docker compose up -d
```

Container: `rubber` on port 8008.
API docs: https://ai.smartbuyerai.com/docs (FastAPI Swagger UI)
