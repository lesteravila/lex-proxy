"""
Lex Proxy — Railway deployment
Keeps all API keys server-side.
Endpoints:
  POST /llm        — Gemini
  POST /tts        — GCP Text-to-Speech
  GET  /stt-token  — Deepgram short-lived token
  POST /telegram   — Telegram sendMessage
"""

import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.auth.transport.requests
from google.oauth2 import service_account
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Streamlit URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Secrets (set these as Railway environment variables) ──────────────────────
DEEPGRAM_KEY       = os.environ["DEEPGRAM_API_KEY"]
GEMINI_KEY         = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

GCP_SA_JSON = {
    "type":         "service_account",
    "project_id":   os.environ["GCP_PROJECT_ID"],
    "private_key":  os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n"),
    "client_email": os.environ["GCP_CLIENT_EMAIL"],
    "token_uri":    "https://oauth2.googleapis.com/token",
}

def get_tts_token() -> str:
    creds = service_account.Credentials.from_service_account_info(
        GCP_SA_JSON,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


# ── /llm — Gemini ─────────────────────────────────────────────────────────────
@app.post("/llm")
async def llm_proxy(request: Request):
    body = await request.json()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── /tts — GCP Text-to-Speech ─────────────────────────────────────────────────
@app.post("/tts")
async def tts_proxy(request: Request):
    body = await request.json()
    token = get_tts_token()
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://texttospeech.googleapis.com/v1/text:synthesize",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── /stt-token — Deepgram short-lived token ───────────────────────────────────
@app.get("/stt-token")
async def stt_token():
    return JSONResponse({"key": DEEPGRAM_KEY})


# ── /telegram — send a message ────────────────────────────────────────────────
@app.post("/telegram")
async def telegram_proxy(request: Request):
    body = await request.json()
    chat_id = body.get("chat_id")
    text    = body.get("text")
    if not chat_id or not text:
        raise HTTPException(status_code=400, detail="chat_id and text required")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/health")
def health():
    return {"status": "ok"}
