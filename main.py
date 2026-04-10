"""
Lex Proxy — Railway deployment
Keeps all API keys server-side.
Endpoints:
  POST /llm        — Gemini LLM
  POST /tts        — Deepgram TTS with caching
  GET  /stt-token  — Deepgram short-lived token
  POST /telegram   — Telegram sendMessage
"""
import os
import re
import struct
import base64 as b64mod
import asyncio
import hashlib
import httpx
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPGRAM_KEY       = os.environ["DEEPGRAM_API_KEY"]
GEMINI_KEY         = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

VOICE_MAP = {
    "male":   "aura-2-theron-en",
    "female": "aura-2-luna-en",
}

# ── In-memory TTS cache ───────────────────────────────────────────────────────
_tts_cache: dict = {}
MAX_CACHE_SIZE = 500

def _cache_key(sentence: str, voice: str) -> str:
    return hashlib.md5(f"{voice}:{sentence.strip().lower()}".encode()).hexdigest()

def _cache_get(sentence: str, voice: str):
    return _tts_cache.get(_cache_key(sentence, voice))

def _cache_set(sentence: str, voice: str, value: dict):
    if len(_tts_cache) >= MAX_CACHE_SIZE:
        for k in list(_tts_cache.keys())[:100]:
            del _tts_cache[k]
    _tts_cache[_cache_key(sentence, voice)] = value

# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

async def tts_one(client, sentence: str, voice: str):
    # Check cache first
    cached = _cache_get(sentence, voice)
    if cached:
        logger.info(f"TTS cache hit: '{sentence[:40]}'")
        return cached

    url = f"https://api.deepgram.com/v1/speak?model={voice}&encoding=mp3"
    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "application/json",
    }
    body = {"text": sentence}

    try:
        resp = await client.post(url, json=body, headers=headers, timeout=20)
        if resp.status_code != 200:
            logger.error(f"Deepgram TTS error {resp.status_code}: {resp.text}")
            return None
        audio_b64 = b64mod.b64encode(resp.content).decode()
        result = {"audioContent": audio_b64, "mimeType": "audio/mp3"}
        _cache_set(sentence, voice, result)
        return result
    except Exception as e:
        logger.error(f"TTS exception: {e}")
        return None

@app.post("/llm")
async def llm_proxy(request: Request):
    body = await request.json()
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

@app.post("/tts")
async def tts_proxy(request: Request):
    body = await request.json()
    text = body.get("text") or body.get("input", {}).get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    voice_raw = body.get("voice", "male")
    if isinstance(voice_raw, dict):
        name   = voice_raw.get("name", "")
        gender = "female" if "-F" in name or "female" in name.lower() else "male"
    else:
        gender = "female" if voice_raw == "female" else "male"

    voice     = VOICE_MAP[gender]
    sentences = split_sentences(text)
    logger.info(f"TTS request: {len(sentences)} sentences, voice={voice}, cache_size={len(_tts_cache)}")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[tts_one(client, s, voice) for s in sentences])

    chunks = [r for r in results if r is not None]
    if not chunks:
        raise HTTPException(status_code=500, detail="TTS failed for all sentences")

    if len(chunks) == 1:
        return JSONResponse(chunks[0])

    return JSONResponse({"chunks": chunks})

@app.get("/stt-token")
async def stt_token():
    return JSONResponse({"key": DEEPGRAM_KEY})

@app.post("/telegram")
async def telegram_proxy(request: Request):
    body    = await request.json()
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
    return {"status": "ok", "cache_size": len(_tts_cache)}
