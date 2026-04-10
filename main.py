"""
Lex Proxy — Railway deployment
Keeps all API keys server-side.
Endpoints:
  POST /llm        — Gemini LLM
  POST /tts        — Gemini TTS with sentence-level parallelism
  GET  /stt-token  — Deepgram short-lived token
  POST /telegram   — Telegram sendMessage
"""
import os
import re
import struct
import base64 as b64mod
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

VOICE_MAP = {"male": "Charon", "female": "Kore"}

def pcm_to_wav(raw: bytes) -> bytes:
    sr, ch, bits = 24000, 1, 16
    byte_rate    = sr * ch * bits // 8
    block_align  = ch * bits // 8
    data_size    = len(raw)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1,
        ch, sr, byte_rate, block_align, bits,
        b'data', data_size
    )
    return header + raw

def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

async def tts_one(client, sentence: str, voice_name: str):
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-preview-tts:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": sentence}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            }
        }
    }
    try:
        resp = await client.post(url, json=body, timeout=20)
        if resp.status_code != 200:
            return None
        data      = resp.json()
        part      = data["candidates"][0]["content"]["parts"][0]["inlineData"]
        audio_b64 = part["data"]
        mime      = part.get("mimeType", "audio/wav")
        if "pcm" in mime:
            audio_b64 = b64mod.b64encode(pcm_to_wav(b64mod.b64decode(audio_b64))).decode()
            mime      = "audio/wav"
        return {"audioContent": audio_b64, "mimeType": mime}
    except Exception:
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

    voice_raw  = body.get("voice", "male")
    if isinstance(voice_raw, dict):
        name   = voice_raw.get("name", "")
        gender = "female" if "-F" in name or "female" in name.lower() else "male"
    else:
        gender = "female" if voice_raw == "female" else "male"

    voice_name = VOICE_MAP[gender]
    sentences  = split_sentences(text)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[tts_one(client, s, voice_name) for s in sentences])

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
    return {"status": "ok"}
