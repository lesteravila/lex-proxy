"""
Lex Proxy — Railway deployment
Keeps all API keys server-side.
Endpoints:
  POST /llm        — Gemini
  POST /tts        — Gemini TTS (replaces GCP TTS)
  GET  /stt-token  — Deepgram short-lived token
  POST /telegram   — Telegram sendMessage
"""
import os
import base64
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# ── Voice map — matches isFemale toggle in frontend ───────────────────────────
# Puck  = playful, upbeat, great for kids
# Kore  = soft, gentle, warm — good for female voice
VOICE_MAP = {
    "male":   "Puck",
    "female": "Kore",
}

# ── /llm — Gemini ─────────────────────────────────────────────────────────────
@app.post("/llm")
async def llm_proxy(request: Request):
    body = await request.json()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

# ── /tts — Gemini Text-to-Speech ──────────────────────────────────────────────
# Replaces GCP TTS. Uses Gemini's REST TTS API — no websocket, no regional issues.
# Frontend sends: { text: "...", voice: "male" | "female" }
# Returns: { audioContent: "<base64 mp3>" }  (same shape as before — no frontend changes needed)
@app.post("/tts")
async def tts_proxy(request: Request):
    body = await request.json()

    # Support both old GCP format and new simple format
    text = (
        body.get("text")                          # simple: { text: "..." }
        or body.get("input", {}).get("text", "")  # old GCP: { input: { text: "..." } }
    )
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Determine voice — support both old GCP voice name and new simple gender string
    voice_raw = body.get("voice", "male")
    if isinstance(voice_raw, dict):
        # Old GCP format: { voice: { name: "en-US-Neural2-F" } }
        name = voice_raw.get("name", "")
        gender = "female" if "-F" in name or "female" in name.lower() else "male"
    else:
        # New simple format: { voice: "male" } or { voice: "female" }
        gender = "female" if voice_raw == "female" else "male"

    voice_name = VOICE_MAP[gender]

    # Gemini TTS REST API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_KEY}"

    gemini_body = {
        "contents": [
            {
                "parts": [{"text": text}]
            }
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice_name
                    }
                }
            }
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=gemini_body)

    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    data = resp.json()

    # Extract audio bytes from Gemini response
    try:
        audio_data = (
            data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        )
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="No audio in Gemini TTS response")

    # Return in same shape as old GCP TTS so frontend needs no changes
    return JSONResponse({"audioContent": audio_data})

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
