import os, json, uuid, time, shutil, subprocess, threading
from pathlib import Path
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

KEY = os.getenv("GROQ_API_KEY", "")
STT = os.getenv("STT_MODEL", "whisper-large-v3")
LLM = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
TTL = 3600
ROOT = Path("/tmp/capx"); ROOT.mkdir(parents=True, exist_ok=True)
API = "https://api.groq.com/openai/v1"
# add new languages here later
LANGS = {"en": ("English", "Noto Sans"), "hi": ("Hindi", "Noto Sans Devanagari"), "ur": ("Urdu", "Noto Naskh Arabic")}
JOBS = {}
app = FastAPI()

def upd(i, **k): JOBS[i].update(k)

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(p.stderr[-400:])
    return p.stdout

def transcribe(audio):
    with open(audio, "rb") as f:
        r = httpx.post(f"{API}/audio/transcriptions", headers={"Authorization": f"Bearer {KEY}"},
            data={"model": STT, "response_format": "verbose_json", "temperature": "0"},
            files={"file": ("a.mp3", f, "audio/mpeg")}, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d.get("language", "english").lower(), [s for s in d["segments"] if s["text"].strip()]

def translate(texts, src, tgt):
    out = []
    for n in range(0, len(texts), 30):
        batch = texts[n:n+30]
        sys = (f"You are a professional subtitle translator. Convert each line from {src} to {tgt}, keeping meaning and tone. "
               "Hindi must be in Devanagari script, Urdu in Urdu script. Return JSON: {\"lines\": [...]} with exactly the same number of items.")
        for _ in range(2):
            r = httpx.post(f"{API}/chat/completions", headers={"Authorization": f"Bearer {KEY}"}, timeout=300,
                json={"model": LLM, "temperature": 0, "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}]})
            r.raise_for_status()
            lines = json.loads(r.json()["choices"][0]["message"]["content"]).get("lines", [])
            if len(lines) == len(batch): break
        else:
            raise RuntimeError("Translation mismatch, please retry")
        out += [str(x) for x in lines]
    return out

def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def make_ass(path, segs, texts, w, h, font, bold):
    size = int(min(w, h) / 17)
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"Style: D,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,{bold},0,0,0,100,100,0,0,1,{max(2,size//12)},0,2,{int(w*0.06)},{int(w*0.06)},{int(h*0.08)},1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    ev = ""
    for s, t in zip(segs, texts):
        t = t.strip().replace("{", "").replace("}", "").replace("\n", "\\N")
        ev += f"Dialogue: 0,{ts(s['start'])},{ts(s['end'])},D,,0,0,0,,{t}\n"
    Path(path).write_text(head + ev, encoding="utf-8")

def process(i, lang):
    d = ROOT / i
    try:
        upd(i, status="Audio nikal rahe hain...", progress=10)
        info = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(d / "in")]))
        st = info["streams"][0]; w, h = st["width"], st["height"]
        run(["ffmpeg", "-y", "-i", "in", "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", "a.mp3"], cwd=d)
        upd(i, status="AI speech recognition...", progress=30)
        src, segs = transcribe(d / "a.mp3")
        if not segs: raise RuntimeError("Video me koi awaaz/speech nahi mili")
        texts = [s["text"].strip() for s in segs]
        tgt = LANGS[lang][0]
        if src != tgt.lower():
            upd(i, status=f"{src.title()} se {tgt} translate ho raha hai...", progress=55)
            texts = translate(texts, src, tgt)
        upd(i, status="Captions video me lag rahe hain...", progress=70)
        make_ass(d / "s.ass", segs, texts, w, h, LANGS[lang][1], 1 if lang == "en" else 0)
        run(["ffmpeg", "-y", "-i", "in", "-vf", "ass=s.ass", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "out.mp4"], cwd=d)
        upd(i, status="Ready", progress=100, done=True)
    except httpx.HTTPStatusError as e:
        upd(i, error=f"AI service error ({e.response.status_code}). API key ya limit check karo.")
    except Exception as e:
        upd(i, error=str(e)[:300])

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), lang: str = Form("en")):
    if lang not in LANGS: raise HTTPException(400, "Invalid language")
    if not KEY: raise HTTPException(500, "GROQ_API_KEY set nahi hai")
    i = uuid.uuid4().hex; d = ROOT / i; d.mkdir()
    size = 0
    with open(d / "in", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20:
                f.close(); shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"File {MAX_MB}MB se badi hai")
            f.write(chunk)
    JOBS[i] = {"status": "Queued", "progress": 5, "done": False, "error": None, "t": time.time()}
    threading.Thread(target=process, args=(i, lang), daemon=True).start()
    return {"id": i}

@app.get("/api/status/{i}")
def status(i: str):
    if i not in JOBS: raise HTTPException(404, "Job nahi mila")
    return JOBS[i]

@app.get("/api/video/{i}")
def video(i: str, dl: int = 0):
    p = ROOT / i / "out.mp4"
    if i not in JOBS or not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4", filename="capx-captioned.mp4" if dl else None,
                        content_disposition_type="attachment" if dl else "inline")

def cleaner():
    while True:
        time.sleep(300)
        for i in list(JOBS):
            if time.time() - JOBS[i]["t"] > TTL:
                shutil.rmtree(ROOT / i, ignore_errors=True); JOBS.pop(i, None)

@app.on_event("startup")
def _s(): threading.Thread(target=cleaner, daemon=True).start()

app.mount("/", StaticFiles(directory="static", html=True), name="static")
