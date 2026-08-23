"""HTTP API: run with `uvicorn api.main:app --reload`."""
from pathlib import Path
import shutil, uuid
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from config import ARTIFACT_DIR, MESONET_WEIGHTS
from models.video_detector import load_detector, classify_face
from models.text_detector import ClaimDetector
from services.video import extract_faces
from services.gradcam_service import save_gradcam
from services.speech import transcribe
from services.language import translate, text_to_speech
from services.language import translate, get_hindi_explanation

app = FastAPI(title="Regional Deepfake & Misinformation Prototype")
app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR), name="artifacts")
video_model, device, trained = load_detector(MESONET_WEIGHTS)
text_detector = None

def save_upload(upload):
    """Save an uploaded file under a unique artifact folder."""
    job = ARTIFACT_DIR / str(uuid.uuid4()); job.mkdir()
    path = job / upload.filename
    with path.open("wb") as out: shutil.copyfileobj(upload.file, out)
    return job, path

@app.get("/health")
def health(): return {"status": "ok", "mesonet_weights_loaded": trained}

@app.post("/text")
def analyze_text(text: str = Form(...), source_lang: str = Form("auto"), output_lang: str = Form("hi")):
    """Highlight sentence-level claims. Loads a large Transformer only on first request."""
    global text_detector
    try:
        if text_detector is None: text_detector = ClaimDetector()
        result = text_detector.analyze(text)
        result["translated_explanation"] = get_hindi_explanation(result["verdict"])
        return result
      
    except Exception as exc: raise HTTPException(503, f"Text model unavailable: {exc}")

@app.post("/audio")
def analyze_audio(file: UploadFile = File(...), output_lang: str = Form("hi")):
    """Transcribe audio; send transcript through the text pipeline in the combined endpoint."""
    job, path = save_upload(file); result = transcribe(str(path))
    return {**result, "translation": translate(result["text"], result["language"], output_lang)}

@app.post("/video")
def analyze_video(file: UploadFile = File(...)):
    """Sample faces, classify the first face, and return a Grad-CAM overlay URL."""
    job, path = save_upload(file); faces = extract_faces(str(path), str(job / "faces"))
    if not faces: raise HTTPException(422, "No face detected. Try a clearer, front-facing video.")
    verdict = classify_face(cv2.imread(faces[0]), video_model, device)
    heatmap = save_gradcam(faces[0], job / "gradcam.jpg", video_model, device)
    return {**verdict, "faces_examined": len(faces), "heatmap_url": f"/artifacts/{job.name}/gradcam.jpg",
            "warning": "Prediction is demonstrative until trained MesoNet weights are installed." if not trained else None}

@app.post("/pipeline")
def full_pipeline(content_type: str = Form(...), text: str = Form(""), file: UploadFile | None = File(None), output_lang: str = Form("hi")):
    """One endpoint for UI: routes text, audio, or video through the applicable stages."""
    if content_type == "text": return analyze_text(text, output_lang=output_lang)
    if file is None: raise HTTPException(422, "Please upload a file.")
    if content_type == "audio":
        audio = analyze_audio(file, output_lang); return analyze_text(audio["text"], output_lang=output_lang) | {"transcript": audio["text"]}
    if content_type == "video": return analyze_video(file)
    raise HTTPException(422, "content_type must be text, audio, or video")

@app.post("/tts")
def tts(text: str = Form(...), language: str = Form("hi")):
    """Return a browser-accessible URL for the spoken explanation."""
    job = ARTIFACT_DIR / str(uuid.uuid4()); path = text_to_speech(text, language, job / "verdict.mp3")
    return {"audio_url": f"/artifacts/{job.name}/{Path(path).name}"}
