"""Speech-to-text with local OpenAI Whisper."""
import whisper
from config import WHISPER_MODEL
_model = None
def transcribe(audio_path, language=None):
    """Download Whisper on first use and return transcript plus detected language."""
    global _model
    if _model is None: _model = whisper.load_model(WHISPER_MODEL)
    result = _model.transcribe(audio_path, language=language, fp16=False)
    return {"text": result["text"].strip(), "language": result.get("language", "unknown")}
