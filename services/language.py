"""Translation and voice output. IndicTrans2 is optional due to heavy setup.
The fallback uses Google Translate's free public endpoint through gTTS only for speech;
translation is deliberately a transparent pass-through until IndicTrans2 is configured.
"""
from pathlib import Path
from gtts import gTTS

def translate(text, source_lang="auto", target_lang="en"):
    """Integration point for IndicTrans2. Returns unchanged text if model is not installed."""
    return {"text": text, "source_lang": source_lang, "target_lang": target_lang,
            "provider": "passthrough", "note": "Install IndicTrans2 and replace this function for offline translation."}

def get_hindi_explanation(verdict: str, score: float = None) -> str:
    """
    Verdict ke hisaab se Hindi mein explanation text return karta hai.
    Hardcoded templates use karte hain (live translation ki zaroorat nahi).
    """
    if verdict == "misleading":
        return (
            "सावधान। इस संदेश में भ्रामक जानकारी होने की संभावना अधिक है। "
            "कृपया इसे साझा करने से पहले विश्वसनीय स्रोत से जाँच करें।"
        )
    else:
        return (
            "इस संदेश में भ्रामक जानकारी का जोखिम कम है। "
            "फिर भी विश्वसनीय स्रोत से सत्यापन करें।"
        )

def text_to_speech(text, language, output_path):
    """Create an MP3 using gTTS; language must be a gTTS-supported ISO code."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=text, lang=language if language != "auto" else "hi").save(str(output_path))
    return str(output_path)
