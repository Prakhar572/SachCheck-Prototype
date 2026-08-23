"""Claim-level text classifier using a Hugging Face sequence-classification checkpoint.
MuRIL base is not itself fine-tuned for misinformation, so demo output is marked untrained
until you replace the model ID with a fine-tuned checkpoint.
"""
import re
from transformers import pipeline
from config import TEXT_MODEL_ID

def split_sentences(text: str):
    """Simple Unicode-friendly split suitable for a first prototype."""
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]

class ClaimDetector:
    def __init__(self, model_id=TEXT_MODEL_ID):
        # This needs a checkpoint trained with labels REAL/FAKE (or 0/1).
        self.pipe = pipeline("text-classification", model=model_id, tokenizer=model_id, top_k=None)

    def analyze(self, text):
        claims = []
        for sentence in split_sentences(text):
            scores = self.pipe(sentence)[0]
            top = max(scores, key=lambda item: item["score"])
            misleading = top["label"].upper() in {"LABEL_1", "FAKE", "MISLEADING"}
            claims.append({"sentence": sentence, "label": "misleading" if misleading else "not_flagged", "score": round(float(top["score"]), 4)})
        flagged = [c for c in claims if c["label"] == "misleading"]
        return {"verdict": "misleading" if flagged else "not_flagged", "claims": claims}