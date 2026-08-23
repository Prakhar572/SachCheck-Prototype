"""Central configuration. Change model IDs or paths here, not throughout the app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"  # Modules automatically fall back to CPU when CUDA is absent.
TEXT_MODEL_ID = str(BASE_DIR / "checkpoints" / "muril_hindi_fake_news") #updated to use local checkpoint instead of huggingface model
WHISPER_MODEL = "base"  # tiny is faster on CPU; base is a good Colab demo choice.
MESONET_WEIGHTS = BASE_DIR / "checkpoints" / "mesonet_weights.pt"
