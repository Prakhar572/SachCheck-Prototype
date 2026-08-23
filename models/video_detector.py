"""Small MesoNet-style CNN and a safe inference wrapper.

This architecture is usable immediately but its random weights are NOT a real detector.
Download/fine-tune weights and place them at checkpoints/mesonet_weights.pt.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

class MesoNet(nn.Module):
    """Compact CNN inspired by MesoNet; accepts 256x256 RGB face crops."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(16 * 8 * 8, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.classifier(self.features(x))

def load_detector(weights_path: Path | None = None):
    """Load model and optional state dictionary. Class 0=real, 1=fake."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MesoNet().to(device).eval()
    trained = False
    if weights_path and weights_path.exists():
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state.get("state_dict", state))
        trained = True
    return model, device, trained

def image_to_tensor(image_bgr: np.ndarray, device: str):
    """Resize an OpenCV BGR image and create a PyTorch batch tensor."""
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

def classify_face(image_bgr, model, device):
    """Return label and softmax confidence for one cropped face."""
    with torch.no_grad():
        probs = torch.softmax(model(image_to_tensor(image_bgr, device)), dim=1)[0]
    index = int(probs.argmax())
    return {"label": ["real", "fake"][index], "confidence": round(float(probs[index]), 4)}
