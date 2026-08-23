# SachCheck 🛡️

A prototype that detects misinformation and manipulation in Hindi text posts, voice messages, and short face videos. Built for Smart India Hackathon 2026.

It does not just say "fake" or "real" — it gives a **risk score plus an explanation** (which sentence looks suspicious, or which part of a face looks manipulated).

> ⚠️ This is a hackathon prototype, not a production-ready system.

---

## How to Use This README

If you are setting up this project on a new laptop for the first time follow the steps **in order, from top to bottom**. Do not skip a step, even if it looks familiar.

---

## Table of Contents

1. [What You Need to Install First](#1-what-you-need-to-install-first)
2. [Cloning the Project](#2-cloning-the-project)
3. [Creating the Virtual Environment](#3-creating-the-virtual-environment)
4. [Installing Requirements](#4-installing-requirements)
5. [Installing FFmpeg](#5-installing-ffmpeg)
6. [Files Not Included on GitHub](#6-files-not-included-on-github)
7. [Text Model Training (MuRIL) — Full Process](#7-text-model-training-muril--full-process)
8. [Audio Module Setup](#8-audio-module-setup)
9. [Video Model Training (MesoNet CNN) — Full Process](#9-video-model-training-mesonet-cnn--full-process)
10. [Running the App](#10-running-the-app)
11. [Testing Everything](#11-testing-everything)
12. [Common Errors and Fixes](#12-common-errors-and-fixes)

---

## 1. What You Need to Install First

Make sure these are installed on your computer before starting:

- **Python 3.10 or newer** — download from [python.org](https://www.python.org/downloads/). During installation, make sure to check the box **"Add Python to PATH"**.
- **VS Code** — download from [code.visualstudio.com](https://code.visualstudio.com/).
- **Python extension for VS Code** — open VS Code → click the Extensions icon on the left → search "Python" → install the one made by Microsoft.
- **Git** — download from [git-scm.com](https://git-scm.com/downloads) (needed to get the code from GitHub).
- **A Google account** — needed to use Google Colab (free GPU) for training.

Once installed, open a terminal in VS Code (Terminal → New Terminal) and run:
```powershell
python --version
git --version
```
Both should print a version number. If you see a "not recognized" error, the installation did not work correctly — reinstall and restart your computer.

---

## 2. Cloning the Project

Choose a location for the project (for example, `C:\`), then in the terminal:

```powershell
cd C:\
git clone https://github.com/<username>/SachCheck-Prototype.git
cd SachCheck-Prototype
```

Then open this folder in VS Code: **File → Open Folder → select SachCheck-Prototype**.

---

## 3. Creating the Virtual Environment

A virtual environment is a separate, clean Python setup just for this project, so it doesn't clash with anything else on your computer.

Open a terminal inside the project folder, then run:

```powershell
python -m venv .venv
```

Now **activate** it:

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

If you get an error ("running scripts is disabled"), run this first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
then try the activate command again.

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**Sign that it worked:** the terminal prompt will now start with `(.venv)`, like this:
```
(.venv) PS C:\SachCheck-Prototype>
```

⚠️ **You need to run this activation command every time you open a new terminal.**

---

## 4. Installing Requirements

Once the virtual environment is activated (you see `(.venv)` at the start of the prompt), run:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs all the required libraries (PyTorch, Transformers, FastAPI, Streamlit, and others). It can take **5–10 minutes** depending on your internet speed — do not close the terminal until it finishes.

**Confirm everything installed correctly:**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
Seeing `CUDA available: False` is normal (your laptop has no GPU, which is fine).

---

## 5. Installing FFmpeg

This is required by the audio module (Whisper).

**Windows:**
```powershell
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

After installing, **close VS Code completely and reopen it**, open a new terminal, activate the virtual environment, and confirm:
```powershell
ffmpeg -version
```
You should see version information, with no errors.

---

## 6. Files Not Included on GitHub

These files/folders are **not included in this repository** — you need to create or generate them yourself (full instructions in the sections below):

| File/Folder | What to do |
|---|---|
| `checkpoints/muril_hindi_fake_news/` | Follow [Section 7](#7-text-model-training-muril--full-process) to train it yourself |
| `checkpoints/mesonet_weights.pt` | Follow [Section 9](#9-video-model-training-mesonet-cnn--full-process) to train it yourself |
| `data/raw/`, `data/processed/` | Instructions are in [Section 7](#7-text-model-training-muril--full-process) |
| `demo_media/*.wav`, `*.mp4`, `*.mp3` | Record/generate your own ([Section 8](#8-audio-module-setup), [Section 9](#9-video-model-training-mesonet-cnn--full-process)) |
| `.venv/` | Create it yourself by following Section 3 |

The app will still start without these files, but it won't give meaningful results — so make sure to follow the next sections.

---

## 7. Text Model Training (MuRIL) — Full Process

### Step 7.1: Download a dataset

Download any Hindi fake-news dataset that has `text` and `label` columns (0 = not misleading, 1 = misleading). Place it here:
``'
data/raw/dataset-merged.csv**
```
(Create the `data` and `raw` folders first if they don't exist — right-click in the VS Code file explorer → New Folder)

### Step 7.2: Inspect and clean the dataset

Create a new file: `data/raw/inspect_dataset.py`
```python
import pandas as pd

df = pd.read_csv("data/raw/dataset-merged.csv")
print("Total rows:", len(df))
print(df['label'].value_counts())
print("Missing text:", df['text'].isna().sum())
print("Missing label:", df['label'].isna().sum())
```
Run it:
```powershell
python data/raw/inspect_dataset.py
```
⚠️ **Do not name this file `inspect.py`** — it clashes with a built-in Python library of the same name and causes a crash. Keep it as `inspect_dataset.py`.

### Step 7.3: Create the train/validation/test split

Create a new file: `data/raw/split_dataset.py`
```write the code given below (python)

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/dataset-merged.csv")
df = df[['text', 'label']].dropna()
df['label'] = df['label'].astype(int)

if len(df) > 8000:
    df = df.sample(n=8000, random_state=42)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
```
First create the `data/processed` folder if it doesn't exist, then run:
```powershell
python data/raw/split_dataset.py
```

### Step 7.4: Open Google Colab and turn on the GPU

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account
2. Create a **New Notebook**
3. Go to **Runtime → Change runtime type → select T4 GPU** → Save
4. In a new cell, run:
   ```python
   !nvidia-smi
   ```

### Step 7.5: Connect Google Drive and upload files

In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

In Google Drive (open `drive.google.com` in your browser), create this folder:
```
MyDrive/sachcheck/trained_models/
```

From your laptop, upload `data/processed/train.csv`, `validation.csv`, and `test.csv` into the `sachcheck` folder on Drive.

### Step 7.6: Train the model

In Colab, run these cells one at a time:

```python
!pip install -q transformers datasets scikit-learn evaluate accelerate
```

```python
import pandas as pd
from datasets import Dataset

train_df = pd.read_csv('/content/drive/MyDrive/sachcheck/train.csv')
val_df = pd.read_csv('/content/drive/MyDrive/sachcheck/validation.csv')

train_ds = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
```

```python
from transformers import AutoTokenizer

model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

train_ds = train_ds.map(tokenize_fn, batched=True).rename_column("label", "labels")
val_ds = val_ds.map(tokenize_fn, batched=True).rename_column("label", "labels")
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

```python
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="/content/muril_output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                   eval_dataset=val_ds, compute_metrics=compute_metrics)
```

```python
trainer.train()
```

⚠️ **If you get `ImportError: cannot import name 'VideoReader' from torchvision.io'`:** run `!pip uninstall -y torchvision` in a new cell, then go to **Runtime → Restart session**, then re-run all the cells from the top.

### Step 7.7: Save and download the model

```python
save_path = "/content/drive/MyDrive/sachcheck/trained_models/muril_hindi_fake_news"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
```

In Google Drive, right-click that folder → **Download**. Extract the zip, and copy the inner `muril_hindi_fake_news` folder into your project at:
```
checkpoints/muril_hindi_fake_news/
```

### Step 7.8: Update config.py

In `config.py`, find:
```python
TEXT_MODEL_ID = "google/muril-base-cased"
```
Change it to:
```python
TEXT_MODEL_ID = str(BASE_DIR / "checkpoints" / "muril_hindi_fake_news")
```

**Text model training complete.**

---

## 8. Audio Module Setup

The audio module (Whisper) needs **no training** — it's already pretrained, you just need the library installed (already done via `requirements.txt` in Section 4).

### Creating a sample audio file for testing

Record a 5–12 second Hindi audio clip on your phone (in a quiet room, speaking clearly), transfer it to your laptop, and convert it to `.wav` format if needed:

```powershell
ffmpeg -i "your-file.m4a" -ar 16000 -ac 1 demo_media/hindi_misleading_audio.wav
```

The first time Whisper runs, it will download its model (needs internet), which takes a little time.

---

## 9. Video Model Training (MesoNet CNN) — Full Process

### Step 9.1: Download a real-vs-fake face dataset

Search Kaggle for something like "Real and Fake Face Detection" or "140k Real and Fake Faces" and download it. You can download it directly inside Colab (via the Kaggle API) or download it to your laptop and upload it to Drive.

### Step 9.2: Set up the training environment in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!pip install -q torch torchvision opencv-python-headless
```

### Step 9.3: Define the model architecture

This must be the **exact same class** as the one in `models/video_detector.py` (otherwise the trained weights won't load):

```python
import torch
from torch import nn

class MesoNet(nn.Module):
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
```

### Step 9.4: Load the dataset and train

```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Dataset folder structure should be: dataset/real/*.jpg and dataset/fake/*.jpg
transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
dataset = ImageFolder("/content/drive/MyDrive/sachcheck/face_dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MesoNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
```

⚠️ **Important:** `ImageFolder` assigns class numbers alphabetically (so "fake" might become 0 and "real" might become 1). The comment in `models/video_detector.py` says "Class 0=real, 1=fake". If your dataset's order is reversed, print `dataset.class_to_idx` to check, and swap labels if needed.

### Step 9.5: Save and download the model

```python
torch.save({"state_dict": model.state_dict()}, "/content/drive/MyDrive/sachcheck/trained_models/mesonet_weights.pt")
```

Download it from Drive and place it in your project at:
```
checkpoints/mesonet_weights.pt
```

### Step 9.6: Confirm it loads correctly

```powershell
python -c "from models.video_detector import load_detector; from config import MESONET_WEIGHTS; model, device, trained = load_detector(MESONET_WEIGHTS); print(device, trained)"
```
Expected output: `cpu True`

### Step 9.7: Create test videos

```
demo_media/real_face.mp4          (your own 5–10 sec face video)
demo_media/manipulated_face.mp4   (a manipulated/fake sample)
demo_media/no_face.mp4            (a video with no face in it)
```

**Video model training complete.**

---

## 10. Running the App

**You need two terminals, both with the virtual environment activated.**

**Terminal 1 (Backend):**
```powershell
.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```
Open in your browser: `http://127.0.0.1:8000/docs`

**Terminal 2 (Frontend):**
```powershell
.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```
Open in your browser: `http://localhost:8501`

---

## 11. Testing Everything

In Swagger (`127.0.0.1:8000/docs`), test each endpoint:

| Endpoint | What to test |
|---|---|
| `POST /text` | Paste Hindi text, check the verdict + score |
| `POST /audio` | Upload a `.wav` file from `demo_media`, check the transcript |
| `POST /video` | Upload an `.mp4` file from `demo_media`, check the score + heatmap |
| `GET /health` | Simple check that the server is running |

Then test the same things through the Streamlit UI (`localhost:8501`).

---

## 12. Common Errors and Fixes

| Error | Fix |
|---|---|
| `running scripts is disabled` (PowerShell) | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then try activating again |
| `cannot import name 'VideoReader' from torchvision.io` (Colab) | Run `!pip uninstall -y torchvision` → Restart the runtime → re-run all cells |
| Circular import error when running `inspect.py` | Rename the file to `inspect_dataset.py` |
| `uvicorn: command not found` or import errors | Make sure `(.venv)` is active in the terminal, then run `pip install -r requirements.txt` again |
| Checkpoint loading shows `False` | Check the filename is exactly correct and it's directly inside `checkpoints/` (not in a subfolder) |
| Whisper writes Hindi in Roman script instead of Devanagari | This is a known limitation; `language="auto"` is intentional — outside the scope of this guide to fix |
| `Error loading ASGI app` when running the uvicorn command | The command must be exactly: `uvicorn api.main:app --reload` (colon `:` in the right place, no extra colon) |

For any error not listed here, copy the exact error message and send it to the project maintainer.

---

## Disclaimer

SachCheck is an early-warning verification aid. It does not replace professional fact-checking, legal evidence, or journalistic verification.