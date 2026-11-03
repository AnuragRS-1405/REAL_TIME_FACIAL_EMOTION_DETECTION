# Real-Time Facial Emotion Recognition
### Deep Learning System Powered by ANN & CNN with FastAPI Backend

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![CNN Accuracy](https://img.shields.io/badge/CNN%20Accuracy-91.66%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 📋 What This Project Is All About

This project is an **end-to-end Affective Computing & Deep Learning system** designed to detect and classify human facial expressions in real time from static images, uploaded videos, and live webcam feeds.

The system detects **7 canonical human emotions**:

| Emotion | Label | Visual Features & Morphological Markers |
|:---|:---:|:---|
| 😠 **Angry** | `angry` | Knitted eyebrows, lowered brow, intense stare, tightened lips |
| 🤢 **Disgust** | `disgust` | Wrinkled nose bridge, raised upper lip, narrowed eyes |
| 😨 **Fear** | `fear` | Widened eye aperture, raised inner eyebrows, slightly parted mouth |
| 😊 **Happy** | `happy` | Elevated cheek zygomatic muscles, broad smile, crinkled eye corners |
| 😐 **Neutral** | `neutral` | Relaxed facial musculature, horizontal lip line, resting gaze |
| 😢 **Sad** | `sad` | Downturned mouth corners, inner eyebrows pulled upwards |
| 😲 **Surprise** | `surprise` | Vertically dropped jaw, wide circular eye aperture, arched brows |

### 🔬 Core Scientific & Engineering Highlights

1. **Dual-Model Architectural Comparison:**
   - **Baseline ANN (Multi-Layer Perceptron):** A dense neural network ($4,096 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 7$) that benchmarks performance when 2D spatial locality is discarded. Reaches **46.33% accuracy**.
   - **Primary CNN (Deep Convolutional Network):** A 4-stage hierarchical convolutional network with Batch Normalization, Dropout, and **Global Average Pooling (GAP)**. Preserves 2D spatial topology and achieves **91.66% accuracy** and a **0.9166 F1-score** on 9,956 held-out test samples.
2. **Real-Time Computer Vision Pipeline:**
   - Uses OpenCV's **Haar Feature-Based Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect and isolate facial regions in $\mathcal{O}(1)$ time via integral images.
   - Deterministic preprocessing converts faces to single-channel Grayscale at $64 \times 64$ resolution with $[0.0, 1.0]$ normalization.
   - Ultra-fast inference: **1.92 ms per frame** for the CNN (enabling **500+ FPS**).
3. **Calibrated Softmax Probabilities:**
   - Employs post-processing temperature scaling ($T = 0.40$) to prevent neural overconfidence and deliver calibrated probability distributions across all 7 classes.
4. **Production-Ready Full-Stack Serving:**
   - High-throughput asynchronous **FastAPI** backend with CORS middleware and auto-generated OpenAPI documentation.
   - Interactive, responsive **Dark Glassmorphism UI** with three dedicated viewports:
     - **Home / Prediction Studio:** Live image upload, webcam detection, model toggling (ANN vs. CNN), confidence scores, and probability meters.
     - **Visual Analytics:** Dataset class distribution, training curves, confusion matrices, and sample predictions.
     - **Model Evaluation:** Side-by-side performance comparison tables and detailed classification reports.
5. **Comprehensive Technical Report:**
   - Contains a complete 24-page publication-quality engineering report compiled to **`report.pdf`** and **`report.html`**.

---

## ⚡ Execution Commands & Running the Project

Follow these step-by-step instructions to run the application on your machine.

### Step 1: Open Your Terminal
Open PowerShell or Command Prompt (CMD) in the project root directory:
```bash
cd "c:\Users\Anurag Rajendra\OneDrive\Pictures\PROJECTS\REAL TIME FACIAL EMOTION RECOGNITION"
```

---

### Step 2: Activate the Virtual Environment

Your virtual environment is already installed in the `venv/` directory. Activate it based on your shell:

#### 🔹 Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
> **Tip for PowerShell Users:** If you see an error saying *`Execution of scripts is disabled on this system`*, run this command once in your PowerShell window:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

#### 🔹 Windows Command Prompt (CMD):
```cmd
venv\Scripts\activate.bat
```

#### 🔹 macOS / Linux (if applicable):
```bash
source venv/bin/activate
```

*(Once activated, you will see `(venv)` prefixed in your terminal prompt).*

---

### Step 3: Run the Web Server (Uvicorn / Localhost)

You have two simple ways to launch the server:

#### Option A: Direct Uvicorn Command (Recommended for Development)
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
- `--reload`: Automatically restarts the server whenever code changes are saved.
- `--host 127.0.0.1`: Binds the server to localhost.
- `--port 8000`: Runs on port 8000.

#### Option B: Run via Python Script
```bash
python app.py
```
*(This triggers the `uvicorn.run(...)` entrypoint defined at the bottom of `app.py`).*

#### Option C: One-Line Direct Execution (Without Manual Activation)
If you do not want to activate the venv first, run directly via the venv executable:
```powershell
.\venv\Scripts\uvicorn.exe app:app --reload --host 127.0.0.1 --port 8000
```

---

### Step 4: Access Localhost in Your Browser

Once the terminal outputs `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`:

| Interface | URL | Description |
|:---|:---|:---|
| 🏠 **Web Application (UI)** | **[http://localhost:8000](http://localhost:8000)** | Prediction Studio, Analytics & Evaluation dashboards |
| 📊 **Analytics Dashboard** | **[http://localhost:8000/analysis](http://localhost:8000/analysis)** | Training curves, dataset charts & confusion matrices |
| ⚖️ **Model Evaluation** | **[http://localhost:8000/evaluation](http://localhost:8000/evaluation)** | Quantitative benchmark metrics & comparisons |
| 📖 **Interactive Swagger Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Test the REST API endpoints directly in your browser |
| 📑 **ReDoc Documentation** | **[http://localhost:8000/redoc](http://localhost:8000/redoc)** | Clean, formal OpenAPI specification viewer |
| 🩺 **System Health Status** | **[http://localhost:8000/status](http://localhost:8000/status)** | Model load verification & status JSON |

---

## 🛠️ Other Useful Commands

### 1. Re-Train the Neural Network Models
To re-run the entire data cleaning, stratified train/validation/test split, and ANN + CNN training pipeline from scratch:
```bash
python train_models.py
```
- **Inputs:** Raw images inside `Image_data/` (~49.7K images)
- **Outputs:** Saves trained models to `models/` (`ann_model.keras`, `cnn_model.keras`) and visual charts to `outputs/`.
- **Duration:** ~4 to 10 minutes depending on hardware.

### 2. Generate the Technical Report (PDF & HTML)
To regenerate the 24-page comprehensive project report (`report.pdf` and `report.html`):
```bash
python build_full_report.py
```

### 3. Install / Verify Dependencies
If you need to install or update dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Benchmark Results: ANN vs. CNN

Both models were evaluated on the exact same untouched test split of **9,956 samples**:

| Evaluation Metric | Baseline ANN | Primary CNN | Performance Margin |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 46.33% | **91.66%** | **+45.33% (+97.8% relative)** |
| **Weighted Precision** | 0.4646 | **0.9168** | **+0.4522** |
| **Weighted Recall** | 0.4633 | **0.9166** | **+0.4533** |
| **Weighted F1-Score** | 0.4578 | **0.9166** | **+0.4588** |
| **Macro F1-Score** | 0.4329 | **0.9120** | **+0.4791** |
| **Training Time** | ~42 sec | 245.8 sec (~4.1 min) | CNN requires more training compute |
| **Inference Latency** | **0.17 ms** | 1.92 ms | Both easily satisfy real-time (>500 FPS) |
| **Total Parameters** | 2,266,247 | **423,495** | **CNN is 81.3% lighter in memory** |

### Per-Class Performance (CNN):
- **Happy:** Precision 96.1% \| Recall 96.8% \| **F1: 0.965**
- **Disgust:** Precision 94.2% \| Recall 93.1% \| **F1: 0.936**
- **Surprise:** Precision 93.5% \| Recall 92.1% \| **F1: 0.928**
- **Neutral:** Precision 87.5% \| Recall 91.5% \| **F1: 0.894**
- **Angry:** Precision 89.9% \| Recall 88.5% \| **F1: 0.892**
- **Fear:** Precision 90.4% \| Recall 87.2% \| **F1: 0.887**
- **Sad:** Precision 88.2% \| Recall 88.1% \| **F1: 0.882**

---

## 🏗️ Model Architectures

### ANN Baseline Architecture (2.26M Parameters)
```
Input (4,096)
  └── Dense(512, ReLU, L2=1e-4) ──> BatchNorm ──> Dropout(0.40)
        └── Dense(256, ReLU, L2=1e-4) ──> BatchNorm ──> Dropout(0.30)
              └── Dense(128, ReLU) ──> Dropout(0.20)
                    └── Dense(7, Softmax)
```

### CNN Primary Architecture (423K Parameters)
```
Input (64 × 64 × 1 Grayscale)
  ├── Data Augmentation (Flip, Rotation ±8°, Zoom ±10%, Translation ±5%)
  ├── Block 1: Conv2D(32, 3×3) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.25)
  ├── Block 2: Conv2D(64, 3×3) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.25)
  ├── Block 3: Conv2D(128, 3×3) ──> BatchNorm ──> ReLU ──> MaxPool(2×2) ──> Dropout(0.25)
  ├── Block 4: Conv2D(256, 3×3) ──> BatchNorm ──> ReLU ──> GlobalAveragePooling2D
  └── Dense Head:
        ├── Dense(256, ReLU, L2=1e-4) ──> Dropout(0.40)
        ├── Dense(128, ReLU) ──> Dropout(0.30)
        └── Dense(7, Softmax)
```

---

## 🗂️ Project Directory Structure

```
REAL TIME FACIAL EMOTION RECOGNITION/
├── Image_data/                     # Dataset (7 emotion class folders, ~49.8K images)
│   ├── angry/                      # 5,920 images
│   ├── disgust/                    # 5,920 images
│   ├── fear/                       # 5,920 images
│   ├── happy/                      # 11,398 images
│   ├── neutral/                    # 8,166 images
│   ├── sad/                        # 6,535 images
│   └── surprise/                   # 5,920 images
│
├── models/                         # Serialized model binaries & metadata
│   ├── ann_model.keras             # Trained ANN (native Keras format)
│   ├── cnn_model.keras             # Trained CNN (native Keras format)
│   ├── ann_model.pkl               # ANN metadata wrapper
│   ├── cnn_model.pkl               # CNN metadata wrapper
│   ├── preprocessing_config.pkl    # Preprocessing & label mapping config
│   ├── metrics.json                # Benchmark evaluation metrics
│   └── history.json                # Epoch-by-epoch loss & accuracy history
│
├── outputs/                        # High-resolution generated charts & UI screenshots
│   ├── class_distribution.png      # Dataset class balance visualization
│   ├── sample_images.png           # Sample inputs per emotion class
│   ├── ann_history.png             # ANN accuracy and loss curves
│   ├── cnn_history.png             # CNN accuracy and loss curves
│   ├── ann_confusion_matrix.png    # ANN confusion matrix
│   ├── cnn_confusion_matrix.png    # CNN confusion matrix
│   ├── ann_vs_cnn_comparison.png   # Performance metric bar chart
│   ├── training_time_comparison.png# Wall-clock training duration chart
│   ├── sample_predictions.png      # Visual sample predictions
│   ├── ui_screenshot_home.png      # Prediction studio UI capture
│   ├── ui_screenshot_analysis.png  # Analytics dashboard UI capture
│   └── ui_screenshot_evaluation.png# Model evaluation UI capture
│
├── templates/                      # HTML5 templates (Jinja2)
│   ├── index.html                  # Home & live prediction interface
│   ├── analysis.html               # Visual analytics dashboard
│   └── evaluation.html             # Quantitative model benchmarking
│
├── static/                         # Static web assets
│   ├── css/style.css               # Dark glassmorphism styling
│   └── js/script.js                # Frontend async prediction logic
│
├── venv/                           # Virtual environment
├── app.py                          # FastAPI backend application
├── train_models.py                 # Full dataset processing & model training pipeline
├── build_full_report.py            # Generates report.html and compiles report.pdf
├── report.pdf                      # Comprehensive 24-page project report (PDF)
├── report.html                     # Standalone HTML version of the report
├── facial_emotion.ipynb            # Jupyter notebook exploration
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete project documentation
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | Renders Home & Live Prediction Studio |
| `GET` | `/analysis` | Renders Visual Analytics Dashboard |
| `GET` | `/evaluation` | Renders Model Benchmarking Dashboard |
| `POST` | `/predict/image?model=CNN` | Predicts emotion from uploaded image file |
| `POST` | `/predict/video?model=CNN` | Processes video frames for continuous emotion detection |
| `GET` | `/metrics` | Returns stored ANN & CNN test evaluation JSON metrics |
| `GET` | `/history` | Returns epoch-by-epoch training/validation history |
| `GET` | `/analysis/data` | Returns all analytics data and plot image paths |
| `GET` | `/status` | Returns model loading status and class mapping metadata |

### Sample API Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict/image?model=CNN" \
     -F "file=@face_sample.jpg"
```

### Sample API JSON Response:
```json
{
  "predicted_emotion": "Happy",
  "confidence": 0.9421,
  "probabilities": {
    "Angry": 0.0012,
    "Disgust": 0.0005,
    "Fear": 0.0028,
    "Happy": 0.9421,
    "Neutral": 0.0384,
    "Sad": 0.0089,
    "Surprise": 0.0061
  },
  "inference_ms": 1.92,
  "face_detected": true,
  "model_used": "CNN"
}
```

---

## ❓ Frequently Asked Questions (FAQ) & Troubleshooting

### Q: What if PowerShell displays a script execution error when activating the venv?
Run this command in PowerShell to temporarily permit script execution for your session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Q: What if Port 8000 is already in use?
You can start Uvicorn on another port (such as `8080`):
```bash
uvicorn app:app --reload --port 8080
```
Then open `http://localhost:8080` in your browser.

### Q: Can I run inference directly from my webcam?
Yes! Click **"Start Camera"** on the Home page (`http://localhost:8000`). The frontend captures frames at configured intervals and displays real-time emotion predictions and confidence scores.

---

## 📜 License & Credits

Developed as a Deep Learning & Computer Vision project utilizing **TensorFlow**, **OpenCV**, and **FastAPI**.
- Dataset: 7-Class Facial Emotion Dataset (`Image_data`)
- Documentation Deliverable: [report.pdf](report.pdf)
