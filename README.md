# Truelens
An AI-powered image forensics tool that detects AI-generated images using a fine-tuned vision transformer, classical forensic signal analysis, and Grad-CAM heatmap visualization — served through a Flask backend and a browser-based dashboard.

---

## Overview
This system analyzes a single uploaded image and returns a probability score for whether it is AI-generated or an authentic camera photo. The classification combines a fine-tuned image classification model with four independent forensic checks — EXIF metadata, sensor noise patterns, frequency-domain artifacts, and color channel correlation - then renders a Grad-CAM heatmap and a human-readable explanation alongside the verdict.

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | PyTorch, Hugging Face Transformers, Datasets |
| Backend | Flask, Flask-CORS |
| Forensic analysis | OpenCV, NumPy |
| Heatmap | pytorch-grad-cam, torchvision (ResNet50) |
| Frontend | HTML, CSS, JavaScript |
| Base model | `umm-maybe/AI-image-detector` (fine-tuned) |

## Module Breakdown

| File | Responsibility |
|---|---|
| `train_model.py` | Unzips dataset, fine-tunes the base vision model on FAKE/REAL image folders |
| `evaluate_model.py` | Loads the fine-tuned model and reports accuracy/loss on a held-out slice |
| `app.py` | Flask server — runs inference, forensic checks, and Grad-CAM heatmap generation |
| `index.html` | Upload UI, results dashboard, and probability/heatmap rendering |
| `requirements.txt` | Python dependencies for the inference server |

## System Flow

```
Image Upload (Frontend)
        |
        v
Flask /analyze Endpoint
        |
        v
Fine-Tuned Model Inference
        |
        v
Forensic Signal Analysis
   |     |     |     |
 EXIF  Noise  FFT   Color
        |
        v
Adjusted AI / Real Probability
        |
        v
Grad-CAM Heatmap Generation
        |
        v
Verdict + Explanation -> Dashboard
```

## Highlights
- Fine-tuned vision transformer for AI vs. real image classification
- Multi-signal forensic scoring layered on top of model confidence
- EXIF metadata inspection for camera Make/Model/DateTime signatures
- Sensor noise analysis via Laplacian variance to flag unnaturally smooth images
- Frequency-domain (FFT) check for periodic GAN/diffusion artifacts
- Color channel correlation analysis for unnatural color independence
- Grad-CAM heatmap overlay for visual interpretability
- Single-page upload-and-scan dashboard with animated probability bars
- Self-contained local inference — no external API calls at runtime

## Dashboard
The web dashboard provides drag-and-drop image upload, an animated AI-vs-real probability comparison, a forensic heatmap overlay, and a plain-language explanation of the verdict — all served locally against the Flask backend.

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Train the model** (expects a zipped dataset with `train/`, `val` or `valid`, and `test` folders containing `FAKE`/`REAL` class subfolders)
```
python train_model.py
```

**3. Evaluate the model**
```
python evaluate_model.py
```

**4. Start the backend server**
```
python app.py
```

**5. Open the dashboard**
```
index.html
```

## Intended Use Cases
- Digital forensics and media verification
- Journalism and fact-checking workflows
- Content moderation pipelines
- Academic research into AI-generated image detection

## Roadmap
- Unify model output paths across training, evaluation, and serving
- Replace the generic ResNet50 Grad-CAM with a heatmap derived from the fine-tuned detector itself
- Validate and calibrate forensic heuristic thresholds against labeled datasets
- Batch/multi-image analysis support
- Cloud-hosted deployment with persistent model versioning

## License
For academic purposes only
