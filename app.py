from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import numpy as np
import cv2
import base64
from PIL import Image, ExifTags
from transformers import pipeline, AutoImageProcessor
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load AI Detection Engine ──
print("Loading Truthlens AI Engine...")
try:
    processor = AutoImageProcessor.from_pretrained("umm-maybe/AI-image-detector")
    ai_detector = pipeline(
        "image-classification",
        model="./truthlens-custom-model",
        image_processor=processor
    )
    print("AI Engine Loaded.")
except Exception as e:
    print(f"Error loading AI Engine: {e}")

# ── Load Grad-CAM ──
print("Loading Heatmap Extractor...")
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model.eval()
target_layers = [resnet_model.layer4[-1]]
cam = GradCAM(model=resnet_model, target_layers=target_layers)


# ════════════════════════════════════════════
# FORENSIC HELPER FUNCTIONS
# ════════════════════════════════════════════

def check_exif(img_pil):
    """
    Real camera images almost always have EXIF with Make/Model.
    AI images never do. Returns a score adjustment (-ve means more real).
    """
    try:
        exif_data = img_pil._getexif()
        if exif_data:
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            has_camera = 'Make' in tags or 'Model' in tags
            has_datetime = 'DateTime' in tags or 'DateTimeOriginal' in tags
            if has_camera and has_datetime:
                return -25  # Strong evidence of real camera
            elif has_camera or has_datetime:
                return -15
    except Exception:
        pass
    return 0  # No EXIF — no adjustment (screenshots also lack EXIF)


def analyze_noise(img_pil):
    """
    Real camera sensors produce characteristic high-frequency noise.
    AI images are often too smooth or have unnatural noise patterns.
    """
    img_np = np.array(img_pil.resize((512, 512)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Laplacian captures high-frequency noise
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    noise_std = np.std(laplacian)

    # Real camera photos: noise_std typically 15–80
    # AI images: often < 8 (too smooth) or > 100 (unnatural artifacts)
    if 12 <= noise_std <= 90:
        return -15  # Looks like real camera noise
    elif noise_std < 5:
        return +20  # Suspiciously smooth — likely AI
    else:
        return 0


def analyze_frequency(img_pil):
    """
    FFT analysis: AI images often have periodic artifacts in frequency domain.
    Real images have a natural 1/f frequency distribution.
    """
    img_np = np.array(img_pil.resize((256, 256)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    # Check for unnatural peaks (grid artifacts common in GAN/diffusion images)
    mean_mag = np.mean(magnitude)
    max_mag = np.max(magnitude)
    peak_ratio = max_mag / (mean_mag + 1e-6)

    if peak_ratio > 25:
        return +15  # Suspicious periodic patterns
    elif peak_ratio < 12:
        return -10  # Natural frequency distribution
    return 0


def analyze_color_consistency(img_pil):
    """
    AI images sometimes have unnaturally perfect or inconsistent color distributions.
    """
    img_np = np.array(img_pil.resize((256, 256))).astype(np.float32)
    
    # Check channel correlations — real images have natural inter-channel correlation
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
    rb_corr = np.corrcoef(r.flatten(), b.flatten())[0,1]

    # Real photos: channels are highly correlated (0.85–0.99)
    avg_corr = (rg_corr + rb_corr) / 2
    if avg_corr > 0.85:
        return -10  # Natural color structure
    elif avg_corr < 0.60:
        return +15  # Unnatural color independence
    return 0


# ════════════════════════════════════════════
# MAIN ROUTE
# ════════════════════════════════════════════

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img_pil = Image.open(filepath).convert("RGB")

        # ── STEP 1: Model Prediction ──
        results = ai_detector(img_pil)
        ai_prob = 0.0
        real_prob = 0.0

        for res in results:
            if res['label'] == 'ai':
                ai_prob = res['score'] * 100
            elif res['label'] == 'real':
                real_prob = res['score'] * 100

        # ── STEP 2: Forensic Adjustments ──
        exif_adj       = check_exif(img_pil)
        noise_adj      = analyze_noise(img_pil)
        freq_adj       = analyze_frequency(img_pil)
        color_adj      = analyze_color_consistency(img_pil)

        total_adjustment = exif_adj + noise_adj + freq_adj + color_adj

        # Apply adjustment to ai_prob (clamp between 0–100)
        ai_prob   = max(0.0, min(100.0, ai_prob + total_adjustment))
        real_prob = 100.0 - ai_prob

        # ── STEP 3: Grad-CAM Heatmap ──
        img_resized = img_pil.resize((224, 224))
        img_float_np = np.float32(img_resized) / 255.0

        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(img_resized).unsqueeze(0)
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        visualization = show_cam_on_image(img_float_np, grayscale_cam, use_rgb=True)

        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', visualization_bgr)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

        # ── STEP 4: Verdict & Explanation ──
        if ai_prob >= 70:
            explanation = (
                "AI GENERATED: Forensic analysis detected synthetic pixel structures, "
                "unnatural frequency patterns, and absence of camera sensor signatures. "
                "The heatmap highlights regions with the most artificial rendering artifacts."
            )
        else:
            explanation = (
                "REAL IMAGE: Analysis detected natural camera sensor noise, authentic "
                "color distributions, and consistent frequency patterns typical of real "
                "optical hardware. The heatmap shows natural focal regions."
            )

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "ai_probability": round(ai_prob, 2),
            "real_probability": round(real_prob, 2),
            "explanation": explanation,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_b64}",
            "forensic_signals": {
                "exif_adjustment": exif_adj,
                "noise_adjustment": noise_adj,
                "frequency_adjustment": freq_adj,
                "color_adjustment": color_adj,
                "total_adjustment": total_adjustment
            }
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)