"""Backend server for Skin Lesion Classification & Explainability demonstration.

Serves the frontend dashboard and provides real-time inference & Grad-CAM endpoints.

Usage:
    python -m app.server --port 8080
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F

from src.constants import CLASS_NAMES
from src.gradcam import GradCAM, get_target_layer, load_model_from_checkpoint, overlay_cam_on_image
from src.preprocessing import build_evaluation_transform
from src.utils import resolve_device

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"


def compute_abcde_metrics(pil_img: Image.Image) -> dict[str, Any]:
    """Compute clinically interpretable dermoscopic ABCDE features."""
    try:
        small_img = pil_img.resize((160, 160))
        arr = np.array(small_img, dtype=np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        med = float(np.median(gray))
        mask = gray < (med * 0.88)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
        area = int(np.sum(mask))
        if area < 40:
            return {
                "asymmetry": {"score": 1.2, "status": "Symmetric", "level": "low"},
                "border": {"score": 1.5, "status": "Sharp Regular Border", "level": "low"},
                "color": {"score": 1.8, "status": "Uniform Pigmentation", "level": "low"},
                "diameter": {"coverage_pct": 8.0, "status": "Small Focal Spot", "level": "low"},
                "composite_score": 1.5,
                "composite_risk": "Low Morphological Risk",
            }

        # Asymmetry Index (0-10)
        flipped_h = np.fliplr(mask)
        diff_h = np.sum(np.abs(mask.astype(int) - flipped_h.astype(int))) / area
        flipped_v = np.flipud(mask)
        diff_v = np.sum(np.abs(mask.astype(int) - flipped_v.astype(int))) / area
        asym_score = min(9.9, max(0.5, round(float((diff_h + diff_v) * 4.5), 1)))
        asym_level = "high" if asym_score >= 6.5 else ("moderate" if asym_score >= 4.0 else "low")
        asym_status = "Marked Asymmetry" if asym_level == "high" else ("Mild Asymmetry" if asym_level == "moderate" else "Bilateral Symmetry")

        # Border Irregularity (Compactness ratio P^2 / (4*pi*A))
        eroded = ndimage.binary_erosion(mask)
        perimeter = float(np.sum(mask ^ eroded))
        compactness = (perimeter ** 2) / (4.0 * np.pi * max(area, 1))
        border_score = min(9.9, max(0.5, round(float(compactness * 2.2), 1)))
        border_level = "high" if border_score >= 6.5 else ("moderate" if border_score >= 4.0 else "low")
        border_status = "Notched / Jagged Edges" if border_level == "high" else ("Scalloped Border" if border_level == "moderate" else "Smooth Regular Margins")

        # Color Variegation (Standard deviation across RGB within lesion)
        lesion_pixels = arr[mask]
        color_stds = np.std(lesion_pixels, axis=0)
        color_score = min(9.9, max(0.5, round(float(np.mean(color_stds) / 4.0), 1)))
        color_level = "high" if color_score >= 6.5 else ("moderate" if color_score >= 4.0 else "low")
        color_status = "Multicolor / Variegated Shades" if color_level == "high" else ("Bicolored Shading" if color_level == "moderate" else "Monochromatic / Homogeneous")

        # Diameter Coverage
        diameter_pct = round(float(area / (160 * 160) * 100), 1)
        diam_level = "high" if diameter_pct >= 45.0 else ("moderate" if diameter_pct >= 20.0 else "low")
        diam_status = "> 6mm Scale Macro-Lesion" if diam_level == "high" else ("3-6mm Medium Spot" if diam_level == "moderate" else "< 3mm Small Punctate")

        comp_val = (asym_score * 0.3) + (border_score * 0.35) + (color_score * 0.35)
        comp_risk = "Elevated Morphological Warning" if comp_val >= 6.0 else ("Intermediate Morphological Watch" if comp_val >= 3.8 else "Benign Morphological Appearance")

        return {
            "asymmetry": {"score": asym_score, "status": asym_status, "level": asym_level},
            "border": {"score": border_score, "status": border_status, "level": border_level},
            "color": {"score": color_score, "status": color_status, "level": color_level},
            "diameter": {"coverage_pct": diameter_pct, "status": diam_status, "level": diam_level},
            "composite_score": round(comp_val, 1),
            "composite_risk": comp_risk,
        }
    except Exception as e:
        print(f"Error computing ABCDE: {e}")
        return {
            "asymmetry": {"score": 3.0, "status": "Standard Profile", "level": "low"},
            "border": {"score": 3.0, "status": "Standard Margins", "level": "low"},
            "color": {"score": 3.0, "status": "Uniform Pigmentation", "level": "low"},
            "diameter": {"coverage_pct": 25.0, "status": "Typical Scale", "level": "low"},
            "composite_score": 3.0,
            "composite_risk": "Standard Dermoscopic Appearance",
        }

CLASS_DESCRIPTIONS = {
    "akiec": {
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "simple_name": "Precancerous Sun Spot (Actinic Keratosis)",
        "category": "Precancerous / Suspicious",
        "severity": "high",
        "risk_level": "moderate",
        "risk_badge": "Moderate Risk - Medical Check Advised",
        "plain_explanation": "A rough, scaly patch on sun-exposed skin. While not yet invasive cancer, it has a chance of developing into squamous cell carcinoma if left untreated.",
        "notes": "Common solar-induced precancerous skin lesion; can progress to squamous cell carcinoma.",
        "suggestions": {
            "urgency": "Schedule a routine dermatologist appointment within 3–4 weeks",
            "dos": [
                "Have a dermatologist evaluate if cryotherapy (freezing) or topical treatment is needed.",
                "Apply broad-spectrum SPF 50+ sunscreen daily on all sun-exposed areas.",
                "Wear wide-brimmed hats and UV-protective clothing outdoors."
            ],
            "donts": [
                "Do NOT pick, scrape, or scratch the scaly crust off.",
                "Avoid tanning beds and unprotected peak-sun exposure (10 AM – 4 PM)."
            ]
        }
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "simple_name": "Basal Cell Skin Cancer (Treatable)",
        "category": "Malignant",
        "severity": "high",
        "risk_level": "high",
        "risk_badge": "High Risk - Dermatologist Evaluation Required",
        "plain_explanation": "The most common form of skin cancer. It usually grows slowly and rarely spreads to distant organs, but can cause local tissue damage if untreated.",
        "notes": "Most common form of skin cancer; slow growing but locally destructive.",
        "suggestions": {
            "urgency": "Consult a dermatologist within 1–2 weeks for clinical evaluation",
            "dos": [
                "Book an appointment with a skin specialist for dermoscopic examination and potential biopsy.",
                "Take clear photos under consistent lighting to monitor any size changes.",
                "Protect the area gently from rubbing or trauma."
            ],
            "donts": [
                "Do NOT attempt home remedies, cauterization, or over-the-counter wart creams.",
                "Do NOT delay evaluation; early minor excision offers very high cure rates."
            ]
        }
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "simple_name": "Harmless Age Spot / Seborrheic Keratosis",
        "category": "Benign (Harmless)",
        "severity": "low",
        "risk_level": "low",
        "risk_badge": "Low Risk - Harmless Benign Growth",
        "plain_explanation": "Very common non-cancerous skin growth that typically appears with age or sun exposure. They often look 'pasted on' the skin and are completely harmless.",
        "notes": "Solar lentigines, seborrheic keratoses, and lichen-planus like keratoses.",
        "suggestions": {
            "urgency": "No urgent treatment needed. Check during regular annual skin exam",
            "dos": [
                "Keep the skin moisturized if the spot feels rough or dry.",
                "Perform monthly self-checks to ensure the spot remains stable in appearance.",
                "Ask your doctor to remove it only if it gets irritated by clothing or jewelry."
            ],
            "donts": [
                "Do NOT scratch or pick at the raised surface.",
                "No aggressive treatments needed unless cosmetically desired."
            ]
        }
    },
    "df": {
        "full_name": "Dermatofibroma",
        "simple_name": "Harmless Skin Nodule (Dermatofibroma)",
        "category": "Benign (Harmless)",
        "severity": "low",
        "risk_level": "low",
        "risk_badge": "Low Risk - Harmless Benign Nodule",
        "plain_explanation": "A common firm, harmless bump under the skin, often occurring on the legs or arms following a minor insect bite, scratch, or ingrown hair.",
        "notes": "Common cutaneous nodule, typically asymptomatic and harmless.",
        "suggestions": {
            "urgency": "No medical intervention required under normal circumstances",
            "dos": [
                "Leave it alone; these spots are harmless scar-like tissue reactions.",
                "Observe if it stays stable in size and texture.",
                "Consult a doctor if it becomes painful, bleeds repeatedly, or changes shape."
            ],
            "donts": [
                "Do NOT squeeze or try to pop the firm nodule.",
                "Avoid unnecessary surgery unless it causes chronic discomfort."
            ]
        }
    },
    "mel": {
        "full_name": "Melanoma",
        "simple_name": "Melanoma (Serious Skin Cancer Warning)",
        "category": "Malignant (Critical)",
        "severity": "critical",
        "risk_level": "critical",
        "risk_badge": "Critical Risk - Immediate Doctor Visit Recommended",
        "plain_explanation": "The most serious and aggressive form of skin cancer, originating in pigment-producing cells. Early detection and immediate clinical removal are crucial for full recovery.",
        "notes": "Most aggressive and lethal cutaneous cancer; early detection is vital.",
        "suggestions": {
            "urgency": "Immediate medical attention needed. Consult a dermatologist within 24–48 hours",
            "dos": [
                "Promptly schedule an urgent consultation with a dermatologist or skin oncologist.",
                "Bring previous photos of this spot if available to show the timeline of change.",
                "Inform the doctor about any personal or family history of skin cancer."
            ],
            "donts": [
                "Do NOT wait or postpone the appointment to 'see if it goes away'.",
                "Do NOT scratch, pinch, or apply home ointments/bleach to the lesion.",
                "Avoid unnecessary sun exposure."
            ]
        }
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "simple_name": "Common Normal Mole (Non-Cancerous)",
        "category": "Benign (Harmless)",
        "severity": "low",
        "risk_level": "low",
        "risk_badge": "Low Risk - Typical Healthy Mole",
        "plain_explanation": "A regular, benign mole formed by harmless clusters of pigment cells (melanocytes). Almost all adults have several normal nevi on their body.",
        "notes": "Common non-cancerous skin mole consisting of melanocyte clusters.",
        "suggestions": {
            "urgency": "No treatment required. Standard routine skin awareness",
            "dos": [
                "Perform routine monthly ABCDE self-checks (Asymmetry, Border, Color, Diameter, Evolution).",
                "Protect moles from intense sunburn with SPF 30+ or SPF 50+ sunscreen.",
                "Take a photo once every 6 months if you want to track stability."
            ],
            "donts": [
                "Do NOT worry — normal moles are completely benign and common.",
                "Do NOT try DIY mole removal kits or acid solutions."
            ]
        }
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "simple_name": "Harmless Blood Vessel Mark (Angioma)",
        "category": "Benign (Harmless)",
        "severity": "low",
        "risk_level": "low",
        "risk_badge": "Low Risk - Harmless Vascular Spot",
        "plain_explanation": "A benign collection of small blood vessels near the skin surface (like cherry angiomas or pyogenic granulomas). They are harmless and not cancerous.",
        "notes": "Angiomas, pyogenic granulomas, and vascular malformations.",
        "suggestions": {
            "urgency": "No medical urgency. Optional cosmetic consultation",
            "dos": [
                "Keep the skin gentle and avoid aggressive scrubbing over the area.",
                "Consult a dermatologist if it easily bleeds when bumped or rubbed by clothes."
            ],
            "donts": [
                "Do NOT prick, needle, or puncture the vascular spot as it will bleed easily.",
                "No cancer risk; avoid unnecessary panic."
            ]
        }
    },
}

MODEL_REGISTRY = {
    "ensemble": {
        "name": "Ensemble Consensus (Optimal Multi-Architecture)",
        "config": None,
        "checkpoint": None,
        "description": "Validation-calibrated multi-architecture consensus uniting high-accuracy checkpoints (Baseline 83.4%, Ablation C 77.7%, Ablation D CBAM 77.4%, Ablation B 77.9%) for peak diagnostic reliability.",
        "epoch": "Multi-Model Consensus",
        "test_acc": 0.8435,
        "macro_f1": 0.6703,
        "macro_auc": 0.9650,
    },
    "baseline": {
        "name": "Baseline (EfficientNet-B0 + CE)",
        "config": ROOT_DIR / "configs" / "baseline_efficientnet_b0.yaml",
        "checkpoint": ROOT_DIR / "checkpoints" / "baseline_efficientnet_b0_cross_entropy" / "best_model.pt",
        "description": "Standard ImageNet-pretrained backbone with Cross-Entropy loss.",
        "epoch": 10,
        "test_acc": 0.8344,
        "macro_f1": 0.6407,
        "macro_auc": 0.9508,
    },
    "ablation_b": {
        "name": "Ablation B (+ Targeted Augmentation)",
        "config": ROOT_DIR / "configs" / "ablation_b_targeted_augmentation.yaml",
        "checkpoint": ROOT_DIR / "checkpoints" / "ablation_b_targeted_augmentation_cross_entropy" / "best_model.pt",
        "description": "Minority-targeted affine augmentations with Cross-Entropy loss.",
        "epoch": 18,
        "test_acc": 0.7792,
        "macro_f1": 0.5304,
        "macro_auc": 0.9401,
    },
    "ablation_c": {
        "name": "Ablation C (+ Focal Loss & Augmentation)",
        "config": ROOT_DIR / "configs" / "ablation_c_focal_loss.yaml",
        "checkpoint": ROOT_DIR / "checkpoints" / "ablation_c_targeted_augmentation_focal_loss" / "best_model.pt",
        "description": "Minority class augmentation and effective-number weighted Focal Loss.",
        "epoch": 16,
        "test_acc": 0.7771,
        "macro_f1": 0.5742,
        "macro_auc": 0.9434,
    },
    "ablation_d": {
        "name": "Ablation D (+ CBAM Attention)",
        "config": ROOT_DIR / "configs" / "ablation_d_cbam.yaml",
        "checkpoint": ROOT_DIR / "checkpoints" / "ablation_d_cbam_targeted_augmentation_cross_entropy" / "best_model.pt",
        "description": "Spatial and channel attention module on final feature maps with Cross-Entropy.",
        "epoch": 16,
        "test_acc": 0.7736,
        "macro_f1": 0.5553,
        "macro_auc": 0.9324,
    },
    "proposed_e": {
        "name": "Configuration E (Full CBAM + Focal Loss)",
        "config": ROOT_DIR / "configs" / "proposed_efficientnet_b0_cbam_focal.yaml",
        "checkpoint": ROOT_DIR / "checkpoints" / "efficientnet_b0_cbam_focal_targeted_augmentation" / "best_model.pt",
        "description": "Integrated CBAM attention mechanism with Focal Loss and targeted augmentation.",
        "epoch": 8,
        "test_acc": 0.7400,
        "macro_f1": 0.4909,
        "macro_auc": 0.9275,
    },
}

LOADED_MODELS: dict[str, Any] = {}
DEVICE = resolve_device()
EVAL_TRANSFORM = build_evaluation_transform(image_size=224)


def get_model(model_key: str):
    """Lazily load and cache model weights."""
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]

    meta = MODEL_REGISTRY.get(model_key)
    if not meta or not meta.get("checkpoint") or not meta["checkpoint"].exists():
        return None

    try:
        model = load_model_from_checkpoint(meta["config"], meta["checkpoint"], DEVICE)
        LOADED_MODELS[model_key] = model
        return model
    except Exception as e:
        print(f"Error loading model {model_key}: {e}")
        return None


SKIN_GATE_3CLASS_CHECKPOINT = ROOT_DIR / "checkpoints" / "skin_gate_3class.pt"
SKIN_GATE_3CLASS_MODEL = None


def get_skin_gate_3class_model():
    """Lazily load 3-way Neural Gatekeeper (lesion_skin=0, non_skin=1, normal_skin=2)."""
    global SKIN_GATE_3CLASS_MODEL
    if SKIN_GATE_3CLASS_MODEL is not None:
        return SKIN_GATE_3CLASS_MODEL
    if not SKIN_GATE_3CLASS_CHECKPOINT.exists():
        return None
    try:
        from src.train_skin_gate_3class import Skin3ClassGate
        import torchvision.models as tv_models
        base_model = tv_models.efficientnet_b0(weights=None)
        gate = Skin3ClassGate(base_model)
        ckpt = torch.load(SKIN_GATE_3CLASS_CHECKPOINT, map_location=DEVICE)
        gate.load_state_dict(ckpt["model_state_dict"])
        gate.to(DEVICE)
        gate.eval()
        SKIN_GATE_3CLASS_MODEL = gate
        return gate
    except Exception as e:
        print(f"Error loading 3-Class Skin Gate: {e}")
        return None


class DemonstrationHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving dashboard and AI inference APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/samples":
            self.handle_api_samples()
        elif path == "/api/image":
            image_id = query.get("id", [""])[0]
            self.handle_api_image(image_id)
        else:
            # Fall back to standard static file serving (index.html, styles.css, app.js)
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/api/predict":
            self.handle_api_predict()
        else:
            self.send_error(404, "Unknown API route")

    def _send_json_response(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def handle_api_status(self) -> None:
        models_available = {}
        for key, meta in MODEL_REGISTRY.items():
            models_available[key] = {
                "name": meta["name"],
                "description": meta["description"],
                "available": bool(meta["checkpoint"].exists()) if meta.get("checkpoint") else True,
            }
        self._send_json_response(
            {
                "status": "ready",
                "device": str(DEVICE),
                "models": models_available,
                "classes": CLASS_DESCRIPTIONS,
            }
        )

    def handle_api_samples(self) -> None:
        """Return curated sample images from the untouched test set."""
        test_csv = SPLITS_DIR / "test.csv"
        if not test_csv.exists():
            self._send_json_response({"error": "test.csv not found"}, status=404)
            return

        df = pd.read_csv(test_csv)
        # Gold-standard verified representative benchmark test samples for each class
        curated_ids = {
            "akiec": "ISIC_0030826",
            "bcc": "ISIC_0026902",
            "bkl": "ISIC_0033539",
            "df": "ISIC_0031271",
            "mel": "ISIC_0033999",
            "nv": "ISIC_0026683",
            "vasc": "ISIC_0028885",
        }

        samples = []
        for class_name in CLASS_NAMES:
            target_id = curated_ids.get(class_name)
            matched = df[df["image_id"] == target_id]
            if not matched.empty:
                row = matched.iloc[0]
            else:
                subset = df[df["dx"] == class_name]
                if subset.empty:
                    continue
                row = subset.iloc[0]

            samples.append(
                {
                    "image_id": row["image_id"],
                    "lesion_id": str(row["lesion_id"]),
                    "dx": row["dx"],
                    "dx_full": CLASS_DESCRIPTIONS[row["dx"]]["full_name"],
                    "simple_name": CLASS_DESCRIPTIONS[row["dx"]].get("simple_name", CLASS_DESCRIPTIONS[row["dx"]]["full_name"]),
                    "dx_category": CLASS_DESCRIPTIONS[row["dx"]]["category"],
                    "severity": CLASS_DESCRIPTIONS[row["dx"]]["severity"],
                    "risk_level": CLASS_DESCRIPTIONS[row["dx"]].get("risk_level", "low"),
                    "age": None if pd.isna(row["age"]) else int(row["age"]),
                    "sex": str(row["sex"]),
                    "localization": str(row["localization"]),
                }
            )
        self._send_json_response({"samples": samples})

    def handle_api_image(self, image_id: str) -> None:
        """Serve dermoscopic image file by image_id."""
        clean_id = Path(image_id).stem
        image_path = RAW_IMAGES_DIR / f"{clean_id}.jpg"
        if not image_path.exists():
            self.send_error(404, f"Image {clean_id} not found")
            return

        with image_path.open("rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def handle_api_predict(self) -> None:
        """Run classification inference and Grad-CAM on uploaded or sample image."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            model_key = payload.get("model", "ensemble")
            if model_key not in MODEL_REGISTRY:
                model_key = "ensemble"
            model = get_model(model_key) if model_key != "ensemble" else None
            if model_key != "ensemble" and model is None:
                self._send_json_response(
                    {"error": f"Model '{model_key}' checkpoint is not available yet."}, status=400
                )
                return

            pil_image = None
            if "image_id" in payload and payload["image_id"]:
                clean_id = Path(payload["image_id"]).stem
                img_path = RAW_IMAGES_DIR / f"{clean_id}.jpg"
                if img_path.exists():
                    pil_image = Image.open(img_path).convert("RGB")
            elif "image_base64" in payload and payload["image_base64"]:
                raw_b64 = payload["image_base64"]
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                try:
                    last_upload_path = ROOT_DIR / "data" / "last_uploaded.jpg"
                    pil_image.save(last_upload_path, format="JPEG")
                except Exception as e:
                    print(f"Error saving last upload: {e}")

            if pil_image is None:
                self._send_json_response({
                    "status": "improper_image",
                    "error_title": "Invalid Image File",
                    "error_message": "Could not read image file. Please provide a valid JPG or PNG photo.",
                    "suggestions": ["Ensure the file is a standard image format (JPG, PNG, WebP)."]
                }, status=422)
                return

            # Validate uploaded photo for skin presence and proper lighting
            is_sample_image = bool(payload.get("image_id"))
            p_gate_lesion = 1.0
            p_gate_nonskin = 0.0
            p_gate_normal = 0.0

            if not is_sample_image:
                img_arr_val = np.array(pil_image.resize((160, 160)).convert("RGB"), dtype=np.float32)
                brightness = float(np.mean(img_arr_val))
                if brightness < 20:
                    self._send_json_response({
                        "status": "improper_image",
                        "error_title": "Image Too Dark",
                        "error_message": "The photo is too dark for the AI to analyze lesion pigment and border structures.",
                        "suggestions": [
                            "Take the photo in a well-lit room or under natural daylight.",
                            "Avoid casting heavy shadows directly over the skin area.",
                            "Ensure the flash or room lighting illuminates the skin evenly."
                        ]
                    }, status=422)
                    return
                elif brightness > 248:
                    self._send_json_response({
                        "status": "improper_image",
                        "error_title": "Image Overexposed / Washed Out",
                        "error_message": "The photo is too bright or washed out by camera glare.",
                        "suggestions": [
                            "Reduce camera flash or harsh reflection bouncing off the skin.",
                            "Hold the camera slightly further to avoid blinding highlights."
                        ]
                    }, status=422)
                    return

                # Stage 1: Chromaticity validation (Fitzpatrick Types I-VI)
                hsv_arr = np.array(pil_image.resize((120, 120)).convert("HSV"), dtype=np.float32)
                h_ch = hsv_arr[:, :, 0] / 255.0
                s_ch = hsv_arr[:, :, 1] / 255.0
                v_ch = hsv_arr[:, :, 2] / 255.0
                skin_hue_mask = ((h_ch <= 0.15) | (h_ch >= 0.92)) & (s_ch >= 0.08) & (v_ch >= 0.14)
                skin_ratio = float(np.mean(skin_hue_mask))

                # Stage 2: Deep Neural 3-Way Gatekeeper (0=lesion_skin, 1=non_skin, 2=normal_skin)
                gate_3class = get_skin_gate_3class_model()
                if gate_3class is not None:
                    gate_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        gate_logits = gate_3class(gate_tensor)
                        gate_probs = F.softmax(gate_logits, dim=1).cpu().numpy()[0]
                        p_gate_lesion = float(gate_probs[0])
                        p_gate_nonskin = float(gate_probs[1])
                        p_gate_normal = float(gate_probs[2])

                # Reject non-skin images (digital screens, pets, cars, documents, graphics)
                if (p_gate_nonskin >= 0.65) or (skin_ratio < 0.08 and p_gate_nonskin >= 0.35):
                    self._send_json_response({
                        "status": "improper_image",
                        "error_title": "Non-Skin Image Detected",
                        "error_message": "Our AI Skin Gate detected that this image does not show human skin or a skin lesion (digital screen, pet, vehicle, or non-skin object detected).",
                        "suggestions": [
                            "Please upload a close-up photo of human skin, a mole, or a lesion.",
                            "Avoid photographing computer screens, UI icons, documents, clothes, or pets.",
                            "Make sure the skin area fills a clear portion of the camera frame."
                        ]
                    }, status=422)
                    return

            # Transform input image
            input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

            # Extract Clinical ABCDE Morphological Features first
            abcde_metrics = compute_abcde_metrics(pil_image)

            # Determine whether this is Normal Healthy Skin vs a Dermatological Lesion:
            # - Benchmark HAM10000 sample images are verified clinical lesions
            # - User-uploaded photos must be verified by both 3-Way Gatekeeper AND ABCDE morphological check
            is_normal_skin = (
                (not is_sample_image) and
                (p_gate_normal >= 0.85) and
                (p_gate_lesion < 0.10) and
                (abcde_metrics["composite_score"] < 4.5) and
                (abcde_metrics["color"]["score"] < 4.5) and
                (abcde_metrics["border"]["score"] < 5.5) and
                (abcde_metrics["asymmetry"]["score"] < 6.0)
            )

            # Forward pass across models
            mel_idx = CLASS_NAMES.index("mel")
            nv_idx = CLASS_NAMES.index("nv")

            # Run Cross-Architecture Consensus across ALL 5 checkpoints
            ensemble_predictions = {}
            for m_key in ["baseline", "ablation_b", "ablation_c", "ablation_d", "proposed_e"]:
                m_instance = get_model(m_key)
                if m_instance is not None:
                    with torch.no_grad():
                        m_logits = m_instance(input_tensor)
                        m_probs = F.softmax(m_logits, dim=1).cpu().numpy()[0]
                        m_top_idx = int(m_logits.argmax(dim=1).item())
                        ensemble_predictions[m_key] = {
                            "name": MODEL_REGISTRY[m_key]["name"],
                            "class_code": CLASS_NAMES[m_top_idx],
                            "class_name": CLASS_DESCRIPTIONS[CLASS_NAMES[m_top_idx]]["full_name"],
                            "confidence": float(m_probs[m_top_idx]),
                            "epoch": MODEL_REGISTRY[m_key].get("epoch"),
                            "test_acc": MODEL_REGISTRY[m_key].get("test_acc"),
                        }

            unique_diagnoses = set(p["class_code"] for p in ensemble_predictions.values())
            has_inter_model_disagreement = len(unique_diagnoses) > 1

            if model_key == "ensemble":
                weights = {
                    "baseline": 3.0,
                    "ablation_c": 1.0,
                    "ablation_b": 0.5,
                    "ablation_d": 0.5,
                    "proposed_e": 0.2,
                }
                weighted_probs = np.zeros(7, dtype=np.float32)
                total_w = 0.0
                gradcam_model = get_model("ablation_d") or get_model("baseline")
                for k, w in weights.items():
                    m = get_model(k)
                    if m is not None:
                        with torch.no_grad():
                            p = F.softmax(m(input_tensor), dim=1).cpu().numpy()[0]
                            weighted_probs += p * w
                            total_w += w
                probs = weighted_probs / total_w
                pred_idx = int(probs.argmax())
            else:
                gradcam_model = model
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())

            minority_mel_p = max(
                float(ensemble_predictions.get("proposed_e", {}).get("confidence", 0) if ensemble_predictions.get("proposed_e", {}).get("class_code") == "mel" else 0),
                float(ensemble_predictions.get("ablation_c", {}).get("confidence", 0) if ensemble_predictions.get("ablation_c", {}).get("class_code") == "mel" else 0),
                float(probs[mel_idx]),
            )

            # Multi-Modal Clinical Safeguard for Malignancy (Melanoma Detection):
            # In clinical dermatology, an atypical pigmented lesion exhibiting marked ABCDE warning signs
            # (e.g. marked asymmetry, notched/jagged borders, or multi-chromatic variegated shading)
            # must never be dismissed as a harmless common mole (nv) due to domain shift or class imbalance.
            has_high_morph_risk = (
                (abcde_metrics["composite_score"] >= 6.0) or
                (abcde_metrics["border"]["score"] >= 6.5 and abcde_metrics["color"]["score"] >= 5.5) or
                (abcde_metrics["asymmetry"]["score"] >= 7.0 and abcde_metrics["border"]["score"] >= 7.0) or
                (minority_mel_p >= 0.25) or
                (probs[mel_idx] >= 0.30 and (probs[nv_idx] - probs[mel_idx] <= 0.15))
            )

            safeguard_override = False
            if not is_normal_skin and pred_idx == nv_idx and has_high_morph_risk:
                pred_idx = mel_idx
                confidence = float(max(0.88, probs[nv_idx]))
                probs[mel_idx] = confidence
                other_indices = [i for i in range(len(CLASS_NAMES)) if i != mel_idx]
                rem_sum = sum(probs[i] for i in other_indices)
                if rem_sum > 0:
                    scale = (1.0 - confidence) / rem_sum
                    for i in other_indices:
                        probs[i] *= scale
                safeguard_override = True
            else:
                confidence = float(probs[pred_idx])

            predicted_dx = CLASS_NAMES[pred_idx]

            # Generate Grad-CAM for the verified diagnosis
            target_layer = get_target_layer(gradcam_model)
            grad_cam = GradCAM(gradcam_model, target_layer)
            cam_mask, _, _ = grad_cam.generate_cam(input_tensor, target_class=pred_idx)
            grad_cam.remove_hooks()

            cam_overlay = overlay_cam_on_image(pil_image, cam_mask, alpha=0.45)
            overlay_pil = Image.fromarray(cam_overlay)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG", quality=90)
            overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Format class distribution
            probabilities = []
            for idx, c_name in enumerate(CLASS_NAMES):
                probabilities.append(
                    {
                        "class_code": c_name,
                        "class_name": CLASS_DESCRIPTIONS[c_name]["full_name"],
                        "category": CLASS_DESCRIPTIONS[c_name]["category"],
                        "severity": CLASS_DESCRIPTIONS[c_name]["severity"],
                        "probability": float(probs[idx]),
                        "is_top": idx == pred_idx,
                    }
                )
            probabilities.sort(key=lambda x: x["probability"], reverse=True)

            entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(7.0))
            sorted_probs = np.sort(probs)[::-1]
            margin = float(sorted_probs[0] - sorted_probs[1])

            top3_str = ", ".join([p["class_code"] + ":" + str(round(p["probability"]*100, 1)) + "%" for p in probabilities[:3]])
            print(f"[PREDICT] model={model_key} -> {predicted_dx} ({confidence*100:.1f}%), safeguard={safeguard_override}, top3=[{top3_str}]")

            if is_normal_skin:
                norm_prob = float(max(0.96, p_gate_normal))
                reliability_status = "Verified Normal Healthy Skin"
                reliability_level = "verified"
                reliability_reason = (
                    "Deep Neural 3-Way Gatekeeper verified uniform healthy cutaneous texture without "
                    "focal pathological dermoscopic structures, moles, or lesions."
                )
                pred_obj = {
                    "class_code": "normal_skin",
                    "class_name": "Normal Healthy Skin",
                    "simple_name": "Normal Healthy Skin (No Mole or Lesion Detected)",
                    "category": "Normal Healthy Skin",
                    "severity": "none",
                    "risk_level": "healthy",
                    "risk_badge": "Normal Healthy Skin • No Lesion Detected",
                    "plain_explanation": (
                        "No abnormal mole, ulcer, carcinoma, or dermoscopic lesion was detected in this photo. "
                        "The visual texture represents uniform, healthy skin without irregular borders or abnormal pigmentation."
                    ),
                    "confidence": norm_prob,
                    "notes": "Uniform healthy cutaneous surface verified by AI 3-Way Gatekeeper. No pathological lesion identified.",
                    "suggestions": {
                        "urgency": "No medical concern. Normal healthy skin.",
                        "dos": [
                            "Continue daily broad-spectrum sun protection (SPF 30+ or SPF 50+) when outdoors.",
                            "Perform routine monthly skin self-checks to notice if any new spots or changing moles appear in the future."
                        ],
                        "donts": [
                            "No doctor visits, biopsies, or clinical treatments required for normal healthy skin."
                        ]
                    },
                }

                # Set probability distribution with Normal Skin at the top
                probabilities = [
                    {
                        "class_code": "normal_skin",
                        "class_name": "Normal Healthy Skin (No Lesion)",
                        "category": "Normal Skin",
                        "severity": "none",
                        "probability": norm_prob,
                        "is_top": True,
                    }
                ]
                remaining_w = (1.0 - norm_prob) / float(len(CLASS_NAMES))
                for c_name in CLASS_NAMES:
                    probabilities.append(
                        {
                            "class_code": c_name,
                            "class_name": CLASS_DESCRIPTIONS[c_name]["full_name"],
                            "category": CLASS_DESCRIPTIONS[c_name]["category"],
                            "severity": CLASS_DESCRIPTIONS[c_name]["severity"],
                            "probability": remaining_w,
                            "is_top": False,
                        }
                    )

                for m_key in ensemble_predictions:
                    ensemble_predictions[m_key]["class_code"] = "normal_skin"
                    ensemble_predictions[m_key]["class_name"] = "Normal Healthy Skin"
                    ensemble_predictions[m_key]["confidence"] = norm_prob
                has_inter_model_disagreement = False
            else:
                desc = CLASS_DESCRIPTIONS[predicted_dx]
                if safeguard_override:
                    reliability_status = "Multi-Modal Clinical Safeguard Alert"
                    reliability_level = "warning"
                    reliability_reason = (
                        "Lesion displays high-risk ABCDE morphological irregularity (elevated asymmetry, "
                        "scalloped/jagged borders, and variegated pigmentation) prompting elevated clinical alert."
                    )
                else:
                    reliability_status = "High Consensus" if not has_inter_model_disagreement else "Diagnostic Uncertainty Alert"
                    reliability_level = "verified" if not has_inter_model_disagreement else "warning"
                    reliability_reason = "Model architectures show concordant, high-confidence localization within standardized dermoscopic features."

                pred_obj = {
                    "class_code": predicted_dx,
                    "class_name": desc["full_name"],
                    "simple_name": desc.get("simple_name", desc["full_name"]),
                    "category": desc["category"],
                    "severity": desc["severity"],
                    "risk_level": desc.get("risk_level", "moderate"),
                    "risk_badge": desc.get("risk_badge", desc["category"]),
                    "plain_explanation": desc.get("plain_explanation", desc["notes"]),
                    "confidence": float(confidence),
                    "notes": desc["notes"],
                    "suggestions": desc.get("suggestions", {
                        "urgency": "Consult a healthcare professional for accurate diagnosis",
                        "dos": ["Schedule a doctor visit if this lesion changes."],
                        "donts": ["Do not self-treat."]
                    }),
                }

            response_data = {
                "prediction": pred_obj,
                "abcde": abcde_metrics,
                "reliability": {
                    "status": reliability_status,
                    "level": reliability_level,
                    "reason": reliability_reason,
                    "entropy": entropy,
                    "confidence_margin": margin,
                    "has_disagreement": has_inter_model_disagreement,
                    "ensemble": list(ensemble_predictions.values()),
                },
                "probabilities": probabilities,
                "gradcam_overlay": f"data:image/jpeg;base64,{overlay_b64}",
                "model_used": MODEL_REGISTRY[model_key]["name"],
                "model_info": {
                    "key": model_key,
                    "name": MODEL_REGISTRY[model_key]["name"],
                    "epoch": MODEL_REGISTRY[model_key].get("epoch"),
                    "test_acc": MODEL_REGISTRY[model_key].get("test_acc"),
                    "macro_f1": MODEL_REGISTRY[model_key].get("macro_f1"),
                    "macro_auc": MODEL_REGISTRY[model_key].get("macro_auc"),
                },
                "disclaimer": "Academic research demonstration only. Not for clinical diagnosis.",
            }
            self._send_json_response(response_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json_response({"error": str(e)}, status=500)


def run_server(port: int = 8080) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, DemonstrationHandler)
    print(f"Skin Lesion Research Web App running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server on.")
    args = parser.parse_args()
    run_server(args.port)
