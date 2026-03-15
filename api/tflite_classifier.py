"""
TFLite Wound Classifier — runs the custom trained model on wound images.
Input:  224x224 RGB float32 image, normalized to [0, 1]
Output: 4-class softmax probabilities
Classes (in order from training, mapped to our app categories):
  0 → Normal Healing
  1 → Delayed Healing
  2 → Infection Risk / Active Infection
  3 → High Urgency / Critical
"""

import os
import io
import numpy as np
import warnings

# Suppress TF noisy logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

from PIL import Image, ImageFile

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'classification_model.tflite')
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Maps TFLite output class index → app wound type label
CLASS_LABELS = [
    "Normal Healing",       # index 0
    "Delayed Healing",      # index 1
    "Infection Risk",       # index 2
    "High Urgency",         # index 3
]

# Tissue composition heuristics per class
# Based on clinical profiles — used to seed tissue estimates per model output
TISSUE_PROFILES = {
    "Normal Healing":  {"red": 55, "pink": 35, "yellow": 5,  "black": 0, "white": 5},
    "Delayed Healing": {"red": 35, "pink": 20, "yellow": 30, "black": 5, "white": 10},
    "Infection Risk":  {"red": 40, "pink": 5,  "yellow": 40, "black": 5, "white": 10},
    "High Urgency":    {"red": 15, "pink": 0,  "yellow": 25, "black": 55, "white": 5},
}

_interpreter = None


def _get_interpreter():
    """Lazily load the TFLite interpreter (singleton)."""
    global _interpreter
    if _interpreter is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"TFLite model not found at: {MODEL_PATH}")
        print(f"[TFLite] Loading model from {MODEL_PATH}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        print("[TFLite] Model loaded successfully")
    return _interpreter


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to a [1, 224, 224, 3] float32 tensor in [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # Shape: [1, 224, 224, 3]


def analyze_wound_pixels(image_array: np.ndarray) -> dict:
    """
    Perform pixel-level heuristic analysis for critical markers.
    Input: image_array [1, 224, 224, 3] float32 in [0, 1]
    Returns: {marker: percentage}
    """
    arr = image_array[0]  # Remove batch dim -> [224, 224, 3]
    total_pixels = 224 * 224

    # 1. Necrosis (Very dark pixels)
    # R < 0.2, G < 0.2, B < 0.2
    necrosis_mask = np.all(arr < 0.20, axis=-1)
    necrosis_pct = (np.sum(necrosis_mask) / total_pixels) * 100

    # 2. Metal / Hardware (High brightness, low saturation)
    # Using a slightly loose threshold for gray/silver
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = np.mean(arr, axis=-1)
    # Max difference between channels (proxy for saturation)
    # Metal is typically neutral gray or shiny white
    diff = np.max(arr, axis=-1) - np.min(image_array[0], axis=-1)
    
    # Metal detection: High-ish brightness, extremely low color variation
    # Broadened brightness and lowered diff threshold to catch more metal area
    hardware_mask = (diff < 0.10) & (brightness > 0.30) & (brightness < 0.95)
    hardware_pct = (np.sum(hardware_mask) / total_pixels) * 100

    # 3. Slough/Infection Signs (Yellow/Creamy)
    # R > 0.6, G > 0.5, B < 0.5, |R-G| small
    yellow_mask = (r > 0.6) & (g > 0.5) & (b < 0.5) & (np.abs(r - g) < 0.15)
    yellow_pct = (np.sum(yellow_mask) / total_pixels) * 100

    return {
        "necrosis": round(necrosis_pct, 1),
        "hardware": round(hardware_pct, 1),
        "slough": round(yellow_pct, 1)
    }


def classify_image(image_bytes: bytes) -> dict:
    """
    Run TFLite 4-class inference on image bytes.
    Classes: Normal Healing | Delayed Healing | Infection Risk | High Urgency
    The model's softmax output is the final result — no heuristic overrides.
    """
    interpreter = _get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor = preprocess_image(image_bytes)

    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]  # shape [4]
    # Normalize to 100% (handles both softmax and non-softmax outputs)
    raw_sum = float(sum(raw_output))
    if raw_sum > 0:
        final_probs = {CLASS_LABELS[i]: round(float(raw_output[i]) / raw_sum * 100, 1) for i in range(4)}
    else:
        final_probs = {CLASS_LABELS[i]: 25.0 for i in range(4)}

    # Log all 4 classes clearly
    print(f"[TFLite] 4-class output:")
    for label, prob in final_probs.items():
        print(f"  {label}: {prob}%")

    top_label = max(final_probs, key=final_probs.get)
    top_conf = final_probs[top_label]

    print(f"[TFLite] Final: {top_label} ({top_conf}%)")

    return {
        "wound_type": top_label,
        "confidence": top_conf,
        "probabilities": final_probs,
        "tissue_profile": TISSUE_PROFILES[top_label],
        "heuristics": {},
        "model": "tflite",
    }
