import base64
from io import BytesIO
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageFile
from django.conf import settings

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.clip_validator import clip_validator
from api.cv_processor import process_specialized_wound, detect_open_wound_roi




def calculate_blur_score(image_array: np.ndarray) -> float:
    # Ensure RGB for consistent laplacian variance calculation
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def log_debug(message):
    print(f"[VALIDATE] {message}")






@api_view(['POST'])
def validate_image(request):
    """Validate image for blur and wound detection using local models ONLY."""
    try:
        image_data = request.data.get('image_data', '')
        if not image_data:
            return Response({"success": False, "error": "No image data provided"}, status=status.HTTP_400_BAD_REQUEST)

        if ',' in image_data:
            image_data_b64 = image_data.split(',')[1]
        else:
            image_data_b64 = image_data

        image_bytes = base64.b64decode(image_data_b64)

        # ── Step 0: CLIP validation (Quick categorization) ──────────────────
        clip_category, clip_conf = clip_validator.classify(image_bytes)
        log_debug(f"CLIP categorizer: {clip_category} ({clip_conf:.1f}%)")
        
        # CV-based wound-like area detection for secondary validation
        # Returns (crop, area_px)
        _, cv_wound_area = detect_open_wound_roi(image_bytes)
        log_debug(f"CV-detected wound-like area: {cv_wound_area} px")

        # ── Step 0.0: Strict Rejection Logic ──────────────────────────────
        # CLIP IS FINAL AUTHORITY (User Request: No percentage gates)
        if clip_category == "no_wound":
            log_debug(f"Hard rejection: CLIP detected non-medical or non-wound content ({clip_conf:.1f}%)")
            return Response({
                "success": True,
                "is_valid": False,
                "is_blur": False,
                "has_wound": False,
                "is_healed": False,
                "blur_score": 0,
                "wound_confidence": clip_conf,
                "message": "This doesn't look like a medical photo. Please upload a clear photo of the wound area.",
            })

        if clip_category in ["normal_skin"]:
            log_debug(f"Hard rejection: CLIP classified as {clip_category} ({clip_conf:.1f}%)")
            return Response({
                "success": True,
                "is_valid": False,
                "is_blur": False,
                "has_wound": False,
                "is_healed": False,
                "blur_score": 0,
                "wound_confidence": clip_conf,
                "message": "No active wound detected. Please upload a clear photo of the wound area.",
            })

        # 2. Rejection if CLIP is uncertain but CV finds NO significant wound-like area
        if clip_category == "error" and cv_wound_area < 800:
            log_debug(f"Rejection: CLIP error + No significant CV wound area ({cv_wound_area})")
            return Response({
                "success": True,
                "is_valid": False,
                "is_blur": False,
                "has_wound": False,
                "is_healed": False,
                "blur_score": 0,
                "wound_confidence": 10,
                "message": "We couldn't clearly identify a wound in this photo. Please ensure the area is well-lit and centered.",
            })

        # 3. Crash/Error failsafe
        if clip_category == "error" and clip_conf == 0:
            log_debug("Categorizer error: default rejection for safety")
            return Response({
                "success": True,
                "is_valid": False,
                "is_blur": False,
                "has_wound": False,
                "is_healed": False,
                "blur_score": 0,
                "wound_confidence": 0,
                "message": "Service error during validation. Please try again.",
            })


        # ── Step 0.1: Specialized CV Analysis (Stitches/Scars) ──────────────
        cv_metrics = None
        if clip_category in ["sutured_wound", "scar"] and clip_conf > 60:
            log_debug(f"Triggering specialized CV analysis for: {clip_category}")
            cv_metrics = process_specialized_wound(image_bytes, clip_category)

        pil_image = Image.open(BytesIO(image_bytes))
        image_array = np.array(pil_image)

        # ── Step 1: Blur check (only hard gate) ──────────────────────────────
        blur_score = calculate_blur_score(image_array)
        BLUR_THRESHOLD = 30.0   # lowered from 50 — many real phone photos score 50-200
        is_blur = blur_score < BLUR_THRESHOLD

        log_debug(f"Blur score: {blur_score:.2f} (threshold={BLUR_THRESHOLD}, is_blur={is_blur})")

        if is_blur:
            return Response({
                "success": True,
                "is_valid": False,
                "is_blur": True,
                "has_wound": False,
                "is_healed": False,
                "blur_score": blur_score,
                "wound_confidence": 0,
                "message": "Image is too blurry. Please retake the photo in good lighting and hold the camera steady.",
            })

        # ── Step 2: Local wound detection decision ───────────────────────
        # We now rely exclusively on CLIP and specialized CV logic.
        # If CLIP is confident it's an open_wound, burn, or sutured, we accept it.
        
        is_valid = True
        has_wound = True
        is_healed = False
        wound_confidence = clip_conf
        message = "Wound detected. Proceeding with analysis."

        if clip_category == "no_wound":
            is_valid = False
            has_wound = False
            message = "This doesn't look like a medical photo or no active wound was detected. Please upload a clear photo of the wound area."
        elif clip_category in ["normal_skin"]:

            is_valid = False
            has_wound = False
            message = "No active wound detected. Please upload a clear photo of the wound area."
        elif clip_category in ["scar", "healed_wound"]:
            is_healed = True
            message = "Healed or closed wound detected."
        elif clip_category == "blur":
            is_valid = False
            has_wound = False
            message = "The image appears to be blurry. Please retake the photo."
        elif clip_category == "error":
            # If uncertain, we default to valid=True but with lower confidence
            is_valid = True
            has_wound = False 

            wound_confidence = 30
            message = "Wound not clearly identified. Please ensure the area is well-lit and centered."

        log_debug(f"Final validation: is_valid={is_valid}, category={clip_category}, message={message}")
        if cv_metrics and "mask" in cv_metrics:
            del cv_metrics["mask"]

        return Response({
            "success": True,
            "is_valid": is_valid,
            "is_blur": is_blur,
            "has_wound": has_wound,
            "is_healed": is_healed,
            "blur_score": blur_score,
            "wound_confidence": wound_confidence,
            "message": message,
            "cv_analysis": cv_metrics
        })

    except Exception as e:
        log_debug(f"validate_image top-level error: {type(e).__name__}: {e}")
        # Even if everything fails, return a valid response so the app can continue
        return Response({
            "success": True,
            "is_valid": False,
            "is_blur": False,
            "has_wound": False,
            "is_healed": False,
            "blur_score": 999,
            "wound_confidence": 0,
            "message": "Validation error: Please try again or ensure the photo is clear.",

        })

