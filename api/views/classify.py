"""
classify.py — Wound Classification with TFLite + Local CV Analysis

Pipeline:
  1. Run custom TFLite model → wound type classification (4 classes)
  2. Run CLIP categorizer → wound category validation
  3. Run specialized CV ROI detection (stitches, burns, scars)
  4. Run high-precision Lab-based tissue analysis
  5. Calculate healing score and stage using clinical heuristics
"""

import json
import time
import os
import numpy as np
from pathlib import Path
from PIL import ImageFile

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Classification
from api import tflite_classifier
from api.clip_validator import clip_validator
from api.cv_processor import analyze_tissue_lab, segment_wound, process_specialized_wound, analyze_burn_tissue

# TFLite class order → normalized label
# User-provided Healing Score Logic
def calculate_healing_score(
        granulation,
        epithelial,
        slough,
        necrotic,
        white,
        surgery_days,
        pain_level,
        drainage_type,
        fever,
        redness_spreading,
        dressing_changed):

    # 1. TISSUE BASE SCORE
    gran_score = granulation * 1.0
    epi_score = epithelial * 0.9
    slough_score = slough * -0.6
    nec_score = necrotic * -1.2
    white_score = white * -0.3

    tissue_raw = gran_score + epi_score + slough_score + nec_score + white_score
    tissue_score = (tissue_raw + 100) / 2
    tissue_score = max(0, min(100, tissue_score))

    # 2. CLINICAL SYMPTOM SCORE
    clinical_score = 0
    pain_scores = {"none": 5, "mild": 0, "severe": -10}
    clinical_score += pain_scores.get(pain_level, 0)

    drainage_scores = {"dry": 5, "clear": 0, "yellow": -15, "green": -25}
    clinical_score += drainage_scores.get(drainage_type, 0)

    clinical_score += -20 if fever else 5
    clinical_score += -15 if redness_spreading else 3
    clinical_score += 2 if dressing_changed else -5
    clinical_score = max(-40, min(20, clinical_score))

    # 3. SURGERY TIMELINE CHECK
    timeline_score = 0
    if surgery_days > 10 and necrotic > 20: timeline_score -= 10
    if surgery_days > 14 and slough > 20: timeline_score -= 8
    if surgery_days > 7 and granulation < 20: timeline_score -= 5
    if surgery_days > 10 and epithelial > 15: timeline_score += 5

    # 4. FINAL SCORE
    final_score = tissue_score + clinical_score + timeline_score
    final_score = max(1, min(100, final_score))

    # 5. HEALING STAGE
    if final_score >= 85: stage = "Excellent Healing"
    elif final_score >= 70: stage = "Good Healing"
    elif final_score >= 50: stage = "Moderate Healing"
    elif final_score >= 30: stage = "Poor Healing"
    else: stage = "High Infection Risk"

    return {
        "tissue_score": round(tissue_score, 2),
        "clinical_score": clinical_score,
        "timeline_score": timeline_score,
        "final_score": round(final_score, 2),
        "stage": stage
    }

def compute_severity_from_tissue(tissue: dict, discharge_type: str = "none",
                                  redness_level: float = 0) -> tuple[str, str, float]:
    """
    Compute wound_type and severity purely from tissue composition numbers.
    This is the authoritative classification — not dependent on external AI wound_type strings.

    Returns: (wound_type, severity_level, confidence_boost)
    """
    red    = float(tissue.get("red",    0))
    pink   = float(tissue.get("pink",   0))
    yellow = float(tissue.get("yellow", 0))
    white  = float(tissue.get("white",  0))
    black  = float(tissue.get("black",  0))

    slough = yellow + white
    healthy = red + pink

    # ── Critical: necrosis present ────────────────────────────────────────────
    if black >= 20:
        return "High Urgency", "Critical", 92.0
    if black >= 10:
        return "High Urgency", "High", 88.0

    # ── Active infection signals ───────────────────────────────────────────────
    if discharge_type in ("yellow", "green") and slough >= 15:
        return "Active Infection", "High", 85.0
    if slough >= 40 and redness_level >= 70:
        return "Active Infection", "High", 82.0

    # ── Infection risk ─────────────────────────────────────────────────────────
    if slough >= 30:
        return "Infection Risk", "Moderate", 78.0
    if slough >= 20 and redness_level >= 60:
        return "Infection Risk", "Moderate", 75.0

    # ── Delayed healing ────────────────────────────────────────────────────────
    if slough >= 15:
        return "Delayed Healing", "Moderate", 72.0
    if healthy < 30 and slough >= 10:
        return "Delayed Healing", "Low", 68.0

    # ── Normal / healing well ─────────────────────────────────────────────────
    if pink >= 40:
        return "Normal Healing", "Low", 85.0   # Epithelialization — excellent
    if red >= 50:
        return "Normal Healing", "Low", 80.0   # Good granulation
    if healthy >= 60:
        return "Normal Healing", "Low", 75.0

    # ── Default: borderline ───────────────────────────────────────────────────
    return "Delayed Healing", "Low", 65.0


# ── Wound type normalization map ───────────────────────────────────────────────
WOUND_TYPE_ALIASES = {
    "normal healing":       "Normal Healing",
    "healing":              "Normal Healing",
    "delayed healing":      "Delayed Healing",
    "delayed":              "Delayed Healing",
    "infection risk":       "Infection Risk",
    "infection risk assessment": "Infection Risk",
    "active infection":     "Active Infection",
    "infected":             "Active Infection",
    "high urgency":         "High Urgency",
    "urgency level":        "High Urgency",
    "critical":             "High Urgency",
    "necrotic":             "High Urgency",
}

ALL_TYPES = ["Normal Healing", "Delayed Healing", "Infection Risk", "Active Infection", "High Urgency"]

# TFLite class order → normalized label
TFLITE_CLASSES = ["Normal Healing", "Delayed Healing", "Infection Risk", "High Urgency"]

# Tissue profiles are NO LONGER USED per user request (strict AI only)
TISSUE_PROFILES = {}


def normalize_wound_type(raw: str) -> str:
    """Normalize any wound type string to our 5 canonical labels."""
    return WOUND_TYPE_ALIASES.get(raw.lower().strip(), raw)





@api_view(['POST'])
def classify_wound(request):
    """Classify wound using TFLite + Local CLIP Tissue Analysis."""
    wound_id = request.data.get('wound_id')
    if not wound_id:
        return Response({"success": False, "error": "wound_id required"}, status=status.HTTP_400_BAD_REQUEST)

    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"success": False, "error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    # Resolve actual disk path
    img_path = wound.image_path
    if not os.path.isabs(img_path):
        img_path = os.path.join(settings.BASE_DIR, img_path)
    if not Path(img_path).exists():
        return Response({"success": False, "error": f"Image file not found: {img_path}"}, status=status.HTTP_404_NOT_FOUND)

    try:
        start_time = time.time()
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        # ── Step 1: TFLite Wound Type Classification (Local) ──────────────────
        tflite_result = tflite_classifier.classify_image(image_bytes)
        print(f"[CLASSIFY] TFLite: {tflite_result['wound_type']} ({tflite_result['confidence']}%)")

        # ── Step 2a: CLIP + TFLite Agreement Check ────────────────────────────
        clip_category, clip_conf = clip_validator.classify(image_bytes)
        print(f"[CLASSIFY] CLIP Category: {clip_category} ({clip_conf:.1f}%)")

        # Early rejection for normal skin if confident
        if clip_category in ["normal_skin", "no_wound"] and clip_conf > 75:
            print(f"[CLASSIFY] {clip_category} detected. Rejecting as non-wound image.")
            # Delete record to keep database clean
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except: pass
            wound.delete()
            return Response({
                "success": False, 
                "error": "No wound detected: This photo appears to be normal skin. Please upload a clear photo of the wound.",
                "is_invalid": True
            }, status=status.HTTP_400_BAD_REQUEST)

        # FALLBACK: If CLIP returns normal_skin, sutured_wound, or scar,
        # but TFLite is confident it IS an active healing stage, investigate further.
        tflite_is_wound = tflite_result['wound_type'] in ["Normal Healing", "Delayed Healing", "Infection Risk", "High Urgency"]
        
        # If CLIP says it's NOT an open wound, but TFLite says it IS, try segmentation as a tie-breaker
        # CRITICAL: We EXEMPT sutured_wound from this override because stitches should never be called "open" even if area is large.
        is_exempt = clip_category in ["sutured_wound", "burn_wound"]
        
        if not is_exempt and clip_category != "open_wound" and tflite_is_wound and tflite_result['confidence'] > 25:
            print(f"[CLASSIFY] CLIP/TFLite mismatch — CLIP={clip_category}, TFLite={tflite_result['wound_type']}({tflite_result['confidence']:.1f}%). Verifying...")
            # Run segmentation quickly to see if there's an actual open area
            temp_mask, temp_area = segment_wound(image_bytes)
            # If segmentation model finds a significant area (>2% of image or some absolute threshold), trust it
            if temp_area > 5000: # Increased threshold for "significant" open area (approx 0.5% of 1MP)
                print(f"[CLASSIFY] Segmentation confirmed open area ({temp_area} px). Overriding CLIP to open_wound.")
                clip_category = "open_wound"
            else:
                print(f"[CLASSIFY] Segmentation confirmed NO large open area. Keeping CLIP category: {clip_category}")
                # If CLIP said normal skin and segmentation is small, reject now
                if clip_category in ["normal_skin", "no_wound"]:
                    wound.delete()
                    return Response({
                        "success": False, 
                        "error": "No wound detected: Analysis confirmed this area appears healthy.",
                        "is_invalid": True
                    }, status=status.HTTP_400_BAD_REQUEST)

        mask = None
        wound_area = 0
        specialized_data = None

        if clip_category == "open_wound":
            print("[CLASSIFY] Open wound detected. Running high-precision segmentation...")
            mask, wound_area = segment_wound(image_bytes)
        else:
            print(f"[CLASSIFY] {clip_category} detected. Running specialized ROI detection...")
            specialized_data = process_specialized_wound(image_bytes, clip_category)
            if specialized_data:
                mask = specialized_data.get("mask")
                if mask is not None:
                    wound_area = int(np.sum(mask > 0))
                print(f"[CLASSIFY] Specialized ROI found. Incision: {specialized_data.get('incision_detected')}")
                
            # --- 🚀 NEW: CLIP/CV Tie-breaker for Stitched Wounds 🚀 ---
            # If CLIP said "scar" but specialized CV found an incision, trust the CV.
            if clip_category == "scar" and specialized_data and specialized_data.get("incision_detected"):
                print(f"[CLASSIFY] CLIP=scar, CV detected incision. Overriding to sutured_wound.")
                clip_category = "sutured_wound"

        clip_tissue = clip_validator.classify_tissue(image_bytes)
        
        # High-precision Lab-based tissue analysis ONLY if it's an open wound
        lab_tissue_data = None
        burn_data = None
        if clip_category == "burn_wound":
            try:
                burn_data = analyze_burn_tissue(image_bytes, mask=mask)
                print(f"[CLASSIFY] Burn Tissue Indicators: {burn_data}")
            except Exception as cv_err:
                print(f"[CLASSIFY] Burn tissue analysis failed: {cv_err}")
        
        # Enable high-precision Lab-based tissue analysis for ALL detected categories
        # This ensures overlays (boundary/tissue color) are generated for sutured wounds and scars too.
        try:
            lab_tissue_data = analyze_tissue_lab(image_bytes, mask=mask, category=clip_category)
            if lab_tissue_data:
                # Exclude huge base64 string from logs
                print_data = {k: v for k, v in lab_tissue_data.items() if k != 'overlay_image' and k != 'boundary_image'}
                print(f"[CLASSIFY] Tissue Analysis (LAB): {print_data}")
        except Exception as cv_err:
            print(f"[CLASSIFY] Lab tissue analysis failed: {cv_err}")
        
        # Merge results, prioritize LAB CV for precision
        tissue_composition = {
            "red": 20, "pink": 20, "yellow": 20, "black": 20, "white": 20
        }
        
        if clip_tissue:
            tissue_composition.update(clip_tissue)
            
        if lab_tissue_data:
            tissue_composition["red"] = lab_tissue_data["granulation"]
            tissue_composition["pink"] = lab_tissue_data["epithelial"]
            tissue_composition["yellow"] = lab_tissue_data["slough"]
            tissue_composition["black"] = lab_tissue_data["necrotic"]
            tissue_composition["white"] = lab_tissue_data["white"]

        # ── Step 3: Local Clinical Logic ──────────────────────────────────────
        # Use TFLite model's type as the primary source of truth if confident (>70%)
        # Otherwise, use the clinical heuristic as a tie-breaker.
        tflite_type = tflite_result['wound_type']
        tflite_conf = tflite_result['confidence']
        
        # ── Step 3: Healing Score Calculation ────────────────────────────────
        # Retrieve clinical data from request if available (passed from analyze_full)
        clinical_data = request.data.get('clinical_data', {})
        surgery_days = int(clinical_data.get('daysSinceSurgery', 0))
        pain_level = clinical_data.get('painLevel', 'none')
        drainage_type = clinical_data.get('discharge', 'dry')
        fever = bool(clinical_data.get('fever', False))
        redness_spreading = bool(clinical_data.get('rednessSpread', False))
        dressing_changed = bool(clinical_data.get('dressingChanged', True))

        healing_results = calculate_healing_score(
            granulation=tissue_composition.get('red', 0),
            epithelial=tissue_composition.get('pink', 0),
            slough=tissue_composition.get('yellow', 0),
            necrotic=tissue_composition.get('black', 0),
            white=tissue_composition.get('white', 0),
            surgery_days=surgery_days,
            pain_level=pain_level,
            drainage_type=drainage_type,
            fever=fever,
            redness_spreading=redness_spreading,
            dressing_changed=dressing_changed
        )
        
        final_score = float(healing_results["final_score"])
        heur_type = str(healing_results["stage"])
        
        if clip_category == "burn_wound":
            final_type = "Burn Wound"
            # Maintain existing burn severity logic or override with healing_results
            severity_level = "High" if final_score < 40 else ("Moderate" if final_score < 70 else "Low")
        elif clip_category in ["sutured_wound"]:
            final_type = "Sutured / Surgical"
            severity_level = "Low" if final_score >= 70 else "Moderate"
        elif clip_category in ["scar", "healed_wound"]:
            final_type = "Healed Scar"
            severity_level = "Low"
        elif tflite_conf > 70:
            final_type = tflite_type
            severity_level = "High" if final_score < 30 else ("Moderate" if final_score < 60 else "Low")
        else:
            final_type = heur_type
            severity_level = "High" if final_score < 30 else ("Moderate" if final_score < 60 else "Low")
            
        print(f"[CLASSIFY] Choice: Model={tflite_type}({tflite_conf}%) vs Heur={heur_type}. Final={final_type} Score={final_score}")

        # ── Step 4: Final Probabilities & Formatting ─────────────────────────
        final_confidence = min(98, tflite_result['confidence'] + 5)
        
        # Standard healing-stage types shown in probabilities breakdown
        all_types = ["Low Risk", "Delayed Healing", "Monitor Closely", "Infected / High Risk", "Critical Urgent"]
        final_probs = {t: 0.0 for t in all_types}
        # For CLIP-specific types (Burn Wound, Sutured / Surgical, Healed Scar),
        # map to the nearest standard type so the probabilities dict stays consistent.
        prob_key = str(final_type) if final_type in all_types else str({
            "Burn Wound":        "Monitor Closely",
            "Sutured / Surgical": "Low Risk",
            "Healed Scar":       "Low Risk",
        }.get(str(final_type), "Low Risk"))
        final_probs[prob_key] = float(round(final_confidence, 1))
        remaining = 100 - final_confidence
        others = [t for t in all_types if t != prob_key]
        for t in others:
            final_probs[t] = round(remaining / len(others), 1)

        processing_time = int((time.time() - start_time) * 1000)

        # ── Step 5: Save to Database ──────────────────────────────────────────
        clf = Classification.objects.create(
            wound=wound,
            wound_type=final_type,
            confidence=final_confidence,
            all_probabilities=final_probs,
            processing_time_ms=processing_time,
        )

        wound.status = "analyzed"
        wound.classification = final_type
        wound.confidence = final_confidence
        wound.redness_level = 30.0 # Standard local baseline
        wound.discharge_detected = False
        wound.discharge_type = "none"
        wound.edge_quality = 70
        wound.tissue_composition = tissue_composition
        
        # Strip heavy base64 images from JSON data to prevent DB bloat
        # These are still returned in the API response below but not saved to the persistent analysis history
        lab_metrics_clean = None
        if lab_tissue_data:
            lab_metrics_clean = {k: v for k, v in lab_tissue_data.items() if k not in ['overlay_image', 'boundary_image']}

        wound.analysis = {
            "source": "lab_cv_segmentation_fusion",
            "tissue_composition": tissue_composition,
            "lab_metrics": lab_metrics_clean,
            "wound_area_pixels": wound_area,
            "healing_details": healing_results,
            "notes": f"High-precision segmentation analysis ({'Masked' if mask is not None else 'Unmasked'}). Dominant: {max(tissue_composition, key=lambda k: tissue_composition.get(k, 0))}. Area: {wound_area} px."
        }
        wound.save()

        return Response({
            "success":           True,
            "classification_id": clf.id,
            "wound_type":        final_type,
            "confidence":        final_confidence,
            "probabilities":     final_probs,
            "severity_level":    severity_level,
            "severity_score":    final_score,
            "healing_details":   healing_results,
            "redness_level":     30.0,
            "discharge_detected": False,
            "discharge_type":    "none",
            "edge_quality":      70,
            "tissue_composition":tissue_composition,
            "wound_location":    [0, 0, 1000, 1000],
            "processing_time_ms":processing_time,
            "source":            "local_only",
            "specialized_metrics": {k: v for k, v in specialized_data.items() if k != 'mask'} if specialized_data else None,
            "is_healed":         clip_category in ["scar", "healed_wound"],
            "boundary_image":    lab_tissue_data.get("boundary_image") if lab_tissue_data else None,
            "overlay_image":     lab_tissue_data.get("overlay_image") if lab_tissue_data else None,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[CLASSIFY] Local classification failed: {e}")
        return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
