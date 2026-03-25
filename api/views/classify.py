"""
classify.py — Wound Classification matching Clinical Tissue Analysis

Pipeline:
  1. Run CLIP categorizer → wound category validation
  2. Run specialized CV ROI detection (stitches, burns, scars)
  3. Run high-precision Lab-based tissue analysis
  4. Calculate healing score and stage using clinical heuristics
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
from api.clip_validator import clip_validator
from api.cv_processor import (
    analyze_tissue_lab, analyze_tissue_lab_v2, segment_wound, 
    process_specialized_wound, analyze_burn_tissue, 
    detect_open_wound_roi, get_roi_cropped_bytes
)
import cv2

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
        dressing_changed,
        wound_area=0,
        previous_area=None,
        previous_tissue=None):
    """
    Refined Healing Score Logic with Comparative Progress
    """
    # ── STEP 3: TISSUE BASE SCORE (0-40) ─────────────────────────────────────
    tissue_score = (
        granulation * 0.4 +
        epithelial * 0.35 +
        (slough + white) * 0.1 +
        necrotic * 0
    ) / 100 * 40
    tissue_score = max(0, min(40, tissue_score))

    # ── STEP 4: SYMPTOM SCORE (0-25) ─────────────────────────────────────────
    p_level = str(pain_level).lower()
    pain_score = 10 if p_level in ["none", "no", "0"] else 6 if p_level in ["mild", "moderate", "1"] else 2
    
    d_type = str(drainage_type).lower()
    drainage_score = 10 if d_type in ["dry", "no", "none", "0"] else 8 if d_type == "clear" else 4 if d_type == "yellow" else 0
    
    fever_score = 2.5 if not fever else 0
    redness_score = 2.5 if not redness_spreading else 0
    
    symptom_score = pain_score + drainage_score + fever_score + redness_score

    # ── STEP 5: TIMELINE SCORE (0-10) ────────────────────────────────────────
    timeline_score = 10 if surgery_days > 14 else 8 if surgery_days > 7 else 5

    # ── STEP 6: CARE SCORE (0-10) ────────────────────────────────────────────
    care_score = 10 if dressing_changed else 4

    # ── STEP 7: PROGRESS SCORE (0-15) ────────────────────────────────────────
    progress_score = 7.5 # Neutral starting point
    progress_label = "No history"

    if previous_area is not None and previous_area > 0:
        # 1. Percent Area Reduction (PAR)
        area_reduction = ((previous_area - wound_area) / previous_area) * 100
        
        # 2. Tissue Change Analysis
        tissue_bonus = 0
        if previous_tissue:
            curr_healthy = granulation + epithelial
            prev_healthy = previous_tissue.get('red', 20) + previous_tissue.get('pink', 20)
            tissue_bonus = (curr_healthy - prev_healthy) * 0.1 # Small bonus for healthy growth

        # Calculate final progress component
        # PAR of 20% + some healthy growth yields max bonus
        progress_contribution = (area_reduction * 0.4) + tissue_bonus
        progress_score = max(0, min(15, 7.5 + progress_contribution))

        if area_reduction > 10: progress_label = "Improving"
        elif area_reduction > -5: progress_label = "Stable"
        else: progress_label = "Worsening"

    # ── STEP 8: FINAL HEALING SCORE (0-100) ──────────────────────────────────
    healing_score = tissue_score + symptom_score + timeline_score + care_score + progress_score
    healing_score = round(max(0, min(100, healing_score)), 2)

    # ── STEP 9: INFECTION RISK & STATUS ──────────────────────────────────────
    infection_risk = "HIGH" if (fever or drainage_type == "green" or redness_spreading) else "LOW"
    
    if healing_score >= 85: status_label = "Excellent Healing"
    elif healing_score >= 70: status_label = "Good Healing"
    elif healing_score >= 55: status_label = "Moderate Healing"
    elif healing_score >= 40: status_label = "Poor Healing"
    else: status_label = "Critical Condition"

    return {
        "final_score": healing_score,
        "healingScore": healing_score,
        "tissue_score": round(tissue_score, 2),
        "symptom_score": symptom_score,
        "timeline_score": timeline_score,
        "care_score": care_score,
        "progress_score": round(progress_score, 2),
        "infection_risk": infection_risk,
        "statusLabel": status_label,
        "stage": status_label,
        "progress": progress_label
    }

def compute_severity_from_tissue(tissue: dict, discharge_type: str = "none",
                                  redness_level: float = 0) -> tuple[str, str, float]:
    """
    Compute wound_type and severity purely from tissue composition numbers.
    This is the authoritative classification — not dependent on Gemini's wound_type string.

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



# Tissue profiles are NO LONGER USED per user request (strict AI only)
TISSUE_PROFILES = {}


def normalize_wound_type(raw: str) -> str:
    """Normalize any wound type string to our 5 canonical labels."""
    return WOUND_TYPE_ALIASES.get(raw.lower().strip(), raw)






def _classify_wound_internal(request, save_to_db=True):
    """
    Internal logic for wound classification using CLIP and CV analysis.
    Decoupled from persistence to allow 'preview' mode.
    """
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

        # ── Step 1: Category Identification (Global Context First) ────────
        clip_category, clip_conf = clip_validator.classify(image_bytes)
        print(f"[CLASSIFY] Global Category: {clip_category} ({clip_conf:.1f}%)")

        # CV-based wound-like area detection for secondary validation
        _, cv_wound_area = detect_open_wound_roi(image_bytes)
        print(f"[CLASSIFY] CV-detected wound-like area: {cv_wound_area} px")

        # Categories that are always rejected (no active wound to analyse)
        REJECT_CATEGORIES = ["normal_skin", "no_wound", "scar", "healed_wound"]

        # Early rejection — hard (if CLIP identifies as non-wound) or uncertain (low CV area)
        is_strict_reject = clip_category in ["normal_skin", "no_wound"]
        is_heuristic_reject = clip_category in ["scar", "healed_wound"] and clip_conf > 70
        is_uncertain_reject = clip_category == "error" and cv_wound_area < 500

        if is_strict_reject or is_heuristic_reject or is_uncertain_reject:
            reason_map = {
                "normal_skin":   "healthy skin with no wound",
                "scar":          "a fully healed scar (no active wound)",
                "healed_wound":  "a healed wound (no active wound)",
            }
            rejection_reason = reason_map.get(clip_category, "non-wound image")
            print(f"[CLASSIFY] {clip_category} detected (conf={clip_conf:.1f}, area={cv_wound_area}). Rejecting as {rejection_reason}.")
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except: pass
            wound.delete()
            return Response({
                "success": False,
                "error": f"No active wound detected: This photo appears to show {rejection_reason}. Please upload a clear photo of an active wound that requires monitoring.",
                "is_invalid": True
            }, status=status.HTTP_400_BAD_REQUEST)

        # ── Step 2: ROI Focusing & Refinement ──────────────────────────────
        # Only sutured_wound and burn_wound get specialized processing now
        is_specialized = clip_category in ["sutured_wound", "burn_wound"]
        temp_mask, temp_area = None, 0

        # If it's an open wound OR uncertainty is high, try segmentation to focus
        if clip_category == "open_wound" or (not is_specialized and clip_conf < 75):
            temp_mask, temp_area = segment_wound(image_bytes)
            if temp_area > 1000:
                print(f"[CLASSIFY] Segment model found focus area ({temp_area} px). Refining...")
                roi_bytes = get_roi_cropped_bytes(image_bytes, temp_mask)
                roi_category, roi_conf = clip_validator.classify(roi_bytes)
                
                # If segmentation is very confident (>8000px) and global was normal_skin, override
                if clip_category in ["normal_skin", "no_wound"] and temp_area > 8000:
                    print(f"[CLASSIFY] Segmentation confirmed open area. Overriding {clip_category} to open_wound.")
                    clip_category = "open_wound"
                    clip_conf = 80.0
                elif not is_specialized:
                    # Update category from ROI if not specialized
                    clip_category = roi_category
                    clip_conf = roi_conf
            
            # Final check post-refinement — reject healed/non-wound categories
            if clip_category in ["normal_skin", "no_wound", "scar", "healed_wound"]:
                print(f"[CLASSIFY] Post-refinement rejection: {clip_category}")
                if os.path.exists(img_path):
                    try: os.remove(img_path)
                    except: pass
                wound.delete()
                return Response({
                    "success": False,
                    "error": "No active wound detected: Analysis confirmed this area does not contain an active wound.",
                    "is_invalid": True
                }, status=status.HTTP_400_BAD_REQUEST)

        mask = None
        wound_area = 0
        specialized_data = None

        if clip_category == "open_wound":
            # Re-use the mask from the ROI step if available, or re-run
            if 'temp_mask' in locals() and temp_area > 1000:
                mask = temp_mask
                wound_area = temp_area
            else:
                mask, wound_area = segment_wound(image_bytes)
        else:
            specialized_data = process_specialized_wound(image_bytes, clip_category)
            if specialized_data:
                mask = specialized_data.get("mask")
                if mask is not None:
                    wound_area = int(np.sum(mask > 0))
        # ROI focused tissue analysis (using segment mask if open_wound)
        if clip_category == "open_wound" and mask is not None:
            roi_bytes_focused = get_roi_cropped_bytes(image_bytes, mask)
        else:
            roi_bytes_focused = image_bytes
            
        clip_tissue = clip_validator.classify_tissue(roi_bytes_focused)
        
        # High-precision Lab-based tissue analysis
        try:
            if clip_category in ["sutured_wound", "scar"]:
                print(f"[CLASSIFY] Running Specialized V2 Tissue Analysis for {clip_category}...")
                lab_tissue_data = analyze_tissue_lab_v2(image_bytes, mask=mask)
            else:
                print(f"[CLASSIFY] Running Standard Tissue Analysis for {clip_category}...")
                lab_tissue_data = analyze_tissue_lab(image_bytes, mask=mask, category=clip_category)
                
            if lab_tissue_data:
                # Exclude huge base64 string from logs
                print_data = {k: v for k, v in lab_tissue_data.items() if k != 'overlay_image' and k != 'boundary_image'}
                # print(f"[CLASSIFY] Tissue Analysis (LAB): {print_data}")
        except Exception as cv_err:
            print(f"[CLASSIFY] Lab tissue analysis failed: {cv_err}")
        
        # ── Combined Tissue Analysis (CLIP semantic + LAB pixel precision) ────
        tissue_composition = {
            "red": 0, "pink": 0, "yellow": 0, "black": 0, "white": 0
        }
        
        # ── Combined Tissue Analysis (CLIP semantic + LAB pixel precision) ────
        if lab_tissue_data:
            # User requested pure CV analysis (LAB pixel-level precision)
            # Re-mapping LAB keys to standard UI keys
            for key in ["red", "pink", "yellow", "black", "white"]:
                lab_key = {
                    "red": "granulation", "pink": "epithelial", 
                    "yellow": "slough", "black": "necrotic", "white": "white"
                }[key]
                # Use 100% Lab precision if available
                tissue_composition[key] = float(round(lab_tissue_data.get(lab_key, 0.0), 1))
        elif clip_tissue:
            # Fallback to CLIP only if CV analysis failed/missing
            tissue_composition.update(clip_tissue)
        # ── Step 3: Healing Score Calculation ────────────────────────────────
        # Retrieve clinical data from request if available (passed from analyze_full)
        clinical_data = request.data.get('clinical_data', {})
        surgery_days = int(clinical_data.get('daysSinceSurgery', 0))
        pain_level = clinical_data.get('painLevel', 'none')
        drainage_type = clinical_data.get('discharge', 'dry')
        if drainage_type == 'no': drainage_type = 'dry' # Handle frontend 'no' mapping
        fever = bool(clinical_data.get('fever', False))
        redness_spreading = bool(clinical_data.get('rednessSpread', False))
        dressing_changed = bool(clinical_data.get('dressingChanged', True))

        # Healing Progress Logic: Compare current wound area with previous scan if available
        previous_area = None
        previous_tissue = None
        if wound.case:
            last_wound = Wound.objects.filter(
                case=wound.case,
                status='analyzed'
            ).exclude(id=wound.id).order_by('-upload_date').first()
            
            if last_wound:
                previous_area = last_wound.analysis.get('wound_area_pixels') if last_wound.analysis else None
                previous_tissue = last_wound.tissue_composition
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
            dressing_changed=dressing_changed,
            wound_area=wound_area,
            previous_area=previous_area,
            previous_tissue=previous_tissue
        )
        
        final_score = float(healing_results["final_score"])
        heur_type = str(healing_results["stage"])
        # ── Tissue-based severity (primary signal for ALL active wounds) ────────
        # compute_severity_from_tissue() maps tissue color % → severity label
        tissue_type, tissue_severity, _ = compute_severity_from_tissue(
            tissue_composition,
            discharge_type=drainage_type,
            redness_level=30.0   # baseline; no dynamic redness sensor yet
        )
        # Map to 3-level UI labels used downstream
        _SEVERITY_MAP = {
            "Critical":  "Critical",   # black >= 20 %
            "High":      "High",       # black >= 10 % or active infection indicators
            "Moderate":  "Moderate",   # slough-dominant / infection risk
            "Low":       "Low",        # healthy tissue dominant
        }
        tissue_severity = _SEVERITY_MAP.get(tissue_severity, "Moderate")

        if clip_category == "burn_wound":
            final_type = "Burn Wound"
            severity_level = tissue_severity   # tissue color drives burn severity too
        elif clip_category == "sutured_wound":
            final_type = "Sutured / Surgical"
            severity_level = tissue_severity
        elif clip_category in ["scar", "healed_wound"]:
            # Safety fallback — should never reach here (rejected earlier)
            final_type = "Healed Scar"
            severity_level = "Low"
        else:
            # open_wound — use the heuristic label from healing score
            final_type = heur_type
            severity_level = tissue_severity

        print(f"\n[CLASSIFY] Tissue → {tissue_severity} | Result: {final_type} | OVERALL SCORE: {final_score:.2f}%")
        print(f"[CLASSIFY] Tissue composition: { {k: f'{v}%' for k,v in tissue_composition.items()} }")
        
        # Risk level mapping for Android UI
        android_risk = "normal"
        if final_score < 35: android_risk = "critical"
        elif final_score < 55: android_risk = "infected"
        elif final_score < 75: android_risk = "warning"

        # Detected Features
        features = [f"Analysis: {final_type}"]
        
        # Inject raw tissue data for UI debugging
        tissue_summary = f"CV: R={tissue_composition.get('red',0)}% P={tissue_composition.get('pink',0)}% Y={tissue_composition.get('yellow',0)}%"
        features.append(tissue_summary)
        
        # ── Step 4: Final Probabilities & Formatting ─────────────────────────
        final_confidence = min(98, final_score + 5)
        
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

        # ── Step 5: Persistence (Conditional) ────────────────────────────────
        clf_id = None
        if save_to_db:
            clf = Classification.objects.create(
                wound=wound,
                wound_type=final_type,
                confidence=final_confidence,
                all_probabilities=final_probs,
                processing_time_ms=processing_time,
            )
            clf_id = clf.id

            wound.status = "analyzed"
            wound.classification = final_type
            wound.confidence = final_confidence
            wound.redness_level = 30.0 # Standard local baseline
            wound.discharge_detected = False
            wound.discharge_type = "none"
            wound.edge_quality = 70
            wound.tissue_composition = tissue_composition
            
            # Strip heavy base64 images from JSON data to prevent DB bloat
            lab_metrics_clean = None
            if lab_tissue_data:
                lab_metrics_clean = {k: v for k, v in lab_tissue_data.items() if k not in ['overlay_image', 'boundary_image']}

            wound.analysis = {
                "source": "lab_cv_segmentation_fusion",
                "tissue_composition": tissue_composition,
                "lab_metrics": lab_metrics_clean,
                "wound_area_pixels": wound_area,
                "healing_details": healing_results,
                "healingScore": healing_results.get("healingScore", final_score),
                "riskLevel": android_risk,
                "severityLevel": severity_level,
                "severityLabel": tissue_type,
                "notes": f"High-precision segmentation analysis ({'Masked' if mask is not None else 'Unmasked'}). Dominant: {max(tissue_composition, key=lambda k: tissue_composition.get(k, 0))}. Area: {wound_area} px."
            }
            wound.save()

        # Build response payload (Always returned)
        response_data = {
            "success":           True,
            "classification_id": clf_id,
            "wound_id":          wound.id, # Useful for confirmation
            "wound_type":        final_type,
            "confidence":        final_confidence,
            "probabilities":     final_probs,
            "severity_level":    severity_level,
            "severity_label":    tissue_type,         # e.g. "Active Infection", "Normal Healing"
            "severity_score":    final_score,
            "risk_level":        android_risk,
            "healing_details":   healing_results,
            "redness_level":     30.0,
            "discharge_detected": False,
            "discharge_type":    "none",
            "edge_quality":      70,
            "tissue_composition":tissue_composition,
            "detectedFeatures":  features,
            "wound_location":    None,
            "processing_time_ms":processing_time,
            "source":            "clinical_analysis",
            "specialized_metrics": {k: v for k, v in specialized_data.items() if k != 'mask'} if specialized_data else None,
            "is_healed":         clip_category in ["scar", "healed_wound"],
            "boundary_image":    lab_tissue_data.get("boundary_image") if lab_tissue_data else None,
            "overlay_image":     lab_tissue_data.get("overlay_image") if lab_tissue_data else None,
            "burn_detail":       lab_tissue_data.get("burn_detail") if lab_tissue_data else None,
            "scar_detail":       lab_tissue_data.get("scar_detail") if lab_tissue_data else None,
        }

        return Response(response_data) if save_to_db else response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[CLASSIFY] Local classification failed: {e}")
        error_resp = {"success": False, "error": str(e)}
        return Response(error_resp, status=status.HTTP_500_INTERNAL_SERVER_ERROR) if save_to_db else error_resp

@api_view(['POST'])
def classify_wound(request):
    """
    Classify wound using CLIP + Local CV Analysis.
    Wraps the internal logic for the public API endpoint.
    """
    return _classify_wound_internal(request, save_to_db=True)
