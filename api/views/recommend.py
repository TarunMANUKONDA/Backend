import json
import time
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Classification, Recommendation
from api.views.classify import calculate_healing_score

def generate_enhanced_local_recommendations(classification, wound_type, risk_level, severity_score, tissue, symptoms):
    """
    Generates granular, deterministic clinical recommendations based on wound metadata.
    This replaces the OpenAI implementation to ensure 100% local, private analysis.
    """
    # Base defaults
    summary = ""
    cleaning = ["Remove visible dirt carefully.", "Clean the wound gently with sterile saline."]
    dressing = ["Apply a sterile non-adherent dressing."]

    warnings = ["Do not pour alcohol, hydrogen peroxide, or iodine inside deep wounds — they damage tissue and delay healing.", "Increased pain or throbbing", "New or worsening fever", "Spreading redness"]

    seek_help = ["If symptoms worsen significantly", "Fever above 101°F (38.3°C)"]
    diet = ["Ensure adequate protein intake (chicken, beans, eggs) for tissue repair."]
    activity = ["Avoid strenuous activity that stretches the wound area."]
    healing_time = "2-4 weeks (est), pending clinical review."

    black = float(tissue.get('black', 0))
    yellow = float(tissue.get('yellow', 0))
    white = float(tissue.get('white', 0))
    red = float(tissue.get('red', 0))
    pink = float(tissue.get('pink', 0))
    
    fever = bool(symptoms.get('fever', False))
    discharge = symptoms.get('discharge_type', 'none')
    redness_spread = bool(symptoms.get('redness_spread', False))

    # Extract wound area if available
    analysis = classification.wound.analysis or {}
    wound_area = analysis.get("wound_area_pixels", 0)

    # --- Rule 1: Necrosis / Criticality ---
    if risk_level == "Critical" or "Critical" in wound_type:
        summary = "CRITICAL URGENT: Urgent medical review required. Signs of significant complications detected."
        seek_help.insert(0, "SEEK IMMEDIATE MEDICAL ATTENTION (ER OR SURGEON).")
        warnings.extend(["Foul odor", "Deepening black or leathery tissue"])
        cleaning.append("DO NOT scrub black tissue; wait for professional debridement.")
        dressing = ["Cover loosely with sterile gauze.", "Do not apply ointments without surgical approval."]
        diet.append("High-calorie, high-protein support is vital.")
        healing_time = "Requires professional clinical management."

    # --- Rule 2: Active Infection / High Risk ---
    elif risk_level == "High" or "Infected" in wound_type:
        summary = "INFECTED / HIGH RISK: Signs of active infection or significant dead tissue detected. Monitoring is critical."
        cleaning = ["Use an antimicrobial wash if prescribed.", "Clean twice daily with sterile saline."]
        warnings.append("Pus-like or cloudy discharge")
        seek_help.insert(0, "Contact your healthcare provider within 24 hours.")
        dressing = ["Use an absorbent dressing (foam or alginate) if discharge is heavy."]
        diet.append("Vitamin C and Zinc help support immune response.")
        activity.append("Limit movement to prevent spreading irritation.")
        healing_time = "4-6 weeks, may depend on infection control."

    # --- Rule 3: Monitor Closely ---
    elif risk_level == "Monitor closely (wounds are open)" or wound_type == "Monitor Closely":
        summary = "MONITOR CLOSELY: Wound requires vigilant observation due to slough presence or slow progression. Regularly monitor your wound and if you find some abnormal pain then seek medical help immediately."

        if risk_level == "Monitor closely (wounds are open)":
            summary = "MONITOR CLOSELY: Wound is open with good granulation. Maintain protection and pressure as needed. Regularly monitor your wound and if you find some abnormal pain then seek medical help immediately."

            cleaning.append("Use clean gauze or cloth and apply direct pressure on the wound.")
            cleaning.append("Apply antibiotic ointment if available.")
            dressing = ["Cover with sterile gauze or a clean bandage."]


        
        cleaning.append("Ensure the skin around the wound (periwound) is kept dry.")
        if "wounds are open" not in risk_level:
            dressing.extend(["Use a moisture-wicking dressing if needed.", "Apply a barrier cream to healthy skin edges."])
        warnings.append("Skin around the wound turning white or soggy")
        healing_time = "2-3 weeks, monitoring status."

    # --- Rule 4: Sutured / Surgical Wounds ---
    elif wound_type == "Sutured / Surgical" or "sutured" in wound_type.lower():
        summary = "SUTURED WOUND: Incision is closed. Keep the area clean, dry, and protected from tension to ensure optimal healing."
        cleaning.append("Clean gently with saline or mild soap (if approved) and pat completely dry.")
        cleaning.append("Do not soak or submerge the wound in water (no baths or swimming).")
        dressing = ["Cover with a light, sterile dressing to protect from clothing rubbing and external contaminants."]
        warnings.extend([
            "Increased redness or warmth around sutures", 
            "Pus or cloudy drainage from the incision line", 
            "Sutures opening, popping, or 'gaping' in the incision",
            "Foul odor from the surgical site"
        ])
        activity.append("Avoid lifting weights or activities that cause tension across the incision line.")
        healing_time = "7-14 days until suture/staple removal (typical)."

    # --- Rule 4b: Healed Scars ---
    elif wound_type == "Healed Scar" or "scar" in wound_type.lower():
        summary = "HEALED SCAR: Wound is closed and matured. Focus on scar remodeling and protection."
        cleaning = ["Wash with mild soap and water.", "Keep the area moisturized to improve elasticity."]
        dressing = ["Apply silicone gel or silicone sheets (if recommended by your surgeon) to flatten and soften the scar."]
        warnings = ["Excessive itching or pain in the scar area", "Rapid thickening or widening of the scar (potential keloid)"]
        seek_help = ["If the scar becomes painful, restricted, or develops a new opening."]
        diet.append("Stay hydrated to maintain skin health.")
        activity = ["Gradually resume normal movements; massage the scar gently to break up adhesions if approved by a therapist."]
        healing_time = "Mature scars continue to remodel for 6-12 months."
        warnings.append("UV Exposure: Keep the scar out of direct sunlight or use SPF 50+ to prevent permanent discoloration.")

    # --- Rule 5: Burn Wounds ---
    elif wound_type == "Burn Wound":
        summary = "BURN WOUND: Burn injury detected. Protection from infection and proper moisture balance is essential."
        cleaning = ["Cool the burn with cool (not ice-cold) running water for 10-20 minutes if it is a recent injury.", "Clean gently with mild soap and water. DO NOT pop any blisters."]
        dressing = ["Apply a thin layer of antibiotic ointment or aloe vera.", "Cover with a sterile, non-stick gauze (e.g., Telfa) and wrap loosely."]
        warnings.extend(["Deep burn appearance (white, leathery, or charred)", "Burn covers a large area of the body", "Burn is on the face, hands, feet, groin, or major joints"])
        seek_help.insert(0, "Seek emergency care for deep burns, electrical burns, or chemical burns.")
        healing_time = "1-3 weeks depending on burn depth. Deep burns require a doctor."

    # --- Rule 6: Normal / Healthy Granulation ---
    elif pink > 50 or (red + pink) > 80 or wound_type == "Low Risk":
        summary = "LOW RISK: Wound shows good signs of granulation and healing. Maintain current hygiene."
        cleaning.append("Continue gentle cleaning once daily.")
        dressing = ["Protect the thin new skin with a non-stick pad."]
        diet.append("Continue balanced nutrition.")
        activity.append("Light activity is generally acceptable.")
        healing_time = "1-2 weeks for full closure."

    # --- Rule 7: Delayed Healing ---
    elif wound_type == "Delayed Healing":
        summary = "DELAYED HEALING: Progression is slower than expected. Consistent care is required."
        healing_time = "3-4 weeks (est), pending clinical review."

    else:
        summary = f"WOUND ASSESSMENT: {wound_type} is stable but requires consistent care and protection."

    # --- Rule 8: Wound Area Adjustments ---
    if wound_area > 30000:
        summary += " Large wound area noted."
        dressing.append("Ensure dressing size is sufficient to cover the entire wound bed.")
        activity.append("Strictly limit movement to avoid tension on the large wound area.")

    # --- Rule 9: Specific Wound Type Adjustments ---
    t_lower = wound_type.lower()
    if 'surgical' in t_lower or 'post-op' in t_lower or 'sutured' in t_lower:
        activity.append("Avoid lifting weights over 5 lbs (2.3 kg).")
        cleaning.insert(0, "Check for any broken sutures or gap in the incision.")
    elif 'pressure' in t_lower:
        activity.append("Relieve pressure from the site every 2 hours.")

    # --- Rule 10: Bleeding / Bloody Discharge Adjustments ---
    if discharge == 'bloody':
        summary = "BLEEDING DETECTED: Wound shows signs of bloody discharge. " + summary
        cleaning.insert(0, "Apply direct pressure with a clean, sterile gauze for 10-15 minutes if active bleeding is present.")
        cleaning.append("Avoid vigorous cleaning or scrubbing which may restart bleeding.")
        warnings.append("Bleeding that does not stop after 15 minutes of continuous pressure")
        warnings.append("Feeling lightheaded or dizzy")
        seek_help.insert(0, "Seek medical attention if bleeding is persistent or heavy.")
        activity.append("Keep the affected area elevated if possible.")

    return {

        "summary": summary,
        "cleaningInstructions": cleaning,
        "dressingRecommendations": dressing,
        "warningsSigns": warnings,
        "whenToSeekHelp": seek_help,
        "dietAdvice": diet,
        "activityRestrictions": activity,
        "expectedHealingTime": healing_time,
        "followUpSchedule": ["Follow up with your surgeon in 1 week.", "Daily self-inspection."],
        "confidence": 95
    }

def get_clinical_recommendations_logic(classification, wound_type, confidence, symptoms=None):
    """
    Core logic for generating recommendations using the rule-based engine.
    """
    symptoms = symptoms or {}
    pain_level = symptoms.get('pain_level', 'none')
    fever = bool(symptoms.get('fever', False))
    discharge_type = symptoms.get('discharge_type', 'none')
    redness_spread = bool(symptoms.get('redness_spread', False))

    classification.wound.refresh_from_db()
    img_redness = float(classification.wound.redness_level or 0)
    img_discharge = classification.wound.discharge_type or 'none'

    final_discharge_type = symptoms.get('discharge_type', 'none')
    if final_discharge_type in ['none', ''] and img_discharge != 'none':
        final_discharge_type = img_discharge
    
    discharge_detected = final_discharge_type not in ['none', '']

    try:
        tissue = classification.wound.tissue_composition or {}
        if not tissue:
            w = wound_type.lower()
            if 'normal' in w or 'healing' in w:
                tissue = {"pink": 70, "red": 30, "yellow": 0, "black": 0, "white": 0}
            elif 'delayed' in w:
                tissue = {"pink": 30, "red": 40, "yellow": 30, "black": 0, "white": 0}
            else:
                tissue = {"pink": 0, "red": 30, "yellow": 30, "black": 40, "white": 0}

        pink  = float(tissue.get('pink', 0))
        red   = float(tissue.get('red', 0))
        yellow = float(tissue.get('yellow', 0))
        black = float(tissue.get('black', 0))
        white = float(tissue.get('white', 0))        # Use the healing results passed from classify_wound or calculate if missing
        healing_details = classification.wound.analysis.get("healing_details", {})
        if not healing_details:
             # Fallback calculation if not stored
             healing_details = calculate_healing_score(
                granulation=red, epithelial=pink, slough=yellow, necrotic=black, white=white,
                surgery_days=int(symptoms.get('daysSinceSurgery', 0)),
                pain_level=symptoms.get('pain_level', 'none'),
                drainage_type=symptoms.get('discharge_type', 'dry'),
                fever=bool(symptoms.get('fever', False)),
                redness_spreading=bool(symptoms.get('redness_spread', False)),
                dressing_changed=bool(symptoms.get('dressingChanged', True))
             )

        severity_score = healing_details.get("final_score", 0)
        severity_level = healing_details.get("stage", "Unknown")

        result = generate_enhanced_local_recommendations(classification, wound_type, severity_level, severity_score, tissue, symptoms)
        
        rec = Recommendation.objects.create(
            classification=classification,
            summary=result.get("summary", ""),
            cleaning_instructions=result.get("cleaningInstructions", []),
            dressing_recommendations=result.get("dressingRecommendations", []),
            warning_signs=result.get("warningsSigns", []),
            when_to_seek_help=result.get("whenToSeekHelp", []),
            diet_advice=result.get("dietAdvice", []),
            activity_restrictions=result.get("activityRestrictions", []),
            expected_healing_time=result.get("expectedHealingTime", ""),
            follow_up_schedule=result.get("followUpSchedule", []),
            ai_confidence=result.get("confidence", 95),
        )

        return {
            "success": True,
            "recommendation_id": rec.id,
            "recommendation": result,
            "risk_level": severity_level,
            "severity_score": float(severity_score),
            "tissue_composition": tissue,
            "is_fallback": False
        }
    except Exception as e:
        print(f"[RECOMMEND] Error: {e}")
        return {"success": False, "error": str(e)}

@api_view(['POST'])
def get_recommendations(request):
    """API wrapper for getting recommendations."""
    classification_id = request.data.get('classification_id')
    wound_type = request.data.get('wound_type', '')
    confidence = float(request.data.get('confidence', 0))
    symptoms = {
        'pain_level': request.data.get('pain_level', 'none'),
        'fever': request.data.get('fever', False),
        'discharge_type': request.data.get('discharge_type', 'none'),
        'redness_spread': request.data.get('redness_spread', False)
    }

    clf = Classification.objects.select_related('wound').filter(id=classification_id).first()
    if not clf:
        return Response({"success": False, "error": "Classification not found"}, status=status.HTTP_404_NOT_FOUND)

    result = get_clinical_recommendations_logic(clf, wound_type, confidence, symptoms)
    if result.get("success"):
        return Response(result)
    return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

