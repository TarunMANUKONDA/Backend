from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.models import Wound, Classification, Recommendation

@api_view(['POST'])
def confirm_wound(request, wound_id):
    """
    Mark a wound report as confirmed/saved by the user.
    This also persists the classification and recommendations to the DB.
    """

    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"success": False, "error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)
    
    data = request.data # Expects the full analysis payload from the frontend
    
    try:
        # 1. Create Classification (or update if already exists from internal call)
        # We use update_or_create to prevent duplicate classifications for the same wound
        clf_data = {
            "wound_type": data.get("wound_type", "Wound"),
            "confidence": data.get("confidence", 0),
            "all_probabilities": data.get("probabilities", {}),
            "processing_time_ms": data.get("processing_time_ms", 0)
        }
        clf, _ = Classification.objects.update_or_create(wound=wound, defaults=clf_data)
        
        # 2. Create Recommendation
        rec_data = data.get("recommendation", {})
        if rec_data:
            # Clear old recommendations for this classification to prevent duplicates
            Recommendation.objects.filter(classification=clf).delete()
            Recommendation.objects.create(
                classification=clf,
                summary=rec_data.get("summary", ""),
                cleaning_instructions=rec_data.get("cleaningInstructions", []),
                dressing_recommendations=rec_data.get("dressingRecommendations", []),
                warning_signs=rec_data.get("warningsSigns", []),
                when_to_seek_help=rec_data.get("whenToSeekHelp", []),
                diet_advice=rec_data.get("dietAdvice", []),
                activity_restrictions=rec_data.get("activityRestrictions", []),
                expected_healing_time=rec_data.get("expectedHealingTime", ""),
                follow_up_schedule=rec_data.get("followUpSchedule", []),
                ai_confidence=rec_data.get("confidence", 95)
            )
            
        # 3. Update Wound Top-Level Metrics
        wound.status = "analyzed"
        wound.classification = data.get("wound_type", wound.classification)
        wound.confidence = data.get("confidence", wound.confidence)
        
        # Robust Mapping: Frontend uses tissueColor, Backend model uses tissue_composition
        tissue = data.get("tissue_composition") or data.get("tissueColor")
        if tissue:
            wound.tissue_composition = tissue
            
        # Persist standard metrics if provided
        if "rednessLevel" in data: wound.redness_level = data["rednessLevel"]
        if "dischargeDetected" in data: wound.discharge_detected = data["dischargeDetected"]
        if "dischargeType" in data: wound.discharge_type = data["dischargeType"]
        if "edgeQuality" in data: wound.edge_quality = data["edgeQuality"]
        
        # Prepare analysis JSON for history (Flattened for easy consumption)
        healing_details = data.get("healing_details", {})
        tissue_comp = tissue or {}
        
        # Extract critical metrics for top-level persistence in analysis JSON
        healing_score = data.get("healingScore") or healing_details.get("healingScore") or data.get("severity_score")
        risk_level = data.get("riskLevel") or data.get("risk_level")
        severity_level = data.get("severityLevel") or data.get("severity_level")
        severity_label = data.get("severityLabel") or data.get("severity_label")

        wound.analysis = {
            "source": "manual_confirmation_persistence",
            "tissue_composition": tissue_comp,
            "wound_area_pixels": data.get("wound_area_pixels") or healing_details.get("wound_area") or data.get("woundSize", {}).get("area"),
            "healing_details": healing_details,
            "healingScore": healing_score,
            "riskLevel": risk_level,
            "severityLevel": severity_level,
            "severityLabel": severity_label,
            "notes": f"Saved via User Confirmation. Stage: {healing_details.get('stage') or 'N/A'}."
        }
        
        wound.is_confirmed = True
        wound.save()
        
        return Response({
            "success": True, 
            "message": f"Wound {wound_id} analysis confirmed and saved to history",
            "is_confirmed": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({"success": False, "error": f"Persistence failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

