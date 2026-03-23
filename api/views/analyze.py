import time
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Classification, Recommendation
from api.views.classify import _classify_wound_internal

def generate_full_analysis(request, wound_id, symptoms=None):
    """
    Performs full local analysis WITHOUT saving to DB:
    1. CLIP + Heuristic Tissue Classification (Preview)
    2. Rule-based Recommendations (Preview)
    """
    from api.models import Wound
    from api.views.classify import _classify_wound_internal
    from rest_framework.request import Request
    
    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return {"success": False, "error": "Wound not found"}

    # Mock a request for _classify_wound_internal
    factory_request = Request(request._request)
    factory_request.data.update({
        "wound_id": wound_id,
        "clinical_data": symptoms
    })
    
    # Run classification in preview mode (no DB save)
    data = _classify_wound_internal(factory_request, save_to_db=False)
    if not data.get("success"):
        return data

    # Prepare symptoms with tissue/healing context for recommendation logic
    # since we haven't saved them to the wound object yet
    rec_symptoms = (symptoms or {}).copy()
    rec_symptoms['tissue_composition'] = data.get('tissue_composition')
    rec_symptoms['healing_details'] = data.get('healing_details')
    
    # Generate Recommendations in preview mode (no DB save)
    from api.views.recommend import get_clinical_recommendations_logic
    rec_result = get_clinical_recommendations_logic(
        wound, 
        data["wound_type"], 
        data["confidence"], 
        rec_symptoms, 
        save_to_db=False
    )

    
    data["recommendation"] = rec_result.get("recommendation") if rec_result.get("success") else None
    data["risk_level"] = rec_result.get("risk_level", "Unknown")
    data["severity_score"] = rec_result.get("severity_score", 0)
    
    return data

@api_view(['POST'])
def analyze_full(request):
    """Unified endpoint for full wound analysis (Preview Mode)."""

    wound_id = request.data.get('wound_id')
    symptoms = request.data.get('symptoms', {})

    if not wound_id:
        return Response({"success": False, "error": "wound_id required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        result = generate_full_analysis(request, wound_id, symptoms)
        return Response(result)
    except Exception as e:
        import traceback
        traceback.print_exc()

        return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
