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
from api import tflite_classifier
from api.views.classify import classify_wound

# I'll need to refactor recommend.py or just copy the logic for now to keep it fast.
# Given the urgency, I'll implement a helper that contains the core recommendation logic.

def generate_full_analysis(request, wound_id, symptoms=None):
    """
    Performs full local analysis:
    1. TFLite + CLIP Tissue Classification
    2. Rule-based Recommendations
    """
    # Simply reuse the logic from classify_wound
    from rest_framework.request import Request
    from rest_framework.parsers import JSONParser
    
    # Mock a request for classify_wound
    factory_request = Request(request._request)
    factory_request.data.update({
        "wound_id": wound_id,
        "clinical_data": symptoms # This is the React PatientAnswers object
    })
    
    classify_response = classify_wound(factory_request)
    if not classify_response.data.get("success"):
        return classify_response.data

    data = classify_response.data
    clf = Classification.objects.get(id=data["classification_id"])
    
    # Generate Recommendations
    from api.views.recommend import get_clinical_recommendations_logic
    rec_result = get_clinical_recommendations_logic(clf, data["wound_type"], data["confidence"], symptoms)
    
    data["recommendation"] = rec_result.get("recommendation") if rec_result.get("success") else None
    data["risk_level"] = rec_result.get("risk_level", "Unknown")
    data["severity_score"] = rec_result.get("severity_score", 0)
    
    return data

@api_view(['POST'])
def analyze_full(request):
    """Unified endpoint for full wound analysis."""
    wound_id = request.data.get('wound_id')
    symptoms = request.data.get('symptoms', {})

    if not wound_id:
        return Response({"success": False, "error": "wound_id required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        result = generate_full_analysis(request, wound_id, symptoms)
        return Response(result)
    except Exception as e:
        return Response({"success": False, "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
