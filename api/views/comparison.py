import json
import os
from pathlib import Path

import google.generativeai as genai
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Comparison as ComparisonModel

genai.configure(api_key=settings.GEMINI_API_KEY)

COMPARE_PROMPT = """Compare these two surgical wound images (first = baseline, second = current state).

Provide analysis in JSON format:
{
  "overallAssessment": "improving/stable/worsening",
  "healingProgress": 0-100,
  "sizeChange": "percentage change",
  "colorChange": "description",
  "inflammationChange": "increased/decreased/stable",
  "dischargeChange": "improved/worsened/no change",
  "edgeHealing": "description",
  "riskLevel": "low/moderate/high",
  "recommendations": ["array"],
  "concerningChanges": ["array"],
  "positiveChanges": ["array"],
  "summary": "overall summary",
  "confidence": 0-100
}"""


@api_view(['POST'])
def compare_wounds(request):
    """Compare two wound images using Gemini Vision API."""
    base_id = request.data.get('base_wound_id')
    curr_id = request.data.get('current_wound_id')

    base = Wound.objects.filter(id=base_id).first()
    curr = Wound.objects.filter(id=curr_id).first()

    if not base or not curr:
        return Response({"success": False, "error": "One or both wounds not found"}, status=status.HTTP_404_NOT_FOUND)

    def resolve_path(wound):
        p = wound.image_path
        if not os.path.isabs(p):
            p = os.path.join(settings.BASE_DIR, p)
        return p

    base_path = resolve_path(base)
    curr_path = resolve_path(curr)

    if not Path(base_path).exists() or not Path(curr_path).exists():
        return Response({"success": False, "error": "One or both image files not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        base_file = genai.upload_file(base_path)
        curr_file = genai.upload_file(curr_path)

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([base_file, curr_file, COMPARE_PROMPT])
        response_text = response.text.strip()

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return Response({"success": True, "comparison": result})

    except json.JSONDecodeError as e:
        return Response({"success": False, "error": f"Failed to parse AI response: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({"success": False, "error": f"Comparison failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def save_comparison(request):
    """Save comparison to database."""
    ComparisonModel.objects.create(
        case_id=request.data.get('case_id'),
        wound_before_id=request.data.get('wound_id_before'),
        wound_after_id=request.data.get('wound_id_after'),
        analysis=request.data.get('analysis'),
    )
    return Response({"success": True, "message": "Comparison saved"})


@api_view(['POST'])
def save_analysis(request):
    """Save or update wound analysis."""
    wound_id = request.data.get('wound_id')
    analysis = request.data.get('analysis')

    if not wound_id or not analysis:
        return Response({"error": "Missing wound_id or analysis"}, status=status.HTTP_400_BAD_REQUEST)

    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    wound.analysis = analysis
    wound.save()
    return Response({"success": True, "message": "Analysis saved"})
