import json
import time
import os
from pathlib import Path

import google.generativeai as genai
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Classification

genai.configure(api_key=settings.GEMINI_API_KEY)

MODEL_NAMES = [
    'models/gemini-2.5-flash',
    'models/gemini-flash-latest',
    'models/gemini-2.0-flash-lite',
    'models/gemini-3-flash-preview',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-flash-8b',
]

CLASSIFY_PROMPT = """Act as a specialized Wound Care AI. Calculate the TISSUE COMPOSITION with extreme cynicism.

CRITICAL: DO NOT HALLUCINATE HEALTHY TISSUE.

TISSUE DEFINITIONS:
1. RED (Granulation): MUST be beefy red, bloody, moist, bumpy. Pale red = SLOUGH.
2. PINK (Epithelial): ONLY at wound edges (silvery/translucent). FORBIDDEN in centre. Pale/whitish/cream/yellowish = SLOUGH.
3. BLACK (Necrotic): Hard, dry, leathery, dark brown/black.
4. YELLOW/WHITE (Slough): Anything NOT clearly Beefy Red or Dead Black.

Provide response in this exact JSON:
{
  "wound_type": "category name",
  "confidence": 0-100,
  "probabilities": {"Normal Healing": 0-100, "Delayed Healing": 0-100, "Infection Risk": 0-100, "Active Infection": 0-100, "High Urgency": 0-100},
  "redness_level": 0-100,
  "discharge_detected": true/false,
  "discharge_type": "none/clear/yellow/green/bloody",
  "edge_quality": 0-100,
  "tissue_composition": {"red": 0-100, "pink": 0-100, "yellow": 0-100, "black": 0-100, "white": 0-100},
  "wound_location": "description",
  "notes": "explanation"
}"""


@api_view(['POST'])
def classify_wound(request):
    """Classify wound using Gemini Vision API."""
    wound_id = request.data.get('wound_id')
    if not wound_id:
        return Response({"success": False, "error": "wound_id required"}, status=status.HTTP_400_BAD_REQUEST)

    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"success": False, "error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    # Resolve actual disk path from stored relative or absolute path
    img_path = wound.image_path
    if not os.path.isabs(img_path):
        img_path = os.path.join(settings.BASE_DIR, img_path)
    if not Path(img_path).exists():
        return Response({"success": False, "error": f"Image file not found: {img_path}"}, status=status.HTTP_404_NOT_FOUND)

    try:
        start_time = time.time()
        uploaded_file = genai.upload_file(img_path)

        response = None
        last_error = None
        for m_name in MODEL_NAMES:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(
                    [uploaded_file, CLASSIFY_PROMPT],
                    generation_config={"response_mime_type": "application/json"}
                )
                if response:
                    break
            except Exception as e:
                last_error = e
                continue

        if not response:
            raise last_error or Exception("All Gemini models failed")

        processing_time = int((time.time() - start_time) * 1000)
        response_text = response.text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            s = response_text.find("{")
            e = response_text.rfind("}") + 1
            result = json.loads(response_text[s:e])

        # Save classification
        clf = Classification.objects.create(
            wound=wound,
            wound_type=result.get("wound_type", "Unknown"),
            confidence=result.get("confidence", 0),
            all_probabilities=result.get("probabilities", {}),
            processing_time_ms=processing_time,
        )

        # Update wound cache
        wound.status = "analyzed"
        wound.classification = result.get("wound_type")
        wound.confidence = result.get("confidence")
        wound.redness_level = result.get("redness_level")
        wound.discharge_detected = result.get("discharge_detected")
        wound.discharge_type = result.get("discharge_type")
        wound.edge_quality = result.get("edge_quality")
        wound.tissue_composition = result.get("tissue_composition")
        wound.analysis = result
        wound.save()

        # Deterministic override
        t = result.get("tissue_composition", {})
        total_slough = t.get("yellow", 0) + t.get("white", 0)
        total_necrosis = t.get("black", 0)

        final_type = result.get("wound_type", "Unknown")
        final_probs = result.get("probabilities", {})

        if total_necrosis >= 10:
            final_type = "High Urgency"; final_probs["High Urgency"] = 100
        elif total_slough >= 20:
            if result.get("discharge_detected") and result.get("discharge_type") in ["yellow", "green"]:
                final_type = "Active Infection"; final_probs["Active Infection"] = 90
            else:
                final_type = "Delayed Healing"; final_probs["Delayed Healing"] = 90
        elif total_slough >= 5 and "Normal" in final_type:
            final_type = "Delayed Healing"; final_probs["Delayed Healing"] = 80

        return Response({
            "success": True,
            "classification_id": clf.id,
            "wound_type": final_type,
            "confidence": result.get("confidence"),
            "probabilities": final_probs,
            "redness_level": result.get("redness_level"),
            "discharge_detected": result.get("discharge_detected"),
            "discharge_type": result.get("discharge_type"),
            "edge_quality": result.get("edge_quality"),
            "tissue_composition": result.get("tissue_composition"),
            "wound_location": result.get("wound_location"),
            "processing_time_ms": processing_time,
        })

    except json.JSONDecodeError as e:
        return Response({"success": False, "error": f"Failed to parse AI response: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({"success": False, "error": f"Classification failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
