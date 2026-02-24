import json
import base64
import tempfile
import os
from io import BytesIO

import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import google.generativeai as genai

def log_debug(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    with open("debug_validate.log", "a", encoding='utf-8') as f:
        f.write(log_line)
    print(log_line)
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

genai.configure(api_key=settings.GEMINI_API_KEY)

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

MODEL_NAMES = [
    'models/gemini-flash-latest',
    'models/gemini-2.0-flash',
    'models/gemini-pro-latest',
]

VALIDATE_PROMPT = """Analyze this image with extreme lenience to identify ANY evidence of a medical or surgical procedure.

CRITICAL REQUIREMENT:
You MUST classify the following as an ACTIVE WOUND (has_wound: true):
1. ANY surgical incision, even if it is clean, straight, and neatly closed.
2. ANY site with stitches, sutures, staples, or surgical tape/strips.
3. ANY recent surgical scar that still appears pink, raised, or has visible marks from needles/staples.
4. ANY dressing, bandage, or medical tape that covers a suspected wound area.

Only classify has_wound: false if the skin is perfectly intact, healthy, and shows absolutely no signs of recent medical intervention or injury.

OUTPUT FORMAT (JSON only):
{
  "has_wound": boolean,
  "is_healed": boolean, (true only for mature, flat, old white scars)
  "confidence": number, (percentage 0-100)
  "explanation": "Brief reasoning for your classification"
}
"""


def calculate_blur_score(image_array: np.ndarray) -> float:
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_wound_with_gemini(image_data: str):
    try:
        log_debug("Attempting wound detection...")
        try:
            available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            log_debug(f"Available models according to genai: {available}")
        except Exception as e:
            log_debug(f"Error listing models inside view: {e}")

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        try:
            uploaded_file = genai.upload_file(temp_path)
            response = None
            last_error = None

            for m_name in MODEL_NAMES:
                try:
                    model = genai.GenerativeModel(m_name)
                    response = model.generate_content(
                        [uploaded_file, VALIDATE_PROMPT],
                        generation_config={"response_mime_type": "application/json"},
                        safety_settings=SAFETY_SETTINGS
                    )
                    if response:
                        # Check if response was blocked
                        if not response.candidates:
                            print(f"DEBUG: Response from {m_name} was blocked by safety filters.")
                            continue
                        break
                except Exception as e:
                    print(f"DEBUG: Model {m_name} failed: {e}")
                    last_error = e
                    continue

            if not response:
                raise last_error or Exception("All models failed")

            response_text = response.text.strip()
            log_debug(f"Raw Gemini Response: {response_text}")
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                log_debug("JSON parsing failed, attempting extraction")
                s = response_text.find("{")
                e = response_text.rfind("}") + 1
                result = json.loads(response_text[s:e])

            log_debug(f"Parsed Gemini Result: {result}")

            has_wound = result.get("has_wound", False)
            if isinstance(has_wound, str): has_wound = has_wound.lower() == 'true'
            is_healed = result.get("is_healed", False)
            if isinstance(is_healed, str): is_healed = is_healed.lower() == 'true'

            conf_val = result.get("confidence", 0)
            try:
                confidence = float(str(conf_val).replace('%', ''))
            except:
                confidence = 0

            explanation = result.get("explanation", "No explanation provided")
            log_debug(f"Final Detection Results -> has_wound={has_wound}, is_healed={is_healed}, confidence={confidence}")
            return has_wound, is_healed, confidence, explanation

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"Gemini wound detection error: {e}")
        return False, False, 0, f"Error during detection: {str(e)}"


@api_view(['POST'])
def validate_image(request):
    """Validate image for blur and wound detection."""
    try:
        key = getattr(settings, 'GEMINI_API_KEY', '')
        log_debug(f"Validating with key: {key[:10]}...")
        genai.configure(api_key=key)

        image_data = request.data.get('image_data', '')

        if ',' in image_data:
            image_data_b64 = image_data.split(',')[1]
        else:
            image_data_b64 = image_data

        image_bytes = base64.b64decode(image_data_b64)
        pil_image = Image.open(BytesIO(image_bytes))
        image_array = np.array(pil_image)

        blur_score = calculate_blur_score(image_array)
        BLUR_THRESHOLD = 50.0
        is_blur = blur_score < BLUR_THRESHOLD

        has_wound, is_healed, wound_confidence, explanation = detect_wound_with_gemini(request.data.get('image_data', ''))
        WOUND_CONFIDENCE_THRESHOLD = 15.0

        log_debug(f"Validation Debug: Blur={blur_score:.2f}, Wound={has_wound}, Healed={is_healed}, Conf={wound_confidence}%")

        # More lenient detection: lower confidence threshold and consider healed wounds as valid
        wound_detected = has_wound and (wound_confidence >= WOUND_CONFIDENCE_THRESHOLD)
        is_valid = (not is_blur) and wound_detected

        if is_blur:
            message = "Image is too blurry. Please retake the photo clearly."
        elif not wound_detected:
            message = "Wound is not detected. Please capture the wound area clearly."
        elif is_healed:
            message = "Healed wound detected."
            is_valid = True
        else:
            message = "Wound detected successfully."

        return Response({
            "success": True,
            "is_valid": is_valid,
            "is_blur": is_blur,
            "has_wound": has_wound,
            "is_healed": is_healed,
            "blur_score": blur_score,
            "wound_confidence": wound_confidence,
            "message": message,
        })

    except Exception as e:
        return Response({"success": False, "error": f"Validation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
