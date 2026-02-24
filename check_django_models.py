import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'woundcare.settings')
django.setup()

import google.generativeai as genai

print(f"Using settings.GEMINI_API_KEY: {settings.GEMINI_API_KEY[:10]}...")
genai.configure(api_key=settings.GEMINI_API_KEY)

print("\nAvailable models in Django env:")
try:
    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for m in models:
        print(f"- {m}")
except Exception as e:
    print(f"Error listing models: {e}")
