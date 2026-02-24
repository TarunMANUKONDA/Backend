import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    # try frontend .env just in case
    load_dotenv('../frontend/.env')
    api_key = os.getenv('VITE_GEMINI_API_KEY')

print(f"Using API Key: {api_key[:10]}...")
genai.configure(api_key=api_key)

print("\nAvailable models:")
try:
    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for m in models:
        print(f"Testing model: {m}")
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content("Hi")
            print(f"  Success: {response.text[:20]}...")
            break # Just need to find one working one
        except Exception as e:
            print(f"  Failed: {e}")
except Exception as e:
    print(f"Error listing models: {e}")
