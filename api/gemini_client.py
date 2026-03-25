import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._is_configured = False
        if self.api_key and self.api_key != "your_google_api_key_here":
            genai.configure(api_key=self.api_key)
            self._is_configured = True
        
        self.model_name = "gemini-1.5-flash" # High speed, low cost/free
        self.system_prompt = """
You are a specialized Surgical Wound Care AI Assistant. 
Your goal is to provide helpful, accurate, and concise information strictly related to surgical wound care, recovery, and clinical guidance.

RULES:
1. ONLY answer questions related to wound care, nutrition for recovery, healing stages, infection signs, and surgical aftercare.
2. If a user asks about non-medical topics (e.g., jokes, history, coding, general news), politely refuse and state that you are specialized in wound care assistance.
3. ALWAYS include a disclaimer that your advice is educational and not a substitute for professional medical consultation.
4. If symptoms sound critical (fever over 101F, red streaks, heavy bleeding), tell the user to seek emergency medical attention immediately.
5. Provide specific food recommendations for wound recovery (high protein, Vitamin C, Zinc, etc.) when asked.
"""

    def generate_response(self, prompt):
        # Try to pick up key from environment if not already set (helps if server didn't restart)
        if not self._is_configured:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if self.api_key and self.api_key != "your_google_api_key_here":
                genai.configure(api_key=self.api_key)
                self._is_configured = True

        if not self._is_configured:
            return {
                "success": False,
                "error": "Google API Key is not configured. Please add GOOGLE_API_KEY to your .env file and RESTART your server."
            }

        print(f"\n[GEMINI BRIDGE] Sending prompt to {self.model_name}: {prompt[:50]}...")
        
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt
            )
            
            response = model.generate_content(prompt)
            
            return {
                "success": True,
                "response": response.text,
                "model": self.model_name
            }
        except Exception as e:
            print(f"[GEMINI BRIDGE] Error: {str(e)}")
            return {
                "success": False,
                "error": f"Gemini API error: {str(e)}"
            }

gemini_client = GeminiClient()
