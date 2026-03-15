import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434"):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.model = "llama3.2"
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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "response": data.get("response", ""),
                "model": data.get("model", "")
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"AI Bridge connection error: {str(e)}"
            }

ollama_client = OllamaClient()
