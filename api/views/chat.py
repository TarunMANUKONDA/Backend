from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.ollama_client import ollama_client
from api.gemini_client import gemini_client
import os

@api_view(['POST'])
def ollama_chat(request):
    """Bridge endpoint for AI Assistant."""
    prompt = request.data.get('prompt', '')
    if not prompt:
        return Response({"success": False, "error": "No prompt provided"}, status=status.HTTP_400_BAD_REQUEST)

    # Choose provider based on environment variable
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "gemini":
        result = gemini_client.generate_response(prompt)
    else:
        result = ollama_client.generate_response(prompt)

    if result.get("success"):
        return Response(result)
    else:
        # If AI bridge fails, we provide a local fallback or error message
        return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
