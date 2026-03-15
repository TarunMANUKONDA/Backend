from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.models import Wound

@api_view(['POST'])
def confirm_wound(request, wound_id):
    """Mark a wound report as confirmed/saved by the user."""
    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"success": False, "error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)
    
    wound.is_confirmed = True
    wound.save()
    
    return Response({
        "success": True, 
        "message": f"Wound {wound_id} confirmed and saved to history",
        "is_confirmed": True
    })
