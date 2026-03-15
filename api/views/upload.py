import os
import time
from pathlib import Path

from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    """Upload wound image and save to database."""
    image = request.FILES.get('image')
    if not image:
        return Response({"success": False, "error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

    user_id = int(request.data.get('user_id', 1))
    case_id = request.data.get('case_id')
    case_id = int(case_id) if case_id else None

    # Validate extension
    ext = Path(image.name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return Response({"success": False, "error": f"Invalid file type. Only {', '.join(settings.ALLOWED_EXTENSIONS)} allowed."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Validate size
    contents = image.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        return Response({"success": False, "error": f"File too large. Max {settings.MAX_FILE_SIZE // 1024 // 1024}MB."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filename = f"wound_{int(time.time())}_{os.urandom(4).hex()}{ext}"
    upload_path = os.path.join(settings.UPLOAD_DIR, filename)
    # Relative path stored in DB — used to build URL: /uploads/<filename>
    relative_path = f"uploads/{filename}"
    try:
        with open(upload_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        return Response({"success": False, "error": f"Failed to save image: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Save DB record
    try:
        from api.models import User, Case
        user = User.objects.filter(id=user_id).first()
        case = Case.objects.filter(id=case_id).first() if case_id else None
        wound = Wound.objects.create(
            user=user,
            case=case,
            image_path=relative_path,   # ← store relative path
            original_filename=image.name,
            status='pending',
        )
        return Response({
            "success": True,
            "wound_id": wound.id,
            "image_path": request.build_absolute_uri('/' + relative_path.lstrip('/')),
            "message": "Image uploaded successfully",
        })
    except Exception as e:
        if os.path.exists(upload_path):
            os.remove(upload_path)
        return Response({"success": False, "error": f"Database error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

