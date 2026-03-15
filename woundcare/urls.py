from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def root(request):
    return Response({
        "message": "Surgical Wound Care API",
        "version": "3.0.0",
        "framework": "Django REST Framework",
        "status": "running",
    })

@api_view(['GET'])
def health(request):
    return Response({
        "status": "healthy",
        "database": "postgresql",
    })

urlpatterns = [
    path('', root),
    path('health', health),
    path('api/', include('api.urls')),
    re_path(r'^uploads/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
