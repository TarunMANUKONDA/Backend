from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Classification, Recommendation, Case


@api_view(['GET'])
def get_history(request):
    """Get wound history with classifications and recommendations."""
    user_id = int(request.query_params.get('user_id', 1))
    case_id = request.query_params.get('case_id')
    limit = int(request.query_params.get('limit', 50))
    offset = int(request.query_params.get('offset', 0))

    qs = Wound.objects.filter(user_id=user_id)
    if case_id:
        qs = qs.filter(case_id=int(case_id))

    total = qs.count()
    wounds = qs.order_by('-upload_date')[offset:offset + limit]

    wounds_data = []
    for wound in wounds:
        clf = Classification.objects.filter(wound=wound).first()
        recommendation = None
        if clf:
            rec = Recommendation.objects.filter(classification=clf).first()
            if rec:
                recommendation = {
                    "summary": rec.summary,
                    "cleaning_instructions": rec.cleaning_instructions,
                    "dressing_recommendations": rec.dressing_recommendations,
                    "warning_signs": rec.warning_signs,
                }

        # Normalize path: strip absolute prefix if old records have it,
        # new records already store 'uploads/filename'
        img_path = wound.image_path.replace('\\', '/').replace('\\', '/')
        # Strip any absolute path prefix before 'uploads/'
        if '/uploads/' in img_path:
            img_path = 'uploads/' + img_path.split('/uploads/')[-1]
        elif img_path.startswith('./'):
            img_path = img_path[2:]

        wounds_data.append({
            "wound_id": wound.id,
            "case_id": wound.case_id,
            "image_path": img_path,
            "original_filename": wound.original_filename,
            "upload_date": wound.upload_date.isoformat(),
            "status": wound.status,
            "analysis": wound.analysis,
            "redness_level": wound.redness_level,
            "discharge_detected": wound.discharge_detected,
            "discharge_type": wound.discharge_type,
            "edge_quality": wound.edge_quality,
            "tissue_composition": wound.tissue_composition,
            "classification": {
                "classification_id": clf.id,
                "wound_type": clf.wound_type,
                "confidence": clf.confidence,
                "probabilities": clf.all_probabilities,
            } if clf else (
                {"wound_type": wound.classification, "confidence": wound.confidence}
                if wound.classification else None
            ),
            "recommendation": recommendation,
        })

    return Response({"success": True, "wounds": wounds_data, "total": total, "limit": limit, "offset": offset})


@api_view(['POST'])
def create_case(request):
    """Create a new wound case."""
    user_id = int(request.data.get('user_id', 1))
    name = request.data.get('name', '')
    description = request.data.get('description')

    from api.models import User
    user = User.objects.filter(id=user_id).first()
    if not user:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    case = Case.objects.create(user=user, name=name, description=description)
    return Response({
        "success": True,
        "case": {
            "id": case.id,
            "user_id": case.user_id,
            "name": case.name,
            "description": case.description,
            "created_at": case.created_at.isoformat(),
        }
    })


@api_view(['GET'])
def get_cases(request):
    """Get all cases for a user."""
    user_id = int(request.query_params.get('user_id', 1))
    cases = Case.objects.filter(user_id=user_id).order_by('-created_at')

    cases_data = []
    for case in cases:
        wound_count = Wound.objects.filter(case=case).count()
        latest_wound = Wound.objects.filter(case=case).order_by('-upload_date').first()
        latest_score = 0
        latest_risk = 'normal'
        latest_image = None
        if latest_wound:
            img = latest_wound.image_path.replace('\\', '/')
            if '/uploads/' in img:
                img = 'uploads/' + img.split('/uploads/')[-1]
            elif img.startswith('./'):
                img = img[2:]
            latest_image = img
            
            # Extract latest score and risk from saved analysis
            if latest_wound.analysis:
                latest_score = latest_wound.analysis.get('healingScore', latest_wound.analysis.get('overallHealth', 70))
                latest_risk = latest_wound.analysis.get('riskLevel', 'normal')

        cases_data.append({
            "id": case.id,
            "name": case.name,
            "description": case.description,
            "created_at": case.created_at.isoformat(),
            "wound_count": wound_count,
            "latest_image": latest_image,
            "latest_score": latest_score,
            "latest_risk": latest_risk,
        })

    return Response({"success": True, "cases": cases_data})


@api_view(['DELETE'])
def delete_wound(request, wound_id):
    """Delete a wound and its child records."""
    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    # Cascade manually
    for clf in Classification.objects.filter(wound=wound):
        Recommendation.objects.filter(classification=clf).delete()
    Classification.objects.filter(wound=wound).delete()
    wound.delete()
    return Response({"success": True, "message": f"Wound {wound_id} deleted"})


@api_view(['DELETE'])
def delete_case(request, case_id):
    """Delete a case and all its wounds."""
    case = Case.objects.filter(id=case_id).first()
    if not case:
        return Response({"error": "Case not found"}, status=status.HTTP_404_NOT_FOUND)

    for wound in Wound.objects.filter(case=case):
        for clf in Classification.objects.filter(wound=wound):
            Recommendation.objects.filter(classification=clf).delete()
        Classification.objects.filter(wound=wound).delete()
        wound.delete()

    case.delete()
    return Response({"success": True, "message": f"Case {case_id} and all its wounds deleted"})
