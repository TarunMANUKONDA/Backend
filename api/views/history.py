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

    qs = Wound.objects.filter(user_id=user_id, is_confirmed=True)
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
            
            summary = ""
            cleaning = []
            warnings = []
            dressing = []
            
            if rec:
                summary = rec.summary
                cleaning = rec.cleaning_instructions
                warnings = rec.warning_signs
                dressing = rec.dressing_recommendations
            
            # REPAIR LOGIC: If historical record is missing or has 'unavailable' message
            if not rec or "unavailable" in (summary or "").lower():
                from .recommend import get_clinical_recommendations_logic
                
                # Extract data for rule engine
                # confidence used as fallback for historical records
                confidence = wound.confidence or 70.0
                tissue = wound.tissue_composition or {}
                w_type = clf.wound_type if clf else (wound.classification or "Wound")
                
                symptoms = {
                    "fever": wound.analysis.get('fever', False) if wound.analysis else False,
                    "discharge_type": wound.discharge_type or "none",
                    "pain_level": wound.analysis.get('pain_level', 'none') if wound.analysis else 'none',
                    "redness_spread": False 
                }
                
                # Generate robust clinical guidance using the core logic
                fallback_result = get_clinical_recommendations_logic(clf, w_type, confidence, symptoms)
                
                if fallback_result.get("success"):
                    fallback = fallback_result.get("recommendation", {})
                    summary = fallback.get("summary", "")
                    cleaning = fallback.get("cleaningInstructions", [])
                    warnings = fallback.get("warningsSigns", [])
                    dressing = fallback.get("dressingRecommendations", [])

            recommendation = {
                "summary": summary,
                "cleaning_instructions": cleaning,
                "dressing_recommendations": dressing,
                "warning_signs": warnings,
            }

        # Normalize path: strip absolute prefix if old records have it,
        # new records already store 'uploads/filename'
        img_path = wound.image_path.replace('\\', '/').replace('\\', '/')
        # Strip any absolute path prefix before 'uploads/'
        if '/uploads/' in img_path:
            img_path = 'uploads/' + img_path.split('/uploads/')[-1]
        elif img_path.startswith('./'):
            img_path = img_path[2:]
        elif not img_path.startswith('uploads/') and not img_path.startswith('http'):
            img_path = 'uploads/' + img_path


        wounds_data.append({
            "wound_id": wound.id,
            "case_id": wound.case_id,
            "image_path": img_path,  # Return relative path for frontend localization

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
    from django.db.models import Count, Q
    cases = Case.objects.filter(user_id=user_id).annotate(
        confirmed_count=Count('wounds', filter=Q(wounds__is_confirmed=True))
    ).filter(confirmed_count__gt=0).order_by('-created_at')

    cases_data = []
    for case in cases:
        wound_count = Wound.objects.filter(case=case, is_confirmed=True).count()
        latest_wound = Wound.objects.filter(case=case, is_confirmed=True).order_by('-upload_date').first()
        latest_score = 0
        latest_risk = 'normal'
        latest_image = None
        latest_tissue = None
        if latest_wound:
            img = latest_wound.image_path.replace('\\', '/')
            if '/uploads/' in img:
                img = 'uploads/' + img.split('/uploads/')[-1]
            elif img.startswith('./'):
                img = img[2:]
            elif not img.startswith('uploads/') and not img.startswith('http'):
                img = 'uploads/' + img
            
            # Return relative path for frontend localization
            latest_image = img

            
            # Extract latest score and risk from saved analysis
            if latest_wound.analysis:
                latest_score = latest_wound.analysis.get('healingScore', latest_wound.analysis.get('overallHealth', 70))
                latest_risk = latest_wound.analysis.get('riskLevel', 'normal')
            
            latest_tissue = latest_wound.tissue_composition or (latest_wound.analysis.get('tissue_composition') if latest_wound.analysis else None)

        cases_data.append({
            "id": case.id,
            "name": case.name,
            "description": case.description,
            "created_at": case.created_at.isoformat(),
            "wound_count": wound_count,
            "latest_image": latest_image,
            "latest_score": latest_score,
            "latest_risk": latest_risk,
            "latest_tissue": latest_tissue,
            "status": case.status,
        })

    return Response({"success": True, "cases": cases_data})


@api_view(['DELETE'])
def delete_wound(request, wound_id):
    """Delete a wound (cascade handled by DB)."""
    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    wound.delete()
    return Response({"success": True, "message": f"Wound {wound_id} deleted"})


@api_view(['DELETE'])
def delete_case(request, case_id):
    """Delete a case and all its wounds (cascade handled by DB)."""
    case = Case.objects.filter(id=case_id).first()
    if not case:
        return Response({"error": "Case not found"}, status=status.HTTP_404_NOT_FOUND)

    case.delete()
    return Response({"success": True, "message": f"Case {case_id} and all its wounds deleted"})
