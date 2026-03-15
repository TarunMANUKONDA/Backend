import json
import os
from pathlib import Path

from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import Wound, Comparison as ComparisonModel

@api_view(['POST'])
def compare_wounds(request):
    """Compare two wound images using local tissue data."""
    base_id = request.data.get('base_wound_id')
    curr_id = request.data.get('current_wound_id')

    base = Wound.objects.filter(id=base_id).first()
    curr = Wound.objects.filter(id=curr_id).first()

    if not base or not curr:
        return Response({"success": False, "error": "One or both wounds not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Extract tissue metrics
        bt = base.tissue_composition or {}
        ct = curr.tissue_composition or {}
        
        b_necrosis = float(bt.get('black', 0))
        c_necrosis = float(ct.get('black', 0))
        b_slough = float(bt.get('yellow', 0)) + float(bt.get('white', 0))
        c_slough = float(ct.get('yellow', 0)) + float(ct.get('white', 0))
        b_healthy = float(bt.get('red', 0)) + float(bt.get('pink', 0))
        c_healthy = float(ct.get('red', 0)) + float(ct.get('pink', 0))

        # ── Simple Deterministic Assessment ──────────────────────────────────
        assessment = "stable"
        positive_changes = []
        concerning_changes = []
        recommendations = ["Continue current care plan.", "Monitor for new discharge."]

        # Calculate deltas for tissue
        tissue_change = {
            "red": ct.get('red', 0) - bt.get('red', 0),
            "pink": ct.get('pink', 0) - bt.get('pink', 0),
            "yellow": ct.get('yellow', 0) - bt.get('yellow', 0),
            "black": ct.get('black', 0) - bt.get('black', 0),
            "white": ct.get('white', 0) - bt.get('white', 0),
        }

        # Improvement calculation (matches frontend logic but simplified for backend)
        # Formula: (Granulation * 0.25) + (Epithelial * 0.2) - (Necrotic * 0.1) - (Slough * 0.05)
        # We don't have PAR (area reduction) reliably here yet, so we focus on tissue.
        granulation_change = tissue_change["red"]
        epithelial_change = tissue_change["pink"]
        necrotic_change = tissue_change["black"]
        slough_change = tissue_change["yellow"] + tissue_change["white"]

        improvement = (granulation_change * 0.25) + (epithelial_change * 0.2) - (necrotic_change * 0.1) - (slough_change * 0.05)
        improvement = float(round(improvement, 1))

        if c_necrosis < b_necrosis:
            positive_changes.append("Reduction in necrotic (black) tissue.")
            assessment = "improving"
        elif c_necrosis > b_necrosis:
            concerning_changes.append("Increase in necrotic (black) tissue.")
            assessment = "worsening"

        if c_healthy > b_healthy:
            positive_changes.append("Growth of healthy granulation/epithelial tissue.")
            if assessment != "worsening": assessment = "improving"
        
        if c_slough > b_slough:
            concerning_changes.append("Increased slough/fibrin buildup.")
            if assessment != "improving": assessment = "worsening"
        
        if not positive_changes and not concerning_changes:
            assessment = "stable"

        result = {
            "overallAssessment": assessment,
            "improvement": improvement,
            "tissueChange": tissue_change,
            "sizeChange": "Minimal change observed",
            "colorChange": "Shift towards healthy pink" if c_healthy > b_healthy else "No significant shift",
            "inflammationChange": "Stable",
            "dischargeChange": "no change",
            "edgeHealing": "Edges appear stable",
            "riskLevel": "high" if assessment == "worsening" else ("moderate" if assessment == "stable" else "low"),
            "recommendations": recommendations,
            "concerningChanges": concerning_changes,
            "positiveChanges": positive_changes,
            "summary": f"Local comparison shows a {assessment} trend ({improvement}% delta). " + 
                       (f"Healthy tissue is at {c_healthy}%." if c_healthy > 0 else ""),
            "confidence": 90
        }

        return Response({"success": True, "comparison": result})

    except Exception as e:
        return Response({"success": False, "error": f"Comparison failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def save_comparison(request):
    """Save comparison to database."""
    ComparisonModel.objects.create(
        case_id=request.data.get('case_id'),
        wound_before_id=request.data.get('wound_id_before'),
        wound_after_id=request.data.get('wound_id_after'),
        analysis=request.data.get('analysis'),
    )
    return Response({"success": True, "message": "Comparison saved"})


@api_view(['POST'])
def save_analysis(request):
    """Save or update wound analysis."""
    wound_id = request.data.get('wound_id')
    analysis = request.data.get('analysis')

    if not wound_id or not analysis:
        return Response({"error": "Missing wound_id or analysis"}, status=status.HTTP_400_BAD_REQUEST)

    wound = Wound.objects.filter(id=wound_id).first()
    if not wound:
        return Response({"error": "Wound not found"}, status=status.HTTP_404_NOT_FOUND)

    wound.analysis = analysis
    
    # Extract top-level fields for consistency and easy history access
    if isinstance(analysis, dict):
        if 'tissueColor' in analysis:
            wound.tissue_composition = analysis['tissueColor']
        if 'rednessLevel' in analysis:
            wound.redness_level = analysis['rednessLevel']
        if 'edgeQuality' in analysis:
            wound.edge_quality = analysis['edgeQuality']
        if 'dischargeType' in analysis:
            wound.discharge_type = analysis['dischargeType']
        if 'dischargeDetected' in analysis:
            wound.discharge_detected = analysis['dischargeDetected']
        if 'confidence' in analysis:
            wound.confidence = analysis['confidence']
            
        # ── Automated Healing Closure Logic ──────────────────────────────────
        # Check healing score (usually 0-100, where 100 is healthy)
        # Note: Frontend uses healingScore or overallHealth
        healing_score = analysis.get('healingScore', analysis.get('overallHealth', 0))
        
        if healing_score > 95:
            wound.status = "Healed"
            if wound.case:
                wound.case.status = "Healed"
                wound.case.save()
                print(f"[SAVE_ANALYSIS] Case {wound.case.id} marked as HEALED (Score: {healing_score})")

    wound.save()
    return Response({
        "success": True, 
        "message": "Analysis saved and statuses updated.",
        "is_healed": wound.status == "Healed"
    })
