import os
import django
import json
import sys

# Setup Django environment
sys.path.append(r'c:\Users\vamsi88\project PDD\AI-Powered Surgical Wound Care Tool\backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'woundcare.settings')
django.setup()

from api.models import Wound, User, Case

def test_analysis_persistence():
    print("Testing analysis persistence...")
    
    # Create a dummy user and case if they don't exist
    user, _ = User.objects.get_or_create(email="test@example.com", defaults={"name": "Test User"})
    case, _ = Case.objects.get_or_create(user=user, name="Test Case")
    
    # Create a dummy wound
    wound = Wound.objects.create(
        user=user,
        case=case,
        image_path="test/path.jpg",
        status="pending"
    )
    
    # Simulated classification results
    tissue_composition = {"red": 40, "pink": 30, "yellow": 20, "black": 5, "white": 5}
    healing_results = {
        "final_score": 75.5,
        "healingScore": 75.5,
        "statusLabel": "Healing Well",
        "stage": "Healing Well",
        "progress": "Improving"
    }
    severity_level = "Moderate"
    tissue_type = "Delayed Healing"
    wound_area = 5000
    lab_metrics_clean = {"granulation": 40, "slough": 20}
    
    # This matches the code in classify.py
    wound.analysis = {
        "source": "lab_cv_segmentation_fusion",
        "tissue_composition": tissue_composition,
        "lab_metrics": lab_metrics_clean,
        "wound_area_pixels": wound_area,
        "healing_details": healing_results,
        "healingScore": healing_results.get("healingScore", 75.5),
        "severityLevel": severity_level,
        "severityLabel": tissue_type,
        "notes": "Test notes"
    }
    wound.save()
    
    # Refresh from DB
    wound.refresh_from_db()
    
    # Verify
    print(f"Analysis: {json.dumps(wound.analysis, indent=2)}")
    
    expected_keys = ["healingScore", "severityLevel", "severityLabel"]
    missing_keys = [k for k in expected_keys if k not in wound.analysis]
    
    if not missing_keys:
        print("SUCCESS: All expected keys found in wound.analysis!")
    else:
        print(f"FAILURE: Missing keys in wound.analysis: {missing_keys}")
        sys.exit(1)

    # Clean up
    wound.delete()
    print("Cleanup complete.")

if __name__ == "__main__":
    test_analysis_persistence()
