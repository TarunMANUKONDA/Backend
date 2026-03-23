import requests
import base64
import os

BASE_URL = "http://127.0.0.1:8000/api/validate"
UPLOADS_DIR = r"c:\Users\vamsi88\project PDD\AI-Powered Surgical Wound Care Tool\backend\uploads"

test_images = [
    ("wound_1774241184_64c6b4df.jpg", "Sutured Wound (Should be VALID)"),
    ("wound_1774242850_c3cbec7e.jpg", "Google Form Screenshot (Should be INVALID)"),
    ("wound_1774247110_19aceb2b.jpg", "Flowchart (Should be INVALID)"),
]

def test_validation():
    for filename, description in test_images:
        path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping {filename}: File not found")
            continue
            
        with open(path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
        print(f"\nTesting: {description}")
        payload = {"image_data": img_b64}
        try:
            response = requests.post(BASE_URL, json=payload)
            result = response.json()
            print(f"Status: {response.status_code}")
            print(f"is_valid: {result.get('is_valid')}")
            print(f"message: {result.get('message')}")
            # print(f"CLIP Confidence: {result.get('wound_confidence')}%")
        except Exception as e:
            print(f"Error testing {filename}: {e}")

if __name__ == "__main__":
    test_validation()
