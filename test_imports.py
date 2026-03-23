import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import django
    print(f"Django version: {django.get_version()}")
except ImportError as e:
    print(f"Django import failed: {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
except Exception as e:
    print(f"TensorFlow loaded but failed: {e}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Torch import failed: {e}")
except Exception as e:
    print(f"Torch loaded but failed: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV import failed: {e}")
except Exception as e:
    print(f"OpenCV loaded but failed: {e}")
