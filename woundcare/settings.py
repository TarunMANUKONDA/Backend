import os
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse as _urlparse

# ─────────────────────────────────────────────────────────────
# Base directory
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env (works locally; Render uses env variables)
load_dotenv(BASE_DIR / ".env")

# ─────────────────────────────────────────────────────────────
# Security
# ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-this-in-production"
)

DEBUG = os.getenv("DEBUG", "False") == "True"

ALLOWED_HOSTS = [
    "*",
    ".onrender.com",
]

# ─────────────────────────────────────────────────────────────
# Installed Apps
# ─────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",

    "rest_framework",
    "corsheaders",

    "api",
]

# ─────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
]

# ─────────────────────────────────────────────────────────────
# URLs / WSGI
# ─────────────────────────────────────────────────────────────
ROOT_URLCONF = "woundcare.urls"
WSGI_APPLICATION = "woundcare.wsgi.application"

# ─────────────────────────────────────────────────────────────
# Custom User Model
# ─────────────────────────────────────────────────────────────
AUTH_USER_MODEL = "api.User"

# ─────────────────────────────────────────────────────────────
# DATABASE (PostgreSQL via DATABASE_URL)
# ─────────────────────────────────────────────────────────────
_db_url = os.getenv("DATABASE_URL")

if not _db_url:
    raise Exception("DATABASE_URL not set")

_parsed = _urlparse.urlparse(_db_url)

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": _parsed.path.lstrip("/"),
        "USER": _parsed.username,
        "PASSWORD": _parsed.password,
        "HOST": _parsed.hostname,
        "PORT": _parsed.port or 5432,
    }
}

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# ─────────────────────────────────────────────────────────────
# REST Framework
# ─────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}

# ─────────────────────────────────────────────────────────────
# Static / Media Files
# ─────────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STATICFILES_STORAGE = (
    "whitenoise.storage.CompressedManifestStaticFilesStorage"
)

MEDIA_URL = "/uploads/"
MEDIA_ROOT = BASE_DIR / "uploads"

# ─────────────────────────────────────────────────────────────
# Internationalization
# ─────────────────────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = False

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ─────────────────────────────────────────────────────────────
# Gemini / Upload / Email Config
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

OTP_EXPIRY_MINUTES = 10
UPLOAD_DIR = str(BASE_DIR / "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Surgical Wound Care")

# ─────────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────────
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [],
        },
    },
]