import os
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse as _urlparse
from PIL import ImageFile

# Fix for "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ─────────────────────────────────────────────────────────────
# Base directory
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env (works locally)
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
import dj_database_url

DATABASES = {
    "default": dj_database_url.config(
        default=os.getenv("DATABASE_URL"),
        conn_max_age=600,
        conn_health_checks=True,
    )
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
# Upload / Email Config
# ─────────────────────────────────────────────────────────────

OTP_EXPIRY_MINUTES = 10
UPLOAD_DIR = str(BASE_DIR / "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Email Configuration (SMTP)
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.getenv("SMTP_USERNAME", "")
EMAIL_HOST_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_from_name = os.getenv("SMTP_FROM_NAME", "Surgical Wound Care")
_from_email = os.getenv("SMTP_FROM_EMAIL", "no-reply@woundcare.com")
DEFAULT_FROM_EMAIL = f'{_from_name} <{_from_email}>'

SMTP_FROM_EMAIL = _from_email
SMTP_FROM_NAME = _from_name

# Prevent SMTP from hanging the gunicorn worker (10 second limit)
EMAIL_TIMEOUT = 10

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

# ─────────────────────────────────────────────────────────────
# Logging (Suppress verbose dev logs)
# ─────────────────────────────────────────────────────────────
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django.server': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}