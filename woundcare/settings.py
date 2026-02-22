import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-wound-care-secret-key-change-in-production-2024')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

# ─── Apps ─────────────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'woundcare.urls'
WSGI_APPLICATION = 'woundcare.wsgi.application'

# ─── Custom User Model ────────────────────────────────────────────────────────
AUTH_USER_MODEL = 'api.User'

# ─── Database (PostgreSQL) ─────────────────────────────────────────────────────
_db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/woundcare')

# Parse DATABASE_URL → Django DATABASES dict
import urllib.parse as _urlparse
_parsed = _urlparse.urlparse(_db_url)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': _parsed.path.lstrip('/'),
        'USER': _parsed.username,
        'PASSWORD': _parsed.password,
        'HOST': _parsed.hostname,
        'PORT': _parsed.port or 5432,
    }
}

# ─── CORS ─────────────────────────────────────────────────────────────────────
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# ─── REST Framework ───────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# ─── Static / Media files ─────────────────────────────────────────────────────
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/uploads/'
MEDIA_ROOT = BASE_DIR / 'uploads'

# ─── Internationalisation ─────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = False
USE_TZ = False          # Store naive datetimes (same as FastAPI behaviour)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ─── Gemini / Email config (read by views directly) ──────────────────────────
GEMINI_API_KEY      = os.getenv('GEMINI_API_KEY', '')
OTP_EXPIRY_MINUTES  = 10
UPLOAD_DIR          = str(BASE_DIR / 'uploads')
MAX_FILE_SIZE       = int(os.getenv('MAX_FILE_SIZE', 10485760))
ALLOWED_EXTENSIONS  = {'.jpg', '.jpeg', '.png'}

SMTP_HOST       = os.getenv('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT       = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME   = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD   = os.getenv('SMTP_PASSWORD', '')
SMTP_FROM_EMAIL = os.getenv('SMTP_FROM_EMAIL', '')
SMTP_FROM_NAME  = os.getenv('SMTP_FROM_NAME', 'Surgical Wound Care')

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {'context_processors': []},
    },
]
