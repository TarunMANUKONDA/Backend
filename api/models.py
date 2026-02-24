from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.utils import timezone


# ─── Custom User Manager ─────────────────────────────────────────────────────

class UserManager(BaseUserManager):
    def create_user(self, email, name, password=None):
        if not email:
            raise ValueError('Users must have an email address')
        user = self.model(email=self.normalize_email(email), name=name)
        if password:
            user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, name, password):
        user = self.create_user(email, name, password)
        return user


# ─── User ─────────────────────────────────────────────────────────────────────

class User(AbstractBaseUser):
    """Maps to the 'users' table — same columns as the FastAPI SQLAlchemy model."""
    class Meta:
        db_table = 'users'

    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100, unique=True, db_index=True)
    # NOTE: AbstractBaseUser provides 'password' field (stores hash).
    # We alias it via property so the rest of the app uses password_hash.
    phone = models.CharField(max_length=20, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    date_of_birth = models.CharField(max_length=50, null=True, blank=True)
    gender = models.CharField(max_length=20, null=True, blank=True)
    blood_type = models.CharField(max_length=10, null=True, blank=True)
    emergency_contact = models.CharField(max_length=100, null=True, blank=True)
    emergency_phone = models.CharField(max_length=20, null=True, blank=True)
    profile_image = models.CharField(max_length=255, null=True, blank=True)
    email_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    objects = UserManager()

    # Passlib bcrypt hashes are stored in AbstractBaseUser's `password` column.
    # Expose it as password_hash so views can stay consistent with FastAPI naming.
    @property
    def password_hash(self):
        return self.password

    @password_hash.setter
    def password_hash(self, value):
        self.password = value

    def __str__(self):
        return self.email


# ─── Session ──────────────────────────────────────────────────────────────────

class UserSession(models.Model):
    """Custom session tokens (maps to 'sessions' table)."""
    class Meta:
        db_table = 'sessions'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    session_token = models.CharField(max_length=255, unique=True, db_index=True)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)


# ─── Email Verification OTP ───────────────────────────────────────────────────

class EmailVerificationOTP(models.Model):
    class Meta:
        db_table = 'email_verification_otps'

    email = models.CharField(max_length=100, db_index=True)
    otp_code = models.CharField(max_length=6)
    expires_at = models.DateTimeField()
    verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    # For signup OTPs only
    pending_name = models.CharField(max_length=100, null=True, blank=True)
    pending_password_hash = models.CharField(max_length=255, null=True, blank=True)
    pending_phone = models.CharField(max_length=20, null=True, blank=True)
    pending_age = models.IntegerField(null=True, blank=True)
    pending_gender = models.CharField(max_length=20, null=True, blank=True)


# ─── Case ─────────────────────────────────────────────────────────────────────

class Case(models.Model):
    class Meta:
        db_table = 'cases'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cases')
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ─── Wound ────────────────────────────────────────────────────────────────────

class Wound(models.Model):
    class Meta:
        db_table = 'wounds'

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='wounds')
    case = models.ForeignKey(Case, on_delete=models.SET_NULL, null=True, blank=True, related_name='wounds')
    image_path = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255, null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=50, default='pending')
    notes = models.TextField(null=True, blank=True)
    # Cached analysis results
    classification = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    redness_level = models.IntegerField(null=True, blank=True)
    discharge_detected = models.BooleanField(null=True, blank=True)
    discharge_type = models.CharField(max_length=50, null=True, blank=True)
    edge_quality = models.IntegerField(null=True, blank=True)
    tissue_composition = models.JSONField(null=True, blank=True)
    analysis = models.JSONField(null=True, blank=True)


# ─── Classification ───────────────────────────────────────────────────────────

class Classification(models.Model):
    class Meta:
        db_table = 'classifications'

    wound = models.ForeignKey(Wound, on_delete=models.CASCADE, related_name='classifications')
    wound_type = models.CharField(max_length=100, db_index=True)
    confidence = models.FloatField()
    all_probabilities = models.JSONField()
    processing_time_ms = models.IntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)


# ─── Recommendation ───────────────────────────────────────────────────────────

class Recommendation(models.Model):
    class Meta:
        db_table = 'recommendations'

    classification = models.ForeignKey(Classification, on_delete=models.CASCADE, related_name='recommendations')
    summary = models.TextField()
    cleaning_instructions = models.JSONField(null=True, blank=True)
    dressing_recommendations = models.JSONField(null=True, blank=True)
    medication_suggestions = models.JSONField(null=True, blank=True)
    expected_healing_time = models.CharField(max_length=100, null=True, blank=True)
    follow_up_schedule = models.JSONField(null=True, blank=True)
    warning_signs = models.JSONField(null=True, blank=True)
    when_to_seek_help = models.JSONField(null=True, blank=True)
    diet_advice = models.JSONField(null=True, blank=True)
    activity_restrictions = models.JSONField(null=True, blank=True)
    ai_confidence = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ─── Comparison ───────────────────────────────────────────────────────────────

class Comparison(models.Model):
    class Meta:
        db_table = 'comparisons'

    case = models.ForeignKey(Case, on_delete=models.SET_NULL, null=True, blank=True)
    wound_before = models.ForeignKey(Wound, on_delete=models.SET_NULL, null=True, blank=True,
                                     related_name='comparisons_as_before')
    wound_after = models.ForeignKey(Wound, on_delete=models.SET_NULL, null=True, blank=True,
                                    related_name='comparisons_as_after')
    analysis = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
