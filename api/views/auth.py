import secrets
from datetime import datetime, timedelta

from django.conf import settings
from django.utils import timezone
from passlib.context import CryptContext
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.models import User, UserSession, EmailVerificationOTP
from api.email_service import generate_otp, send_otp_email, send_password_reset_email

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__truncate_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def _user_dict(user):
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "email_verified": user.email_verified,
        "phone": user.phone,
        "age": user.age,
        "date_of_birth": user.date_of_birth,
        "gender": user.gender,
        "blood_type": user.blood_type,
        "emergency_contact": user.emergency_contact,
        "emergency_phone": user.emergency_phone,
        "profile_image": user.profile_image,
        "created_at": user.created_at.isoformat(),
    }


# ─── Signup ───────────────────────────────────────────────────────────────────

@api_view(['POST'])
def signup(request):
    """Send OTP — user record is created only after OTP is confirmed."""
    data = request.data
    email = data.get('email', '').lower().strip()
    name = data.get('name', '').strip()
    password = data.get('password', '')
    phone = data.get('phone', '').strip()
    age = data.get('age')
    gender = data.get('gender', '').strip()

    if not email or not name or not password:
        return Response({"success": False, "error": "email, name and password are required"},
                        status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(email=email).exists():
        return Response({"success": False, "error": "Email already registered"},
                        status=status.HTTP_400_BAD_REQUEST)

    otp_code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRY_MINUTES)

    EmailVerificationOTP.objects.filter(email=email, verified=False).delete()
    EmailVerificationOTP.objects.create(
        email=email,
        otp_code=otp_code,
        expires_at=expires_at,
        pending_name=name,
        pending_password_hash=hash_password(password),
        pending_phone=phone,
        pending_age=age,
        pending_gender=gender,
    )

    email_sent = send_otp_email(email, otp_code, name)
    if not email_sent:
        return Response({
            "success": True,
            "message": "OTP delivery failed — please use Resend Code.",
            "email": email,
            "otp_sent": False,
            "email_delivery_failed": True,
        })

    return Response({"success": True, "message": "OTP sent to your email", "email": email, "otp_sent": True})


# ─── Verify OTP (signup) ──────────────────────────────────────────────────────

@api_view(['POST'])
def verify_otp(request):
    """Verify signup OTP — creates the User record on success."""
    email = request.query_params.get('email') or request.data.get('email', '')
    otp_code = request.query_params.get('otp_code') or request.data.get('otp_code', '')
    email = email.lower().strip()

    otp_record = EmailVerificationOTP.objects.filter(
        email=email, otp_code=otp_code, verified=False
    ).first()

    if not otp_record:
        return Response({"error": "Invalid OTP code"}, status=status.HTTP_400_BAD_REQUEST)
    if otp_record.expires_at < datetime.utcnow():
        return Response({"error": "OTP has expired. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)
    if not otp_record.pending_name or not otp_record.pending_password_hash:
        return Response({"error": "Invalid OTP type"}, status=status.HTTP_400_BAD_REQUEST)
    if User.objects.filter(email=email).exists():
        return Response({"error": "Email already registered"}, status=status.HTTP_400_BAD_REQUEST)

    user = User(
        name=otp_record.pending_name, 
        email=email, 
        email_verified=True,
        phone=otp_record.pending_phone,
        age=otp_record.pending_age,
        gender=otp_record.pending_gender
    )
    user.password = otp_record.pending_password_hash  # already hashed
    user.save()

    session_token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(days=30)
    UserSession.objects.create(user=user, session_token=session_token, expires_at=expires_at)
    otp_record.delete()

    return Response({
        "success": True,
        "user": _user_dict(user),
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
    })


# ─── Resend OTP ───────────────────────────────────────────────────────────────

@api_view(['POST'])
def resend_otp(request):
    email = (request.query_params.get('email') or request.data.get('email', '')).lower().strip()

    existing = EmailVerificationOTP.objects.filter(
        email=email, verified=False, pending_name__isnull=False
    ).first()
    if not existing:
        return Response({"error": "No pending signup for this email. Please sign up again."},
                        status=status.HTTP_404_NOT_FOUND)

    pending_name = existing.pending_name
    pending_hash = existing.pending_password_hash
    existing.delete()

    otp_code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRY_MINUTES)
    EmailVerificationOTP.objects.create(
        email=email, otp_code=otp_code, expires_at=expires_at,
        pending_name=pending_name, pending_password_hash=pending_hash,
    )
    email_sent = send_otp_email(email, otp_code, pending_name)
    if not email_sent:
        return Response({
            "success": True,
            "message": "OTP delivery failed — please try Resend again.",
            "email_delivery_failed": True
        })
    return Response({"success": True, "message": "OTP resent successfully"})


# ─── Login ────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def login(request):
    email = request.data.get('email', '').lower().strip()
    password = request.data.get('password', '')

    user = User.objects.filter(email=email).first()
    if not user or not verify_password(password, user.password):
        return Response({"error": "Invalid email or password"}, status=status.HTTP_401_UNAUTHORIZED)

    session_token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(days=30)
    UserSession.objects.create(user=user, session_token=session_token, expires_at=expires_at)

    return Response({
        "success": True,
        "user": _user_dict(user),
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
    })


# ─── Logout ───────────────────────────────────────────────────────────────────

@api_view(['POST'])
def logout(request):
    token = request.data.get('session_token', '')
    UserSession.objects.filter(session_token=token).delete()
    return Response({"success": True, "message": "Logged out successfully"})


# ─── Verify Session ───────────────────────────────────────────────────────────

@api_view(['POST'])
def verify_session(request):
    token = request.data.get('session_token', '')
    session = UserSession.objects.select_related('user').filter(session_token=token).first()
    if not session or session.expires_at < datetime.utcnow():
        return Response({"success": False, "error": "Invalid or expired session"})

    return Response({"success": True, "user": _user_dict(session.user)})


# ─── Forgot Password ──────────────────────────────────────────────────────────

@api_view(['POST'])
def forgot_password(request):
    email = (request.query_params.get('email') or request.data.get('email', '')).lower().strip()

    user = User.objects.filter(email=email).first()
    if not user:
        return Response({"error": "No account found with this email address."},
                        status=status.HTTP_404_NOT_FOUND)

    reset_key = "reset:" + email
    EmailVerificationOTP.objects.filter(email=reset_key, verified=False).delete()

    otp_code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRY_MINUTES)
    EmailVerificationOTP.objects.create(email=reset_key, otp_code=otp_code, expires_at=expires_at)

    email_sent = send_password_reset_email(email, otp_code, user.name)
    if not email_sent:
        EmailVerificationOTP.objects.filter(email=reset_key).delete()
        return Response({"error": "Failed to send reset email. Please try again."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response({"success": True, "message": "Password reset code sent to your email."})


# ─── Verify Reset OTP ─────────────────────────────────────────────────────────

@api_view(['POST'])
def verify_reset_otp(request):
    email = (request.query_params.get('email') or request.data.get('email', '')).lower().strip()
    otp_code = request.query_params.get('otp_code') or request.data.get('otp_code', '')

    reset_key = "reset:" + email
    otp_record = EmailVerificationOTP.objects.filter(
        email=reset_key, otp_code=otp_code, verified=False
    ).first()

    if not otp_record:
        return Response({"error": "Invalid or already used reset code."}, status=status.HTTP_400_BAD_REQUEST)
    if otp_record.expires_at < datetime.utcnow():
        return Response({"error": "Reset code has expired. Please request a new one."}, status=status.HTTP_400_BAD_REQUEST)

    otp_record.verified = True
    otp_record.save()
    return Response({"success": True, "message": "Code verified. You may now set a new password."})


# ─── Reset Password ───────────────────────────────────────────────────────────

@api_view(['POST'])
def reset_password(request):
    email = (request.query_params.get('email') or request.data.get('email', '')).lower().strip()
    otp_code = request.query_params.get('otp_code') or request.data.get('otp_code', '')
    new_password = request.query_params.get('new_password') or request.data.get('new_password', '')

    reset_key = "reset:" + email
    otp_record = EmailVerificationOTP.objects.filter(
        email=reset_key, otp_code=otp_code, verified=True
    ).first()

    if not otp_record:
        return Response({"error": "Reset code not verified or already used."}, status=status.HTTP_400_BAD_REQUEST)
    if otp_record.expires_at < datetime.utcnow():
        return Response({"error": "Reset code expired. Please restart."}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.filter(email=email).first()
    if not user:
        return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

    user.password = hash_password(new_password)
    user.save()
    otp_record.delete()
    return Response({"success": True, "message": "Password has been reset successfully."})


# ─── Profile Update ───────────────────────────────────────────────────────────

@api_view(['POST'])
def update_profile(request):
    token = request.data.get('session_token', '')
    session = UserSession.objects.select_related('user').filter(session_token=token).first()
    if not session or session.expires_at < datetime.utcnow():
        return Response({"success": False, "error": "Invalid or expired session"}, status=status.HTTP_401_UNAUTHORIZED)

    user = session.user
    data = request.data

    # Update allowed fields
    if 'name' in data: user.name = data['name']
    if 'phone' in data: user.phone = data['phone']
    if 'date_of_birth' in data: user.date_of_birth = data['date_of_birth']
    if 'gender' in data: user.gender = data['gender']
    if 'blood_type' in data: user.blood_type = data['blood_type']
    if 'emergency_contact' in data: user.emergency_contact = data['emergency_contact']
    if 'emergency_phone' in data: user.emergency_phone = data['emergency_phone']
    
    user.save()
    return Response({"success": True, "message": "Profile updated successfully", "user": _user_dict(user)})
