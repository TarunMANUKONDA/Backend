import random
import os
from django.conf import settings
from django.core.mail import send_mail
from django.conf import settings
from django.core.mail import send_mail


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))



def _send_via_smtp(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    """Send email via SMTP – handles Gmail, works locally and on Render."""
    try:
        sent_count = send_mail(
            subject=subject,
            message=text_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[to_email],
            html_message=html_body,
            fail_silently=False,
        )
        if sent_count > 0:
            print(f"✅ Email sent to {to_email} via SMTP.")
            return True
        else:
            print(f"❌ SMTP send failed: No email sent.")
            return False
    except Exception as e:
        print(f"❌ SMTP send failed: {e}")
        return False


def send_otp_email(to_email: str, otp_code: str, user_name: str = "User") -> bool:
    """Send OTP verification email via SMTP."""
    subject = f"Your Verification Code: {otp_code}"
    expiry = getattr(settings, 'OTP_EXPIRY_MINUTES', 10)

    html_body = f"""
    <html><body style="font-family:sans-serif;background:#f4f4f4;margin:0;padding:20px;">
    <div style="max-width:480px;margin:auto;background:#fff;border-radius:12px;
                overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1);">
        <div style="background:#2F80ED;padding:28px 24px;text-align:center;">
            <h1 style="color:#fff;margin:0;font-size:22px;">Email Verification</h1>
        </div>
        <div style="padding:28px 24px;">
            <p>Hi <strong>{user_name}</strong>,</p>
            <p>Thank you for signing up! Use the code below to verify your email.
               It expires in <strong>{expiry} minutes</strong>.</p>
            <div style="text-align:center;letter-spacing:8px;font-size:36px;font-weight:700;
                        color:#2F80ED;padding:20px;background:#EBF4FF;
                        border-radius:8px;margin:20px 0;">
                {otp_code}
            </div>
            <p style="color:#888;font-size:13px;">
                If you didn't request this, you can safely ignore this email.
            </p>
            <p>Best regards,<br>The Surgical Wound Care Team</p>
        </div>
    </div>
    </body></html>
    """
    text_body = f"Email Verification\n\nHi {user_name},\n\nYour code: {otp_code}\nExpires in {expiry} minutes.\n"

    sent = _send_via_smtp(to_email, subject, html_body, text_body)
    if not sent:
        print(f"📧 [DEV fallback] OTP for {to_email}: {otp_code}")
    return sent


def send_password_reset_email(to_email: str, otp_code: str, user_name: str = "User") -> bool:
    """Send password-reset OTP email via SMTP."""
    subject = f"Password Reset Code: {otp_code}"
    expiry = settings.OTP_EXPIRY_MINUTES

    html_body = f"""
    <html><body style="font-family:sans-serif;background:#f4f4f4;margin:0;padding:20px;">
    <div style="max-width:480px;margin:auto;background:#fff;border-radius:12px;
                overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1);">
        <div style="background:#2F80ED;padding:28px 24px;text-align:center;">
            <h1 style="color:#fff;margin:0;font-size:22px;">Password Reset</h1>
        </div>
        <div style="padding:28px 24px;">
            <p>Hi <strong>{user_name}</strong>,</p>
            <p>Use the code below to reset your password.
               It expires in <strong>{expiry} minutes</strong>.</p>
            <div style="text-align:center;letter-spacing:8px;font-size:36px;font-weight:700;
                        color:#2F80ED;padding:20px;background:#EBF4FF;
                        border-radius:8px;margin:20px 0;">
                {otp_code}
            </div>
            <p style="color:#888;font-size:13px;">
                If you did not request a password reset, ignore this email.
            </p>
            <p>Best regards,<br>The Surgical Wound Care Team</p>
        </div>
    </div>
    </body></html>
    """
    text_body = f"Password Reset\n\nHi {user_name},\n\nYour reset code: {otp_code}\nExpires in {expiry} minutes.\n"

    sent = _send_via_smtp(to_email, subject, html_body, text_body)
    if not sent:
        print(f"📧 [DEV fallback] Reset OTP for {to_email}: {otp_code}")
    return sent
