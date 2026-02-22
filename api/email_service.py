import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from django.conf import settings


def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))


def _send_via_smtp(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    """Send email via Gmail SMTP (TLS on port 587)."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
        msg["To"] = to_email

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, to_email, msg.as_string())

        print(f"✅ Email sent to {to_email}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP auth failed — check SMTP_USERNAME and SMTP_PASSWORD (Gmail App Password).")
        return False
    except Exception as e:
        print(f"❌ SMTP send failed: {e}")
        return False


def send_otp_email(to_email: str, otp_code: str, user_name: str = "User") -> bool:
    """Send OTP verification email via Gmail SMTP."""
    subject = f"Your Verification Code: {otp_code}"
    expiry = settings.OTP_EXPIRY_MINUTES

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

    if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
        print(f"📧 [DEV] OTP for {to_email}: {otp_code}")
        return True

    sent = _send_via_smtp(to_email, subject, html_body, text_body)
    if not sent:
        print(f"📧 [DEV fallback] OTP for {to_email}: {otp_code}")
    return sent


def send_password_reset_email(to_email: str, otp_code: str, user_name: str = "User") -> bool:
    """Send password-reset OTP email via Gmail SMTP."""
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

    if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
        print(f"📧 [DEV] Reset OTP for {to_email}: {otp_code}")
        return True

    sent = _send_via_smtp(to_email, subject, html_body, text_body)
    if not sent:
        print(f"📧 [DEV fallback] Reset OTP for {to_email}: {otp_code}")
    return sent
