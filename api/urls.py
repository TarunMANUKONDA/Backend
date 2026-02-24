from django.urls import path
from api.views import auth, upload, classify, recommend, history, comparison, validate

urlpatterns = [
    # ─── Auth ────────────────────────────────────────────────────────────────
    path('signup',            auth.signup),
    path('verify-otp',        auth.verify_otp),
    path('resend-otp',        auth.resend_otp),
    path('login',             auth.login),
    path('logout',            auth.logout),
    path('verify_session',    auth.verify_session),
    path('forgot-password',   auth.forgot_password),
    path('verify-reset-otp',  auth.verify_reset_otp),
    path('reset-password',    auth.reset_password),
    path('update-profile',    auth.update_profile),

    # ─── Upload ───────────────────────────────────────────────────────────────
    path('upload',            upload.upload_image),

    # ─── Classify ─────────────────────────────────────────────────────────────
    path('classify',          classify.classify_wound),

    # ─── Recommend ────────────────────────────────────────────────────────────
    path('recommend',         recommend.get_recommendations),

    # ─── History / Cases ──────────────────────────────────────────────────────
    path('history',           history.get_history),
    path('create_case',       history.create_case),
    path('cases',             history.get_cases),
    path('wounds/<int:wound_id>', history.delete_wound),
    path('cases/<int:case_id>',   history.delete_case),

    # ─── Comparison ───────────────────────────────────────────────────────────
    path('compare',           comparison.compare_wounds),
    path('save_comparison',   comparison.save_comparison),
    path('save_analysis',     comparison.save_analysis),

    # ─── Validate ─────────────────────────────────────────────────────────────
    path('validate',          validate.validate_image),
]
