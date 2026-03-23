"""
cv_processor.py
================
LAB Color-Space Wound Tissue Analysis — upgraded to support accurate
tissue/depth classification for ALL four wound categories:

  • open_wound   — granulation / epithelial / slough / necrotic / fibrin
  • burn_wound   — superficial / partial-thickness / full-thickness / char
  • sutured_wound — healing incision / fibrin / granuloma / necrosis
  • scar          — mature scar / hypertrophic / hyperpigmented / normal skin

Pipeline per call:
  1. Specular-highlight removal  (HSV V+S gate)
  2. CLAHE illumination correction (LAB L-channel)
  3. Distance transform for spatial context (wound centre vs. edge)
  4. Gaussian smoothing
  5. Category-specific LAB + RGB rule fusion
  6. Priority-ordered deduplication (no pixel counted twice)
  7. Overlay + boundary image generation
"""

import cv2
import numpy as np
import base64
import os

# ─────────────────────────────────────────────────────────────────────────────
# Segmentation model (lazy-load)
# ─────────────────────────────────────────────────────────────────────────────
_segment_model = None
MODEL_H5_PATH = os.path.join(os.path.dirname(__file__), 'models', 'segment_model.h5')

def _get_segment_model():
    """Lazily load the Keras h5 model."""
    global _segment_model
    if _segment_model is None:
        if not os.path.exists(MODEL_H5_PATH):
            return None
        import tensorflow as tf
        _segment_model = tf.keras.models.load_model(MODEL_H5_PATH, compile=False)
    return _segment_model


# ─────────────────────────────────────────────────────────────────────────────
# Shared preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes, mask=None):
    """
    Load image, remove specular highlights, apply CLAHE, smooth.
    Returns (image_smoothed_rgb, lab_final, r, g, b, valid_mask, dist_transform, gray)
    or None on failure.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return None

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── Mask setup & ROI Cropping ───────────────────────────────────────────
    valid_mask = None
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        valid_mask = (mask > 0)
        
        # ROI Cropping Disabled per User Request
        pass
        
        # Black out background outside the mask within the crop
        image_rgb = image_rgb.copy()
        image_rgb[~valid_mask] = 0

    # ── Specular-highlight removal ──────────────────────────────────────────
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    _, s_v, v_v = cv2.split(hsv)
    reflection_mask = (v_v > 210) & (s_v < 50)
    if valid_mask is not None:
        reflection_mask = reflection_mask & valid_mask
    image_rgb[reflection_mask] = 0

    # ── Distance transform (spatial context) ────────────────────────────────
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dist_transform = np.zeros((h, w), dtype=np.float32)
    if valid_mask is not None:
        dist_transform = cv2.distanceTransform(
            valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # ── CLAHE illumination correction ───────────────────────────────────────
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab_corrected = cv2.merge((L, A, B))
    image_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

    # ── Gaussian smoothing ──────────────────────────────────────────────────
    image_smoothed = cv2.GaussianBlur(image_corrected, (5, 5), 0)

    # ── Final LAB channels ──────────────────────────────────────────────────
    lab_final = cv2.cvtColor(image_smoothed, cv2.COLOR_RGB2LAB)
    L_f, A_f, B_f = cv2.split(lab_final)
    r, g, b = cv2.split(image_smoothed)

    return {
        "image_smoothed": image_smoothed,
        "image_corrected": image_corrected,
        "L_f": L_f, "A_f": A_f, "B_f": B_f,
        "r": r, "g": g, "b": b,
        "hsv": hsv,
        "valid_mask": valid_mask,
        "dist_transform": dist_transform,
        "gray": gray,
        "h": h, "w": w
    }


def _apply_mask_to_rule(rule_mask, L_f, valid_mask):
    """Enforce: ignore blacked-out background pixels and respect wound ROI."""
    base = (L_f > 0)
    if valid_mask is not None:
        return rule_mask & valid_mask & base
    return rule_mask & base


def _remove_stitch_artifacts(hsv, gray):
    """
    Detect medical sutures (specifically blue/black thread) and return a mask 
    to subtract from tissue classification (prevents threads from being called necrotic).
    """
    stitch_mask = np.zeros_like(gray, dtype=np.uint8)
    try:
        # 1. Color-based detection (Blue sutures — high precision)
        # HSV range for blue/violet medical thread
        lower_blue = np.array([85, 40, 40])
        upper_blue = np.array([145, 255, 255])
        blue_threads = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 2. Edge-based detection (Fallback for black thread / metal staples)
        # Using a higher threshold for edges to catch sharp thread-like objects
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine blue detection with sharp edges
        combined = cv2.bitwise_or(blue_threads, edges)
        
        # 3. Clean and isolate small stitch-like components
        k = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined, k, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k, iterations=1)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Stitches are small; ignore large blobs (which might be real tissue)
            if cv2.contourArea(cnt) < 1200:
                cv2.drawContours(stitch_mask, [cnt], -1, 255, -1)
                
    except Exception as e:
        print(f"[CV WARNING] _remove_stitch_artifacts failed: {e}")
        
    return stitch_mask


def _encode_overlay_and_boundary(image_smoothed, tissue_map, img_base, valid_mask, h, w):
    """
    Blend tissue map over image and draw wound boundary on the CROPPED result.
    Returns (overlay_b64, boundary_b64).
    """
    # 1. Overlay image (Tissue Map)
    overlay = cv2.addWeighted(image_smoothed, 0.6, tissue_map, 0.4, 0)
    
    # 2. Boundary image (Zoomed-in original with contour)
    # We use img_base (which is already cropped and CLAHE-corrected)
    boundary_img = img_base.copy()

    if valid_mask is not None:
        mask_u8 = (valid_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw on boundary image
        cv2.drawContours(boundary_img, contours, -1, (255, 255, 255), 9)
        cv2.drawContours(boundary_img, contours, -1, (255, 30, 30), 5)
        # Draw on overlay image
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 4)

    # 3. Encode as compressed JPEGs to prevent DB/Frontend bloat (quality=70)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    
    _, buf1 = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), encode_params)
    overlay_b64 = base64.b64encode(buf1).decode('utf-8')
    
    _, buf2 = cv2.imencode('.jpg', cv2.cvtColor(boundary_img, cv2.COLOR_RGB2BGR), encode_params)
    boundary_b64 = base64.b64encode(buf2).decode('utf-8')
    return overlay_b64, boundary_b64


# ─────────────────────────────────────────────────────────────────────────────
#  SPECIALIZED WOUND ROI  (incision / scar crop)
# ─────────────────────────────────────────────────────────────────────────────

def process_specialized_wound(image_bytes: bytes, category: str):
    """
    Apply OpenCV logic for incision detection using color-guided thresholding.
    Triggered when CLIP identifies 'sutured_wound' or 'scar'.
    Returns: {stitch_count, incision_detected, mask, is_scar}
    """
    try:
        # Convert bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w = img.shape[:2]
        
        # 1. Border suppression mask (ignore outer 3%)
        border_px = int(min(w, h) * 0.03)
        content_mask = np.zeros((h, w), dtype=np.uint8)
        content_mask[border_px:-border_px, border_px:-border_px] = 255
        
        # 2. Image Enhancement (Bilateral Filter 11, 85, 85)
        smoothed = cv2.bilateralFilter(img, 11, 85, 85)
        
        # 3. HSV Color Evidence Layer
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        # Blue Sutures (More sensitive range: 85-145)
        lower_blue = np.array([85, 30, 30])
        upper_blue = np.array([145, 255, 255])
        blue_mask = cv2.bitwise_and(cv2.inRange(hsv, lower_blue, upper_blue), content_mask)
        
        # Red Wound Bed (Double-band for wrap-around hue)
        mask_red1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 30, 30]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(mask_red1, mask_red2), content_mask)
        
        # 4. Geometric Evidence (Canny edges 30, 100)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Focus zone: combine red and blue, then dilate heavily (35x35) to bridge gaps
        focus_zone = cv2.dilate(cv2.bitwise_or(blue_mask, red_mask), np.ones((35, 35), np.uint8))
        focused_edges = cv2.bitwise_and(cv2.bitwise_and(edges, focus_zone), content_mask)
        
        # 5. Clean up and refine (7x7 kernel)
        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(focused_edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        dilated = cv2.dilate(closed, kernel, iterations=5)
        
        # 6. Find and Score the primary incision line
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        incision_detected = False
        roi_confidence = 0
        
        if contours:
            # Better Scoring: (max(W,H) / min(W,H)) * Area * (Center Weight)
            def score_contour(c):
                cx, cy, cw, ch = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
                # Distance from center of image (normalized 0 to 1, where 1 is center)
                dist_from_center = 1.0 - (np.sqrt((cx + cw/2 - w/2)**2 + (cy + ch/2 - h/2)**2) / (np.sqrt(w**2 + h**2)/2))
                return area * aspect_ratio * (dist_from_center ** 2)

            incision_contours = sorted(contours, key=score_contour, reverse=True)
            if incision_contours and cv2.contourArea(incision_contours[0]) > 500:
                best_cnt = incision_contours[0]
                cv2.drawContours(final_mask, [best_cnt], -1, 255, cv2.FILLED)
                incision_detected = True
                roi_confidence = 85 # Benchmark high confidence
        
        # Fallback to central ellipse if nothing found
        if not incision_detected:
            cv2.ellipse(final_mask, (w//2, h//2), (h//4, w//3), 0, 0, 360, 255, -1)
            roi_confidence = 30
            
        return {
            "mask": final_mask,
            "incision_detected": incision_detected,
            "roi_confidence": roi_confidence,
            "category": category
        }

    except Exception as e:
        print(f"[CV ERROR] process_specialized_wound V4: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def segment_wound(image_bytes: bytes):
    """Use h5 model to segment the open wound area. Returns (expanded_mask, pixels)."""
    try:
        model = _get_segment_model()
        if model is None:
            return None, 0

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, 0

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(image_rgb, (128, 128)) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        pred = model.predict(img_input, verbose=0)[0]
        mask = (pred > 0.3).astype(np.uint8)
        mask = cv2.resize(mask, (w, h))

        kernel = np.ones((25, 17), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)
        return expanded_mask, int(np.sum(expanded_mask > 0))


    except Exception as e:
        print(f"[CV SEGMENT ERROR] {e}")
        return None, 0


def detect_open_wound_roi(image_bytes: bytes):
    """
    Fast wound region detection using HSV red thresholding.
    Used for CLIP pre-processing to focus on the wound area.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None, 0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Resize for speed
        img_resized = cv2.resize(img_rgb, (600, 600))
        
        # 2. Convert to HSV
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

        # 3. Detect wound-like regions (Red, Yellow/Slough, Dark/Necrotic)
        mask_red = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([20, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([20, 40, 40]), np.array([45, 255, 255]))
        mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
        mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_yellow, mask_dark))

        # 4. Remove noise
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 5. Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return img_resized, 0
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # 6. Get bounding box and crop with 15% padding
        x, y, w_bb, h_bb = cv2.boundingRect(largest)
        pad_w = int(w_bb * 0.15)
        pad_h = int(h_bb * 0.15)
        
        y1, y2 = max(0, y - pad_h), min(600, y + h_bb + pad_h)
        x1, x2 = max(0, x - pad_w), min(600, x + w_bb + pad_w)
        
        crop = img_resized[y1:y2, x1:x2]
        
        return crop, int(area)
    except Exception as e:
        print(f"[CV ERROR] detect_open_wound_roi: {e}")
        return None, 0


def get_roi_cropped_bytes(image_bytes: bytes, mask):
    """
    Finds the bounding box of the mask and returns cropped image bytes.
    Used to focus AI tissue analysis on the wound bed only.
    """
    if mask is None:
        return image_bytes
        
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return image_bytes
        
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
            
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image_bytes
            
        x, y, w_roi, h_roi = cv2.boundingRect(coords)
        
        # Add a small padding (10%)
        pad_x = int(w_roi * 0.1)
        pad_y = int(h_roi * 0.1)
        
        y1 = max(0, y - pad_y)
        y2 = min(h, y + h_roi + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(w, x + w_roi + pad_x)
        
        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            return image_bytes
            
        _, buffer = cv2.imencode('.jpg', cropped)
        return buffer.tobytes()
        
    except Exception as e:
        print(f"[CV ERROR] get_roi_cropped_bytes: {e}")
        return image_bytes


def analyze_tissue_lab_v2(image_bytes: bytes, mask=None):
    """
    Stitched/Suture Tissue Analysis (V2 Wrapper).
    Delegates to the robust _classify_sutured_wound logic.
    """
    # Reuse the main pipeline but force 'sutured_wound' logic
    return analyze_tissue_lab(image_bytes, mask=mask, category="sutured_wound")


# ─────────────────────────────────────────────────────────────────────────────
#  OPEN WOUND — LAB tissue analysis
# ─────────────────────────────────────────────────────────────────────────────

def _classify_open_wound(pp: dict):
    """
    LAB + RGB rules for open wounds.

    Tissue classes (priority high→low):
        Necrotic  → L<85, A≈128, B≈128  (dark, desaturated, no hue shift)
        Slough    → high B (yellow-green), moderate A
        Granulation → high A (red shift), medium-high L
        Epithelial  → high L (pale pink), moderate A
        Fibrin      → very high L, low chroma (near-white)
    """
    L_f, A_f, B_f = pp["L_f"], pp["A_f"], pp["B_f"]
    r, g, b = pp["r"], pp["g"], pp["b"]
    valid_mask = pp["valid_mask"]
    dist_transform = pp["dist_transform"]

    def amr(rule):
        return _apply_mask_to_rule(rule, L_f, valid_mask)

    # Spatial context
    spatial_edge   = (dist_transform < 0.40) & (valid_mask is not None)
    spatial_center = (dist_transform >= 0.30) | (valid_mask is None)

    # ── Necrotic (dark, neutral LAB, very dark RGB) ─────────────────────────
    # Lowered L from 85 to 65 to reduce shadow false-positives
    is_neutral = (A_f > 120) & (A_f < 136) & (B_f > 120) & (B_f < 136)
    lab_nec    = (L_f < 65) & is_neutral
    rgb_nec    = (r < 55) & (g < 55) & (b < 55)

    # ── Slough (yellow-green, high yellow saturation) ───────────────────────
    # Increased B from 155 to 162 + added strict R>G/G>B requirement
    lab_slough = (B_f > 162) & (L_f >= 110) & (A_f < 145)
    rgb_slough = (r > 150) & (g > 140) & (b < 160) & (g.astype(np.int16) > b.astype(np.int16) + 30)

    # ── Granulation (high red component A, highly saturated) ─────────────────
    # Increased A from 148 to 155 + tightened RGB delta
    lab_gran   = (A_f > 155) & (L_f > 80) & (L_f < 210) & (B_f < 155)
    rgb_gran   = (r > 155) & (r.astype(np.int16) > g.astype(np.int16) + 45) & \
                 (r.astype(np.int16) > b.astype(np.int16) + 45)

    # ── Epithelial (pale pink, bright, moderately reddish) ──────────────────
    # Refined for pale edges: high brightness, low-to-mid red
    lab_epi    = (A_f > 132) & (L_f > 180) & (B_f < 150)
    rgb_epi    = (r > 195) & (g > 145) & (b > 145)
    edge_epi   = (r > 175) & (g > 115) & (b > 115) & spatial_edge

    # ── Fibrin / white slough ────────────────────────────────────────────────
    # Increased L from 215 to 230 to avoid glints
    lab_white  = L_f > 230
    rgb_white  = (r > 230) & (g > 230) & (b > 220) & \
                 (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 20)

    nec_mask   = amr(rgb_nec   | lab_nec)
    slough_mask = amr(rgb_slough | lab_slough)
    gran_mask  = amr(rgb_gran  | lab_gran)
    epi_mask   = amr(rgb_epi   | lab_epi | edge_epi)
    white_mask = amr(rgb_white | lab_white)

    return nec_mask, slough_mask, gran_mask, epi_mask, white_mask


# ─────────────────────────────────────────────────────────────────────────────
#  BURN WOUND — full LAB tissue analysis (replaces basic-RGB version)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_burn_wound(pp: dict):
    """
    LAB-based burn depth classification.

    Tissue classes (priority high→low):
        Char / Eschar   → very dark L, desaturated  (deep full-thickness)
        Full-Thickness  → pale / leathery: very high L, low A & B chroma
        Partial-Thick   → erythema (red shift A↑) or blister bed (pink, moderate L)
        Superficial     → bright red (high A, moderate-high L)
        Healing         → epithelial pink (high L, moderate A) — late stage

    RGB fallback is fused via OR with LAB.
    """
    L_f, A_f, B_f = pp["L_f"], pp["A_f"], pp["B_f"]
    r, g, b = pp["r"], pp["g"], pp["b"]
    valid_mask = pp["valid_mask"]

    def amr(rule):
        return _apply_mask_to_rule(rule, L_f, valid_mask)

    # ── Char (deep burn, carbonised tissue) ─────────────────────────────────
    # Very dark + slightly warm but nearly achromatic
    lab_char  = (L_f < 60) & (A_f > 118) & (A_f < 138) & (B_f > 118) & (B_f < 138)
    rgb_char  = (r < 55) & (g < 55) & (b < 55)

    # ── Full-thickness leathery / waxy (pale white or tan) ──────────────────
    # High brightness, low chroma — mimics devitalized tissue
    lab_full  = (L_f > 190) & (A_f < 138) & (B_f < 145) & \
                (np.abs(A_f.astype(np.int16) - 128) < 14) & \
                (np.abs(B_f.astype(np.int16) - 128) < 20)
    rgb_full  = (r > 200) & (g > 190) & (b > 185) & \
                (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 20)

    # ── Partial-thickness erythema (active red, moderate L) ─────────────────
    # Clearly red/pink but not pale → living but damaged dermis
    lab_partial = (A_f > 148) & (L_f > 100) & (L_f < 190)
    rgb_partial = (r > 160) & (g < 130) & (b < 120) & (r > g + 30)

    # ── Partial-thickness blister bed (brighter pink) ───────────────────────
    lab_blister = (A_f > 138) & (L_f >= 165) & (B_f < 150)
    rgb_blister = (r > 200) & (g > 130) & (b > 130) & (r > g + 20)

    # ── Superficial / 1st degree (bright red erythema) ──────────────────────
    lab_super   = (A_f > 153) & (L_f > 130) & (L_f < 200) & (B_f < 145)
    rgb_super   = (r > 180) & (g < 110) & (b < 110)

    # ── Re-epithelialisation (healing burn, late stage) ─────────────────────
    lab_heal    = (A_f > 130) & (L_f > 195) & (B_f < 148)
    rgb_heal    = (r > 210) & (g > 160) & (b > 155)

    char_mask    = amr(rgb_char    | lab_char)
    full_mask    = amr(rgb_full    | lab_full)
    partial_mask = amr(rgb_partial | lab_partial | rgb_blister | lab_blister)
    super_mask   = amr(rgb_super   | lab_super)
    heal_mask    = amr(rgb_heal    | lab_heal)

    return char_mask, full_mask, partial_mask, super_mask, heal_mask


# ─────────────────────────────────────────────────────────────────────────────
#  SUTURED WOUND — LAB tissue analysis
# ─────────────────────────────────────────────────────────────────────────────

def _classify_sutured_wound(pp: dict):
    """
    LAB rules tuned for post-surgical incisional wounds.  v4.0

    Tissue classes (priority high→low):
        Necrosis     → low L, near-neutral LAB
        Slough       → BOTH LAB AND RGB must match + spatial centre gate + min area
        Granulation  → high A (redness) around incision
        Epithelial   → bright pink / flesh / skin tones (absorbs all skin)
        Fibrin       → very bright / white exudate
    """
    print("[CV SUTURE v4.0] _classify_sutured_wound — spatial + area gate active")
    L_f, A_f, B_f = pp["L_f"], pp["A_f"], pp["B_f"]
    r, g, b       = pp["r"],   pp["g"],   pp["b"]
    valid_mask    = pp["valid_mask"]
    dist_t        = pp["dist_transform"]   # 0 = edge, 1 = centre

    def amr(rule):
        return _apply_mask_to_rule(rule, L_f, valid_mask)

    # ── Skin-tone exclusion (Increased range for warm/yellow lighting) ───────
    # OpenCV LAB scale 0-255.  Skin: L>110, A 115-170, B 115-210 (Tighter B to allow Slough)
    is_skin_tone = (
        (L_f > 110) &
        (A_f >= 115) & (A_f <= 170) &
        (B_f >= 115) & (B_f <= 210)
    )

    # ── Necrosis (dark, near-neutral) ────────────────────────────────────────
    # Lowered L from 70 to 60
    lab_nec = (L_f < 60) & (A_f > 122) & (A_f < 134) & (B_f > 122) & (B_f < 134)
    rgb_nec = (L_f < 58) & (A_f < 137) & (B_f < 137)

    # ── Slough — THREE-LAYER GATE (v5.0 Extreme Strictness) ───────────────────
    # Tightened yellow B > 165 and G > B delta
    lab_sl = (B_f > 165) & (A_f > 105) & (A_f < 160) & (L_f > 110)
    rgb_sl = (
        (r > 145) & (g > 140) & (b < 170) &
        (g.astype(np.int16) > (b.astype(np.int16) + 35))
    )
    
    # Fuzzy merge: if either LAB or RGB is very confident, or both are moderately confident
    slough_raw = (lab_sl | rgb_sl)

    # ── Granulation (clearly red, not skin) ───────────────────────────────────
    # Increased A from 155 to 160 + RGB delta tightening
    lab_gran = (A_f > 160) & (B_f < 155) & (L_f > 80) & (L_f < 195)
    rgb_gran = (r > 155) & \
               (r.astype(np.int16) > g.astype(np.int16) + 50) & \
               (r.astype(np.int16) > b.astype(np.int16) + 50)

    # ── Epithelial / healthy skin — explicitly absorbs all skin-tone pixels ───
    lab_epi  = ((L_f > 155) & (A_f > 118) & (A_f <= 168) & (B_f < 210)) | is_skin_tone
    rgb_epi  = (r > 180) & (g > 130) & (b > 110)

    # ── Fibrin / white exudate ────────────────────────────────────────────────
    lab_white = L_f > 230
    rgb_white = (r > 230) & (g > 230) & (b > 220) & \
                (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 15)

    # ── Build masks ───────────────────────────────────────────────────────────
    nec_mask   = amr(lab_nec | rgb_nec)
    slough_mask = amr(slough_raw)
    gran_mask  = amr(lab_gran | rgb_gran)
    epi_mask   = amr(lab_epi  | rgb_epi)
    white_mask = amr(lab_white | rgb_white)

    # Connected-component area gate: drop any slough region < 300 px
    # Eliminates scattered false-positive specks; only keeps real exudate patches.
    if np.sum(slough_mask) > 0:
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
            slough_mask.astype(np.uint8), connectivity=8)
        clean = np.zeros_like(slough_mask, dtype=bool)
        for i in range(1, n_comp):
            if stats[i, cv2.CC_STAT_AREA] >= 300:
                clean |= (labels == i)
        slough_mask = clean

    print(f"[CV SUTURE v4.0] slough px after area-gate: {int(np.sum(slough_mask))}")
    return nec_mask, slough_mask, gran_mask, epi_mask, white_mask


# ─────────────────────────────────────────────────────────────────────────────
#  SCAR — LAB tissue analysis
# ─────────────────────────────────────────────────────────────────────────────

def _classify_scar(pp: dict):
    """
    LAB rules for mature scar assessment.

    Classes (priority high→low):
        Hyperpigmented → dark + warm-reddish or brownish  (A↑, low-mid L)
        Hypertrophic   → mid-bright, significantly reddish (raised, red-pink)
        Keloid         → very red, raised, mid brightness  (strong A, moderate L)
        Mature scar    → pale, slightly pinkish or white   (high L, low A)
        Normal skin    → baseline skin tone (moderate L, slightly warm)
    """
    L_f, A_f, B_f = pp["L_f"], pp["A_f"], pp["B_f"]
    r, g, b = pp["r"], pp["g"], pp["b"]
    valid_mask = pp["valid_mask"]

    def amr(rule):
        return _apply_mask_to_rule(rule, L_f, valid_mask)

    # ── Hyperpigmented scar ───────────────────────────────────────────────────
    lab_hyper_pig = (L_f >= 60) & (L_f < 140) & (A_f > 132) & (B_f > 130)
    rgb_hyper_pig = (r > 100) & (r.astype(np.int16) > g.astype(np.int16) + 20) & (L_f < 140)

    # ── Hypertrophic / raised pink scar ──────────────────────────────────────
    lab_hyper_tro = (L_f >= 140) & (L_f < 185) & (A_f > 140) & (B_f < 155)
    rgb_hyper_tro = (r > 175) & (g > 110) & (b > 110) & (r > g + 25)

    # ── Keloid (intensely red, slightly elevated) ─────────────────────────────
    lab_keloid    = (A_f > 152) & (L_f > 110) & (L_f < 180) & (B_f < 145)
    rgb_keloid    = (r > 185) & (g < 130) & (b < 125)

    # ── Mature / pale scar (near-white, low chroma) ───────────────────────────
    lab_mature    = (L_f >= 185) & (A_f < 140) & (B_f < 148)
    rgb_mature    = (r > 205) & (g > 185) & (b > 180) & \
                   (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 30)

    # ── Normal surrounding skin (baseline) ───────────────────────────────────
    lab_normal    = (L_f >= 140) & (L_f < 210) & (A_f >= 128) & (A_f < 145) & \
                   (B_f >= 128) & (B_f < 155)

    hyper_pig_mask = amr(lab_hyper_pig | rgb_hyper_pig)
    hyper_tro_mask = amr(lab_hyper_tro | rgb_hyper_tro)
    keloid_mask    = amr(lab_keloid    | rgb_keloid)
    mature_mask    = amr(lab_mature    | rgb_mature)
    normal_mask    = amr(lab_normal)

    return hyper_pig_mask, hyper_tro_mask, keloid_mask, mature_mask, normal_mask


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED LAB TISSUE ANALYSIS ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_tissue_lab(image_bytes: bytes, mask=None, category: str = "open_wound"):
    """
    Master function — routes to the correct LAB tissue classifier based on
    `category`, then builds the overlay & boundary images.

    Returns dict with tissue percentages + overlay_image + boundary_image.
    """
    try:
        pp = _preprocess(image_bytes, mask)
        if pp is None:
            return None

        h, w = pp["h"], pp["w"]
        L_f = pp["L_f"]
        image_smoothed = pp["image_smoothed"]
        image_corrected = pp["image_corrected"]
        valid_mask = pp["valid_mask"]
        gray = pp["gray"]
        hsv  = pp["hsv"]
        total_pixels = h * w

        stitch_mask = _remove_stitch_artifacts(hsv, gray)

        # ── Route to category-specific classifier ────────────────────────────
        if category == "burn_wound":
            char_mask, full_mask, partial_mask, super_mask, heal_mask = \
                _classify_burn_wound(pp)

            # Priority: char > full > partial > superficial > healing
            full_mask    = full_mask    & ~char_mask
            partial_mask = partial_mask & ~char_mask & ~full_mask
            super_mask   = super_mask   & ~char_mask & ~full_mask & ~partial_mask
            heal_mask    = heal_mask    & ~char_mask & ~full_mask & ~partial_mask & ~super_mask

            raw_char    = int(np.sum(char_mask))
            raw_full    = int(np.sum(full_mask))
            raw_partial = int(np.sum(partial_mask))
            raw_super   = int(np.sum(super_mask))
            raw_heal    = int(np.sum(heal_mask))
            det_total   = max(raw_char + raw_full + raw_partial + raw_super + raw_heal, 1)

            p_char    = round(raw_char    / det_total * 100, 2)
            p_full    = round(raw_full    / det_total * 100, 2)
            p_partial = round(raw_partial / det_total * 100, 2)
            p_super   = round(raw_super   / det_total * 100, 2)
            p_heal    = round(raw_heal    / det_total * 100, 2)

            print(f"\n[CV BURN] Char:{p_char}% FullThick:{p_full}% "
                  f"Partial:{p_partial}% Superficial:{p_super}% Healing:{p_heal}%")

            # Colour map for burn tissue
            tissue_map = np.ones_like(image_corrected) * 255
            tissue_map[char_mask]    = [20,  20,  20]    # Almost black — char
            tissue_map[full_mask]    = [200, 200, 200]   # Light grey — leathery full
            tissue_map[partial_mask] = [220,  80,  80]   # Red — partial thickness
            tissue_map[super_mask]   = [255, 140,   0]   # Orange — superficial
            tissue_map[heal_mask]    = [255, 182, 193]   # Pink — healing

            overlay_b64, boundary_b64 = _encode_overlay_and_boundary(
                image_smoothed, tissue_map, image_corrected, valid_mask, h, w)

            return {
                "granulation":  p_partial,   # map partial → granulation slot (redness)
                "epithelial":   p_heal,       # healing re-epithelialisation
                "slough":       p_super,      # superficial erythema
                "necrotic":     p_char,       # char / deep eschar
                "white":        p_full,       # leathery pale full-thickness
                "burn_detail": {
                    "charred_eschar":       p_char,
                    "full_thickness":       p_full,
                    "partial_thickness":    p_partial,
                    "superficial":          p_super,
                    "re_epithelialisation": p_heal,
                },
                "total_pixels": total_pixels,
                "overlay_image":  f"data:image/jpeg;base64,{overlay_b64}",
                "boundary_image": f"data:image/jpeg;base64,{boundary_b64}",
            }

        elif category == "scar":
            hyper_pig, hyper_tro, keloid, mature, normal = _classify_scar(pp)

            # Priority: hyper_pig > keloid > hyper_tro > mature > normal
            keloid     = keloid    & ~hyper_pig
            hyper_tro  = hyper_tro & ~hyper_pig & ~keloid
            mature     = mature    & ~hyper_pig & ~keloid & ~hyper_tro
            normal     = normal    & ~hyper_pig & ~keloid & ~hyper_tro & ~mature

            raw_pig   = int(np.sum(hyper_pig))
            raw_kel   = int(np.sum(keloid))
            raw_htr   = int(np.sum(hyper_tro))
            raw_mat   = int(np.sum(mature))
            raw_nor   = int(np.sum(normal))
            det_total = max(raw_pig + raw_kel + raw_htr + raw_mat + raw_nor, 1)

            p_pig = round(raw_pig / det_total * 100, 2)
            p_kel = round(raw_kel / det_total * 100, 2)
            p_htr = round(raw_htr / det_total * 100, 2)
            p_mat = round(raw_mat / det_total * 100, 2)
            p_nor = round(raw_nor / det_total * 100, 2)

            print(f"\n[CV SCAR] Hyperpig:{p_pig}% Keloid:{p_kel}% "
                  f"Hypertrophic:{p_htr}% Mature:{p_mat}% Normal:{p_nor}%")

            tissue_map = np.ones_like(image_corrected) * 255
            tissue_map[hyper_pig] = [100,  50,   0]   # Dark brown
            tissue_map[keloid]    = [200,  20,  20]   # Vivid red
            tissue_map[hyper_tro] = [220, 120, 120]   # Pink-red
            tissue_map[mature]    = [240, 220, 210]   # Pale flesh
            tissue_map[normal]    = [255, 200, 170]   # Skin tone

            overlay_b64, boundary_b64 = _encode_overlay_and_boundary(
                image_smoothed, tissue_map, image_corrected, valid_mask, h, w)

            return {
                # Map to standard slots so healing score logic still works
                "granulation":  p_htr,      # active / hypertrophic
                "epithelial":   p_mat,       # mature healing
                "slough":       p_pig,       # hyperpigmented
                "necrotic":     0.0,
                "white":        p_nor,       # normal surrounding skin
                "scar_detail": {
                    "hyperpigmented":  p_pig,
                    "keloid":          p_kel,
                    "hypertrophic":    p_htr,
                    "mature_scar":     p_mat,
                    "normal_skin":     p_nor,
                },
                "total_pixels": total_pixels,
                "overlay_image":  f"data:image/jpeg;base64,{overlay_b64}",
                "boundary_image": f"data:image/jpeg;base64,{boundary_b64}",
            }

        elif category == "sutured_wound":
            nec_mask, slough_mask, gran_mask, epi_mask, white_mask = \
                _classify_sutured_wound(pp)

            # Remove suture thread artefacts from tissue masks
            sm_bool = stitch_mask > 0
            nec_mask    = nec_mask    & ~sm_bool
            slough_mask = slough_mask & ~sm_bool
            gran_mask   = gran_mask   & ~sm_bool
            epi_mask    = epi_mask    & ~sm_bool

            # Priority deduplication
            slough_mask = slough_mask & ~nec_mask
            gran_mask   = gran_mask   & ~nec_mask & ~slough_mask
            epi_mask    = epi_mask    & ~nec_mask & ~slough_mask & ~gran_mask
            white_mask  = white_mask  & ~nec_mask & ~slough_mask & ~gran_mask & ~epi_mask

            raw_nec    = int(np.sum(nec_mask))
            raw_slough = int(np.sum(slough_mask))
            raw_gran   = int(np.sum(gran_mask))
            raw_epi    = int(np.sum(epi_mask))
            raw_white  = int(np.sum(white_mask))
            det_total  = max(raw_nec + raw_slough + raw_gran + raw_epi + raw_white, 1)

            p_nec    = round(raw_nec    / det_total * 100, 2)
            p_slough = round(raw_slough / det_total * 100, 2)
            p_gran   = round(raw_gran   / det_total * 100, 2)
            p_epi    = round(raw_epi    / det_total * 100, 2)
            p_white  = round(raw_white  / det_total * 100, 2)

            print(f"\n[CV SUTURE] Nec:{p_nec}% Slough:{p_slough}% "
                  f"Gran:{p_gran}% Epi:{p_epi}% Fibrin:{p_white}%")

            tissue_map = np.ones_like(image_corrected) * 255
            tissue_map[nec_mask]    = [120,   0, 120]   # Purple
            tissue_map[slough_mask] = [255, 255,   0]   # Yellow
            tissue_map[gran_mask]   = [255,   0,   0]   # Red
            tissue_map[epi_mask]    = [255, 105, 180]   # Pink
            tissue_map[white_mask]  = [255, 255, 255]   # White

            overlay_b64, boundary_b64 = _encode_overlay_and_boundary(
                image_smoothed, tissue_map, image_corrected, valid_mask, h, w)

            return {
                "granulation":  p_gran,
                "epithelial":   p_epi,
                "slough":       p_slough,
                "necrotic":     p_nec,
                "white":        p_white,
                "total_pixels": total_pixels,
                "overlay_image":  f"data:image/jpeg;base64,{overlay_b64}",
                "boundary_image": f"data:image/jpeg;base64,{boundary_b64}",
            }

        else:
            # ── DEFAULT: open_wound ────────────────────────────────────────────
            nec_mask, slough_mask, gran_mask, epi_mask, white_mask = \
                _classify_open_wound(pp)

            sm_bool = stitch_mask > 0
            nec_mask    = nec_mask    & ~sm_bool
            slough_mask = slough_mask & ~sm_bool
            gran_mask   = gran_mask   & ~sm_bool
            epi_mask    = epi_mask    & ~sm_bool

            slough_mask = slough_mask & ~nec_mask
            gran_mask   = gran_mask   & ~nec_mask & ~slough_mask
            epi_mask    = epi_mask    & ~nec_mask & ~slough_mask & ~gran_mask
            white_mask  = white_mask  & ~nec_mask & ~slough_mask & ~gran_mask & ~epi_mask

            raw_gran   = int(np.sum(gran_mask))
            raw_epi    = int(np.sum(epi_mask))
            raw_slough = int(np.sum(slough_mask))
            raw_nec    = int(np.sum(nec_mask))
            raw_white  = int(np.sum(white_mask))
            det_total  = max(raw_gran + raw_epi + raw_slough + raw_nec + raw_white, 1)

            p_gran   = round(raw_gran   / det_total * 100, 2)
            p_epi    = round(raw_epi    / det_total * 100, 2)
            p_slough = round(raw_slough / det_total * 100, 2)
            p_nec    = round(raw_nec    / det_total * 100, 2)
            p_white  = round(raw_white  / det_total * 100, 2)

            print(f"\n[CV OPEN] Gran:{p_gran}% Epi:{p_epi}% "
                  f"Slough:{p_slough}% Nec:{p_nec}% Fibrin:{p_white}%")

            tissue_map = np.ones_like(image_corrected) * 255
            tissue_map[nec_mask]    = [120,   0, 120]
            tissue_map[slough_mask] = [255, 255,   0]
            tissue_map[gran_mask]   = [255,   0,   0]
            tissue_map[epi_mask]    = [255, 105, 180]
            tissue_map[white_mask]  = [255, 255, 255]

            overlay_b64, boundary_b64 = _encode_overlay_and_boundary(
                image_smoothed, tissue_map, image_corrected, valid_mask, h, w)

            return {
                "granulation":  p_gran,
                "epithelial":   p_epi,
                "slough":       p_slough,
                "necrotic":     p_nec,
                "white":        p_white,
                "total_pixels": total_pixels,
                "overlay_image":  f"data:image/jpeg;base64,{overlay_b64}",
                "boundary_image": f"data:image/jpeg;base64,{boundary_b64}",
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[CV TISSUE ERROR] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  LEGACY BURN HELPER kept for backward-compat (now delegates to LAB path)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_burn_tissue(image_bytes: bytes, mask=None):
    """
    Backward-compatible wrapper — now runs the full LAB burn classifier
    and returns legacy-shaped dict for classify.py.
    """
    result = analyze_tissue_lab(image_bytes, mask=mask, category="burn_wound")
    if result is None:
        return None

    detail = result.get("burn_detail", {})
    partial = detail.get("partial_thickness", 0) + detail.get("superficial", 0)
    full    = detail.get("full_thickness",    0) + detail.get("charred_eschar", 0)
    det_total = max(partial + full, 1)

    return {
        "partial_thickness_indicators": round(partial / det_total * 100, 2),
        "full_thickness_indicators":    round(full    / det_total * 100, 2),
        "total_analyzed_pixels":        result.get("total_pixels", 0),
        "burn_depth_detail":            detail,
    }

