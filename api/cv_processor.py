import cv2
import numpy as np
import base64
import os

_segment_model = None
MODEL_H5_PATH = os.path.join(os.path.dirname(__file__), 'models', 'segment_model.h5')

def _get_segment_model():
    """Lazily load the Keras h5 model."""
    global _segment_model
    if _segment_model is None:
        if not os.path.exists(MODEL_H5_PATH):
            print(f"[CV] Segmentation model not found at {MODEL_H5_PATH}")
            return None
        import tensorflow as tf
        print(f"[CV] Loading segmentation model from {MODEL_H5_PATH}")
        _segment_model = tf.keras.models.load_model(MODEL_H5_PATH, compile=False)
        print("[CV] Segmentation model loaded successfully.")
    return _segment_model

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
        cx, cy = w // 2, h // 2

        # 1. Color Segmentation to find Erythema / Wound Bed
        # Sutured wounds typically have a red/pink inflammatory halo
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Red/Pink ranges in HSV
        lower_red1 = np.array([0, 15, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 15, 50])
        upper_red2 = np.array([179, 255, 255])

        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        color_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # 2. Morphological operations to group the wound track
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3. Find contours and keep the largest one near the center
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        max_central_area = 0
        best_contour = None

        max_dist_x = w * 0.45
        max_dist_y = h * 0.45


        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    dist_x = abs(cX - cx)
                    dist_y = abs(cY - cy)
                    if dist_x < max_dist_x and dist_y < max_dist_y:
                        if area > max_central_area:
                            max_central_area = area
                            best_contour = cnt

        incision_detected = False
        if best_contour is not None:
            # Draw the best contour and expand it substantially to catch the full incision line
            cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
            dilate_kernel = np.ones((40, 40), np.uint8)
            final_mask = cv2.dilate(final_mask, dilate_kernel, iterations=2)
            incision_detected = True
        else:
            # Fallback if no red area is found (old scars, etc.)
            # Create a vertical-ish ellipse in the center
            cv2.ellipse(final_mask, (cx, cy), (w//4, int(h//2.5)), 0, 0, 360, 255, -1)

        return {
            "incision_detected": incision_detected,
            "stitch_count": 0,
            "mask": final_mask,
            "is_scar": category == "scar"
        }

    except Exception as e:
        print(f"[CV ERROR] {e}")
        return None
def analyze_tissue_lab(image_bytes: bytes, mask=None, category=None):
    """
    Detailed tissue classification using LAB color space rules.
    1. Reflection removal
    2. CLAHE illumination correction
    3. LAB threshold-based tissue masking
    """
    try:
        # Keep original bytes for display overlay
        image_bytes_orig = image_bytes
        # 1. Load and prepare
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None
        image_h, image_w = image.shape[:2]
        total_pixels = image_h * image_w
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply mask if provided - isolate wound pixels for ACCURATE analysis
        valid_mask = None
        if mask is not None:
            # Mask should be same size as image
            if mask.shape[:2] != (image_h, image_w):
                mask = cv2.resize(mask, (image_w, image_h))
            valid_mask = (mask > 0)
            
        if valid_mask is not None:
            # Black out non-wound pixels so color corrections don't bleed from background
            image_rgb = image_rgb.copy()
            image_rgb[~valid_mask] = 0

        # 2. Remove very bright reflections (sutures / reflections)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        bright_mask = gray > 220
        if valid_mask is not None:
            # Only remove reflections within the wound, not the already-zeroed background
            bright_mask = bright_mask & valid_mask
        image_rgb[bright_mask] = 0

        # 3. Illumination correction (CLAHE on L channel)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        L = clahe.apply(L)
        lab_corrected = cv2.merge((L, A, B))
        image_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)

        # 4. Smooth noise
        image_smoothed = cv2.GaussianBlur(image_corrected, (5, 5), 0)

        # 5. Convert to LAB again for rule-based detection
        lab_final = cv2.cvtColor(image_smoothed, cv2.COLOR_RGB2LAB)
        L_f, A_f, B_f = cv2.split(lab_final)

        # 6. Tissue Detection Rules (FUSION: LAB + USER RGB)
        # Apply valid_mask if we have it, and ALWAYS ignore black pixels (L=0)
        def apply_mask_to_rule(rule_mask):
            base_mask = (L_f > 0) # Strictly ignore blacked-out background
            if valid_mask is not None:
                return rule_mask & valid_mask & base_mask
            return rule_mask & base_mask

        # --- USER RGB Rules ---
        r, g, b = cv2.split(image_smoothed)
        
        # Red (Granulation): r>150, g<100, b<100, (r-g)>50
        rgb_gran = (r > 150) & (g < 100) & (b < 100) & ((r.astype(np.int16) - g.astype(np.int16)) > 50)
        # Pink (Epithelial): r>200, g>130, b>130
        rgb_epi = (r > 200) & (g > 130) & (b > 130)
        # Yellow (Slough): Tightened thresholds
        rgb_slough = (r > 200) & (g > 180) & (b < 120) & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 40)
        # Black (Necrotic): r<60, g<60, b<60
        rgb_nec = (r < 60) & (g < 60) & (b < 60)
        # White (Fibrin): r>220, g>220, b>200, |r-g|<30
        rgb_white = (r > 220) & (g > 220) & (b > 200) & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 30)

        # --- LAB Rules ---
        # Necrotic: L < 110, A < 150, B < 160
        lab_nec = (L_f < 110) & (A_f < 150) & (B_f < 160)
        # Slough: B > 145, L > 120, A < 150 (Tightened B from 138)
        lab_slough = (B_f > 145) & (L_f > 120) & (A_f < 150)
        
        # --- STITCH / SUTURE EXCLUSION LOGIC ---
        # Detect dark structures (stitches, staples, scabs) to exclude from necrotic count
        stitch_mask = np.zeros_like(gray, dtype=np.uint8)
        try:
            # 1. Use adaptive thresholding to find small dark regions
            inv_gray = 255 - gray
            local_dark = cv2.adaptiveThreshold(inv_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
            
            # 2. Only keep pixels that are absolutely dark
            absolute_dark = gray < 80
            
            # 3. Combine and clean
            stitch_candidates = cv2.bitwise_and(local_dark, absolute_dark.astype(np.uint8) * 255)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            stitch_mask_base = cv2.morphologyEx(stitch_candidates, cv2.MORPH_OPEN, kernel_small, iterations=1)
            stitch_mask = cv2.dilate(stitch_mask_base, kernel_small, iterations=2)
            
            # 4. As an extra precaution, also use Canny for distinct thread lines and add them
            s_edges = cv2.Canny(gray, 30, 100)
            s_lines = cv2.HoughLinesP(s_edges, 1, np.pi/180, threshold=15, minLineLength=5, maxLineGap=10)
            if s_lines is not None:
                line_mask = np.zeros_like(gray, dtype=np.uint8)
                for s_line in s_lines:
                    x1, y1, x2, y2 = s_line[0]
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 4)
                stitch_mask = cv2.bitwise_or(stitch_mask, line_mask)
        except Exception as e:
            print(f"[CV ERROR] Stitch mask error: {e}")
        
        # --- SPECIALIZED OVERRIDES for Sutured/Scar ---
        is_sutured = category in ["sutured_wound", "scar", "normal_skin"]
        if is_sutured:
            # 1. Be VERY strict about Yellow (Slough) on skin - skin tone often hits this.
            lab_slough = (B_f > 160) & (L_f > 130) & (A_f < 145)
            rgb_slough = (r > 210) & (g > 190) & (b < 100) & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 25)
            
            # 2. Be VERY strict about Black (Necrotic) - staples/shadows often hit this.
            rgb_nec = (r < 30) & (g < 30) & (b < 30)
            lab_nec = (L_f < 35)
            
        # Granulation: A > 150, L < 170
        lab_gran = (A_f > 150) & (L_f < 170)
        # Epithelial: A > 135, L > 170
        lab_epi = (A_f > 135) & (L_f > 170)
        # White fibrin: L > 200
        lab_white = L_f > 200

        # Fusion: Merge LAB and RGB detection, then restrict to ROI
        nec_mask = apply_mask_to_rule(rgb_nec | lab_nec)
        
        # APPLY STITCH EXCLUSION: Remove pixels that are likely stitches from the necrotic mask
        if stitch_mask is not None:
            nec_mask = nec_mask & ~(stitch_mask > 0)

        slough_mask = apply_mask_to_rule(rgb_slough | lab_slough)
        gran_mask = apply_mask_to_rule(rgb_gran | lab_gran)
        epi_mask = apply_mask_to_rule(rgb_epi | lab_epi)
        white_mask = apply_mask_to_rule(rgb_white | lab_white)

        # 7. Calculate percentages (normalize to 100% of DETECTED tissue)
        raw_gran = np.sum(gran_mask)
        raw_epi = np.sum(epi_mask)
        raw_slough = np.sum(slough_mask)
        raw_nec = np.sum(nec_mask)
        raw_white = np.sum(white_mask)
        
        detected_total = raw_gran + raw_epi + raw_slough + raw_nec + raw_white
        if detected_total == 0: detected_total = 1
        
        p_gran = round((raw_gran / detected_total) * 100, 2)
        p_epi = round((raw_epi / detected_total) * 100, 2)
        p_slough = round((raw_slough / detected_total) * 100, 2)
        p_nec = round((raw_nec / detected_total) * 100, 2)
        p_white = round((raw_white / detected_total) * 100, 2)

        # Create tissue color map (SAME order as reference script: priority nec > slough > gran > epi > white)
        tissue_map = np.ones_like(image_corrected) * 255  # white background
        tissue_map[nec_mask] = [120, 0, 120]     # Purple (Necrotic)
        tissue_map[slough_mask] = [255, 255, 0]  # Yellow (Slough)
        tissue_map[gran_mask] = [255, 0, 0]      # Red (Granulation)
        tissue_map[epi_mask] = [255, 105, 180]   # Pink (Epithelial)
        tissue_map[white_mask] = [255, 255, 255] # White (Fibrin)

        # ── Overlay: match reference script exactly ──────────────────────────
        # Use wound_only (image_smoothed = the processed, masked wound) as base
        # addWeighted(wound_only, 0.6, tissue_map, 0.4, 0)
        overlay = cv2.addWeighted(image_smoothed, 0.6, tissue_map, 0.4, 0)

        # Draw wound boundary on a COPY of the original image (see reference: outlined = image.copy())
        # We load original for context display
        nparr_orig = np.frombuffer(image_bytes_orig, np.uint8)
        orig_display = cv2.imdecode(nparr_orig, cv2.IMREAD_COLOR)
        orig_display = cv2.cvtColor(orig_display, cv2.COLOR_BGR2RGB)
        if orig_display.shape[:2] != (image_h, image_w):
            orig_display = cv2.resize(orig_display, (image_w, image_h))

        if valid_mask is not None:
            # Draw neon green wound border on the OVERLAY image
            mask_uint8 = (valid_mask.astype(np.uint8) * 255)
            contours_data = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Handle OpenCV 3.x vs 4.x unpacking (3.x returns image, contours, hierarchy)
            contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
            
            # White halo first (thick), then red on top → clearly visible on any background
            cv2.drawContours(orig_display, contours, -1, (255, 255, 255), 9)  # White halo
            cv2.drawContours(orig_display, contours, -1, (255, 30, 30), 5)    # Bright red border
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 4)           # Green on tissue overlay


        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')

        # Also encode the wound boundary image (original + red contour) for UI display
        _, buffer_boundary = cv2.imencode('.jpg', cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR))
        boundary_b64 = base64.b64encode(buffer_boundary).decode('utf-8')

        return {
            "granulation": p_gran,
            "epithelial": p_epi,
            "slough": p_slough,
            "necrotic": p_nec,
            "white": p_white,
            "total_pixels": int(total_pixels),
            "overlay_image": f"data:image/jpeg;base64,{overlay_b64}",
            "boundary_image": f"data:image/jpeg;base64,{boundary_b64}"
        }

    except Exception as e:
        print(f"[CV TISSUE ERROR] {e}")
        return None

def segment_wound(image_bytes: bytes):
    """
    Use .h5 model to segment wound area.
    Returns: expanded_mask (np.array), wound_area_pixels (int)
    """
    try:
        model = _get_segment_model()
        if model is None: return None, 0

        # Load image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None, 0
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare for model (128x128)
        img_input = cv2.resize(image_rgb, (128, 128))
        img_input = img_input / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Predict
        if model is None:
            print("[CV SEGMENT] Model not loaded, skipping prediction.")
            return None, 0
            
        pred = model.predict(img_input, verbose=0)[0]
        # Mask with threshold 0.3
        mask = (pred > 0.3).astype(np.uint8)
        # Resize back
        mask = cv2.resize(mask, (w, h))

        # Expand wound border (25x25 kernel, 2 iterations)
        kernel = np.ones((25, 17), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=2)

        total_pixels = np.sum(expanded_mask > 0)

        return expanded_mask, int(total_pixels)

    except Exception as e:
        print(f"[CV SEGMENT ERROR] {e}")
        return None, 0

def analyze_burn_tissue(image_bytes: bytes, mask=None):
    """
    Simplified color-based analysis for burn depth indicators.
    Superficial/Partial: Red/Pink (erythema/blistering)
    Full-Thickness/Deep: White (leathery) / Black (charred)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return None

        # Just for simplicity, we do basic RGB thresholding
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply mask if provided
        valid_mask = None
        if mask is not None:
            if mask.shape[:2] != image_rgb.shape[:2]:
                mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
            valid_mask = (mask > 0)
        if valid_mask is not None:
            image_rgb = image_rgb.copy()
            image_rgb[~valid_mask] = 0
            
        r, g, b = cv2.split(image_rgb)
        
        # Simple heuristics for burn depth (Not clinical grade, just indicators)
        rgb_red = (r > 150) & (g < 100) & (b < 100) # Erythema (Superficial)
        rgb_pink = (r > 200) & (g > 130) & (b > 130) # Blister bed (Partial)
        rgb_white = (r > 200) & (g > 200) & (b > 200) # Leathery/Pale (Deep)
        rgb_black = (r < 60) & (g < 60) & (b < 60)   # Charred/Eschar (Deep)
        
        # Strictly ignore blacked-out background (R=0, G=0, B=0)
        base_mask = (r > 0) | (g > 0) | (b > 0)
        
        def apply_mask(rule):
            if valid_mask is not None:
                return rule & valid_mask & base_mask
            return rule & base_mask
            
        red_mask = apply_mask(rgb_red)
        pink_mask = apply_mask(rgb_pink)
        white_mask = apply_mask(rgb_white)
        black_mask = apply_mask(rgb_black)
        
        raw_red = np.sum(red_mask)
        raw_pink = np.sum(pink_mask)
        raw_white = np.sum(white_mask)
        raw_black = np.sum(black_mask)
        
        detected_total = raw_red + raw_pink + raw_white + raw_black
        if detected_total == 0: detected_total = 1
        
        return {
            "partial_thickness_indicators": round(((raw_red + raw_pink) / detected_total) * 100, 2),
            "full_thickness_indicators": round(((raw_white + raw_black) / detected_total) * 100, 2),
            "total_analyzed_pixels": int(detected_total)
        }

    except Exception as e:
        print(f"[CV BURN TISSUE ERROR] {e}")
        return None
