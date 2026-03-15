import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile

# Ensure truncated images can be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import io

class CLIPValidator:
    _instance = None
    _model = None
    _processor = None
    
    PROMPTS = [
        "photo of an open wound with raw tissue",
        "photo of a burn wound on skin with redness or blisters",
        "photo of a stitched surgical wound with visible sutures, staples, or threads",
        "photo of a matured flat scar on skin surface",
        "photo of a fully healed wound skin texture",
        "photo of healthy normal skin without marks",
        "photo of clear skin with no wound or scar"
    ]
    
    # Mapping for friendly responses and internal logic
    PROMPT_MAP = {
        "photo of an open wound with raw tissue": "open_wound",
        "photo of a burn wound on skin with redness or blisters": "burn_wound",
        "photo of a stitched surgical wound with visible sutures, staples, or threads": "sutured_wound",
        "photo of a matured flat scar on skin surface": "scar",
        "photo of a fully healed wound skin texture": "healed_wound",
        "photo of healthy normal skin without marks": "normal_skin",
        "photo of clear skin with no wound or scar": "no_wound"
    }

    TISSUE_PROMPTS = [
        "red granulation tissue in a wound",
        "pink epithelial tissue in a wound",
        "yellow slough tissue in a wound",
        "black necrotic tissue in a wound",
        "white fibrin tissue in a wound"
    ]

    TISSUE_MAP = {
        "red granulation tissue in a wound": "red",
        "pink epithelial tissue in a wound": "pink",
        "yellow slough tissue in a wound": "yellow",
        "black necrotic tissue in a wound": "black",
        "white fibrin tissue in a wound": "white"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPValidator, cls).__new__(cls)
            # We will try to lazy load but the first call will trigger it
        return cls._instance

    def _ensure_loaded(self):
        if self._model is None:
            try:
                # Force offline mode
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'clip')
                print(f"[CLIP] Loading local CLIP model from {model_path}...")
                
                self._model = CLIPModel.from_pretrained(model_path, local_files_only=True)
                self._processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
                print("[CLIP] CLIP model loaded successfully (Offline Mode).")
            except Exception as e:
                print(f"[CLIP] Error loading CLIP model: {e}")
                return False
        return True

    def classify(self, image_bytes):
        if not self._ensure_loaded():
            return "error", 0.0
            
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 1. Image features
            inputs_img = self._processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs_img)
                # Handle cases where model returns an object instead of raw tensor
                if not isinstance(image_features, torch.Tensor):
                    if hasattr(image_features, 'pooler_output'):
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, 'image_embeds'):
                        image_features = image_features.image_embeds
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 2. Text features
            inputs_txt = self._processor(text=self.PROMPTS, return_tensors="pt", padding=True)
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs_txt)
                # Handle cases where model returns an object instead of raw tensor
                if not isinstance(text_features, torch.Tensor):
                    if hasattr(text_features, 'pooler_output'):
                        text_features = text_features.pooler_output
                    elif hasattr(text_features, 'text_embeds'):
                        text_features = text_features.text_embeds
                
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 3. Similarity
            # Using 100.0 as logit_scale (typical for CLIP)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            idx = torch.argmax(similarity, dim=1).item()
            prediction_label = self.PROMPTS[idx]
            confidence = similarity[0][idx].item() * 100
            
            result_key = self.PROMPT_MAP.get(prediction_label, "unknown")
            print(f"[CLIP] Validation (Updated Logic): {result_key} ({confidence:.1f}%)")
            
            return result_key, confidence
        except Exception as e:
            print(f"[CLIP] Validation error: {e}")
            return "error", 0.0

    def classify_tissue(self, image_bytes):
        """
        Detect dominant tissue type using CLIP.
        Returns a composition-like dictionary where the predicted tissue has most weight.
        """
        if not self._ensure_loaded():
            return None
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Process inputs
            inputs = self._processor(text=self.TISSUE_PROMPTS, images=image, return_tensors="pt", padding=True)
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
                
            # Create a composition-like object
            composition = {
                "red": round(probs[0].item() * 100, 1),
                "pink": round(probs[1].item() * 100, 1),
                "yellow": round(probs[2].item() * 100, 1),
                "black": round(probs[3].item() * 100, 1),
                "white": round(probs[4].item() * 100, 1)
            }
            
            # Ensure it sums to exactly 100
            total = float(sum(composition.values()))
            if total > 0:
                composition = {k: float(round(float(v) / total * 100, 1)) for k, v in composition.items()}
            
            print(f"[CLIP] Tissue Composition: {composition}")
            return composition
        except Exception as e:
            print(f"[CLIP] Tissue analysis error: {e}")
            return None

# Export as a singleton
clip_validator = CLIPValidator()
