"""
Aspect Ratio & Dimension Resolvers for Shallot-CUI Bot.
"""

import re
import math
import logging

logger = logging.getLogger("DiscordBot.Parsers.Dimensions")

RE_ASPECT_RATIO = re.compile(r'[-\u2014\u2013]{1,2}(?:ar|at)?\s*(\d+(?:\.\d+)?)\s*(?:[x:/]\s*(\d+(?:\.\d+)?))?', re.IGNORECASE)
RE_WHITESPACE = re.compile(r'\s+')

BERTFLOW_ASPECT_RATIOS = {
    "1:1": (1224, 1224),
    "16:9": (1632, 920),
    "9:16": (920, 1632),
    "21:9": (1872, 800),
    "3:4": (1056, 1408),
    "4:3": (1408, 1056),
    "16:9.3": (1632, 880),
}


def parse_aspect_ratio(prompt: str, model_name: str = "", force_sdxl_res: bool = False):
    """
    Parses aspect ratio from prompt like --16:9, --21:9, --9:16, --ar 16:9, --ar 16:9.3, --ar 1920:1032, --ar 1.86:1, etc.
    Returns (cleaned_prompt, width, height).
    Dynamically scales resolution area depending on SD1.5 (~262k pixels) vs SDXL (~1M pixels).
    """
    is_sdxl = force_sdxl_res or any(kw in (model_name or "").lower() for kw in ["xl", "illustrious", "nai", "nova", "juggernaut", "wai"])
    base_area = 1048576 if is_sdxl else 262144
    width, height = (1024, 1024) if is_sdxl else (512, 512)
    max_dim = 1792 if is_sdxl else 1024

    matches = list(RE_ASPECT_RATIO.finditer(prompt))
    if matches:
        ar_match = matches[-1]
        try:
            x = float(ar_match.group(1))
            y_val = ar_match.group(2)
            y = float(y_val) if y_val else 1.0
            
            prompt = RE_ASPECT_RATIO.sub('', prompt)
            prompt = RE_WHITESPACE.sub(' ', prompt).strip()
            
            if x > 0 and y > 0:
                ratio = x / y
                target_h = math.sqrt(base_area / ratio)
                target_w = ratio * target_h
                
                # Round to nearest multiple of 64
                width = int(round(target_w / 64) * 64)
                height = int(round(target_h / 64) * 64)
                
                # Keep boundaries safe
                width = max(256, min(width, max_dim))
                height = max(256, min(height, max_dim))
        except Exception as e:
            logger.error(f"Error parsing aspect ratio: {e}")
            
    return prompt, width, height


def calculate_wan_dimensions(orig_w: int, orig_h: int, target_area: int = 399360) -> tuple:
    """
    Calculates 8GB VRAM optimized width and height for Wan 2.2 Image-to-Video generation.
    Strictly preserves original source image aspect ratio while scaling to target pixel area (~400k pixels, e.g. 832x480).
    Dimensions are rounded to nearest multiple of 16 for VAE/DiT compatibility.
    """
    if orig_w <= 0 or orig_h <= 0:
        return 832, 480
    
    aspect_ratio = orig_w / orig_h
    target_h = math.sqrt(target_area / aspect_ratio)
    target_w = target_h * aspect_ratio
    
    w = max(256, min(896, int(round(target_w / 16.0) * 16)))
    h = max(256, min(896, int(round(target_h / 16.0) * 16)))
    
    return w, h


TRUNCATED_BERTFLOW_AR_MAP = {
    "16": "16:9",
    "21": "21:9",
    "9": "9:16",
    "3": "3:4",
    "4": "4:3",
    "1": "1:1",
}


def resolve_bertflow_dimensions(prompt: str, aspect_ratio_str: str = None) -> tuple[str, int, int]:
    """
    Resolves dimensions for Bertflow (Krea 2 Turbo).
    Target base resolution is ~1.5M pixels (1224x1224).
    Returns (cleaned_prompt, width, height).
    """
    clean_p = prompt or ""
    effective_ar = aspect_ratio_str
    if effective_ar:
        clean_ar = str(effective_ar).strip()
        if clean_ar in TRUNCATED_BERTFLOW_AR_MAP:
            clean_ar = TRUNCATED_BERTFLOW_AR_MAP[clean_ar]
        if clean_ar in BERTFLOW_ASPECT_RATIOS:
            w, h = BERTFLOW_ASPECT_RATIOS[clean_ar]
            clean_p = RE_ASPECT_RATIO.sub('', clean_p)
            clean_p = RE_WHITESPACE.sub(' ', clean_p).strip()
            return clean_p, w, h

        m = re.match(r'^(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)$', clean_ar)
        if m:
            try:
                x = float(m.group(1))
                y = float(m.group(2))
                if x > 0 and y > 0:
                    ratio = x / y
                    base_area = 1498176  # 1224 * 1224
                    target_h = math.sqrt(base_area / ratio)
                    target_w = ratio * target_h
                    w = int(round(target_w / 8) * 8)
                    h = int(round(target_h / 8) * 8)
                    w = max(512, min(w, 2048))
                    h = max(512, min(h, 2048))
                    clean_p = RE_ASPECT_RATIO.sub('', clean_p)
                    clean_p = RE_WHITESPACE.sub(' ', clean_p).strip()
                    return clean_p, w, h
            except Exception as e:
                logger.error(f"Error parsing custom Bertflow aspect ratio string '{clean_ar}': {e}")

    matches = list(RE_ASPECT_RATIO.finditer(clean_p))
    if matches:
        ar_match = matches[-1]
        try:
            x = float(ar_match.group(1))
            y_val = ar_match.group(2)
            y = float(y_val) if y_val else 1.0
            clean_p = RE_ASPECT_RATIO.sub('', clean_p)
            clean_p = RE_WHITESPACE.sub(' ', clean_p).strip()
            if x > 0 and y > 0:
                ratio = x / y
                base_area = 1498176  # 1224 * 1224
                target_h = math.sqrt(base_area / ratio)
                target_w = ratio * target_h
                w = int(round(target_w / 8) * 8)
                h = int(round(target_h / 8) * 8)
                w = max(512, min(w, 2048))
                h = max(512, min(h, 2048))
                return clean_p, w, h
        except Exception as e:
            logger.error(f"Error parsing Bertflow aspect ratio from prompt: {e}")

    return clean_p, 1224, 1224
