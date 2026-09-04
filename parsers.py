import io
import json
import re
import math
import random
import logging
from PIL import Image
from model_architecture import (
    Architecture, 
    SubType, 
    detect_model_architecture, 
    resolve_lora_for_architecture, 
    validate_architecture_compatibility,
    get_architecture_badge
)

logger = logging.getLogger("DiscordBot.Parsers")

SREF_MEDIUMS = [
    "oil painting", "watercolor painting", "pencil sketch", "acrylic painting", 
    "digital illustration", "35mm photograph", "vector art", "3D render", 
    "gouache painting", "pastel drawing", "ink illustration", "screenprint", 
    "stained glass", "charcoal drawing", "linocut print", "claymation", 
    "watercolor wash", "airbrush art", "collage", "concept art",
    "fresco painting", "woodblock print", "pixel art", "graffiti stencil",
    "risograph print", "crayon drawing", "chalk art"
]

SREF_STYLES = [
    "cyberpunk", "synthwave", "gothic", "impressionist", "minimalist", "pop art", 
    "psychedelic", "surrealist", "art deco", "steampunk", "abstract expressionist", 
    "cubist", "fauvist", "baroque", "renaissance", "uio-e", "brutalist", 
    "vintage retro", "pre-raphaelite", "art nouveau", "dadaist", "constructivist",
    "vaporwave", "biopunk", "dieselpunk", "solarpunk", "cottagecore",
    "dark fantasy", "high fantasy", "sci-fi space opera", "noir detective"
]

SREF_LIGHTING = [
    "neon glow", "golden hour lighting", "dramatic chiaroscuro", "volumetric studio lighting", 
    "dreamy soft focus lighting", "harsh dramatic shadows", "luminescent bioluminescence", 
    "moody candlelit lighting", "cinematic rim lighting", "soft diffused daylight", 
    "dappled sunlight", "vibrant stage lighting", "underwater ambient light", 
    "strobe light reflections", "pale moonlight", "high-key bright lighting",
    "low-key moody lighting", "sunset glow", "northern lights reflection"
]

SREF_PALETTES = [
    "neon pink and cyan duotone", "monochrome grayscale", "jewel tones", 
    "vibrant saturated colors", "warm earthy tones", "muted vintage colors", 
    "bold primary colors", "pastel color palette", "dark moody colors", 
    "high-contrast black and white", "analogous cool colors", "rainbow gradient", 
    "sepia tones", "washed-out desaturated colors", "trippy fluorescent colors",
    "gold and obsidian", "emerald and copper", "lavender and peach"
]

SREF_TEXTURES = [
    "grainy 35mm film texture", "thick impasto paint texture", "clean sharp vector lines", 
    "rough textured paper", "VHS scanlines", "vintage halftone dot pattern", 
    "intricate cross-hatching", "smooth glossy finish", "splattered paint drops", 
    "cracked canvas glaze", "distressed grunge texture", "fine digital noise", 
    "geometric pattern overlays", "delicate ink outlines", "canvas fabric texture"
]

MAGIC_ENHANCEMENTS = [
    "cinematic lighting, ultra-detailed micro texture, sharp focus, 8k resolution, masterwork composition, vibrant contrast",
    "volumetric studio lighting, highly intricate details, masterpiece quality, dramatic atmospheric depth",
    "photorealistic render, soft rim lighting, hyper-detailed surface finish, award-winning aesthetics",
    "stunning depth of field, elegant color harmony, rich lighting highlights, professional studio polish",
    "ethereal moody lighting, detailed mist and particles, breathtaking cinematic scale, dark fantasy vibe, intricate rendering",
    "golden hour light, glowing highlights, volumetric dust motes, warm color palette, dreamlike soft focus, nostalgic atmosphere",
    "cyberpunk neon glow, high-contrast shadows, reflections in rain, intricate technical details, vibrant saturated colors",
    "concept art style, speed painting textures, dynamic brush strokes, dramatic scale, epic composition, high fantasy art",
    "analog film style, 35mm grain, vintage color grading, soft natural lighting, candid depth, intimate atmosphere",
    "dramatic chiaroscuro lighting, deep rich shadows, bright focused highlights, classical painting texture, fine art masterpiece",
    "unreal engine 5 render, raytraced reflections, highly detailed materials, subsurface scattering, next-gen graphics fidelity",
    "whimsical watercolor wash, delicate ink outlines, pastel color palette, soft hand-drawn textures, fairytale storybook illustration",
    "macro photography details, shallow depth of field, extreme texture detail, crisp focus, natural soft bokeh background",
    "vibrant anime illustration style, clean crisp line art, dynamic cell shading, bright colorful highlights, expressive character focus",
    "retro futuristic vaporwave aesthetic, pastel pink and teal lighting, wireframe grids, nostalgic 80s synthwave vibe, glitch art details",
    "dark gothic romanticism, candlelit shadows, ornate detailed textures, mysterious foggy atmosphere, elegant melancholy mood",
    "modern minimalist style, clean vector lines, flat muted colors, high design aesthetics, stark geometric composition",
    "psychedelic oil swirl textures, vibrant neon colors, surreal dreamscape distortion, abstract patterns, optical illusion details",
    "rugged hyperrealistic details, natural outdoor overcast light, dramatic textured surfaces, crisp gritty realism, raw emotion",
    "cosmic stardust glow, nebula colors, ethereal space lighting, starry background depth, celestial sci-fi concept art",
    "ancient oil canvas painting, visible heavy impasto paint strokes, cracked varnish texture, warm historical pigment tones",
    "octane render style, highly reflective metallic textures, glowing emissive details, clean futuristic 3D product shot polish",
    "soft pastel chalk drawing, blended textured strokes, delicate shading, gentle muted colors, vintage impressionist style",
    "epic movie poster composition, high action dynamic angle, dramatic rim lighting, particles and debris, professional color grade",
    "serene zen atmosphere, soft diffused light, mist-covered mountains, minimalist design, calm muted earthy color tones"
]

# Precompiled regular expressions for high-frequency prompt parsing
RE_ASPECT_RATIO = re.compile(r'[-\u2014\u2013]{1,2}(?:ar|at)?\s*(\d+(?:\.\d+)?)\s*(?:[x:/]\s*(\d+(?:\.\d+)?))?', re.IGNORECASE)
RE_SEED = re.compile(r'[-\u2014\u2013]{1,2}seed\s+(\d+)', re.IGNORECASE)
RE_RAW = re.compile(r'[-\u2014\u2013]{1,2}raw\b', re.IGNORECASE)
RE_STYLIZE = re.compile(r'[-\u2014\u2013]{1,2}(?:stylize|s)\s+(\d+)', re.IGNORECASE)
RE_SW = re.compile(r'[-\u2014\u2013]{1,2}(?:sw|sref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', re.IGNORECASE)
RE_SREF = re.compile(r'[-\u2014\u2013]{1,2}sref\s+(.+?)(?=\s+[-\u2014\u2013]{1,2}[a-z]+|$)', re.IGNORECASE)
RE_CW = re.compile(r'[-\u2014\u2013]{1,2}(?:cw|cref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', re.IGNORECASE)
RE_CREF = re.compile(r'[-\u2014\u2013]{1,2}cref\s+(\S+)', re.IGNORECASE)
RE_LORA_TAG = re.compile(r'<lora:([^>:]+)(?::([^>]+))?>')
RE_SEMI_REALISM = re.compile(r'[-\u2014\u2013]{1,2}(?:semi-realism|sr)(?:\s+|\.)?([0-9\.]+)?', re.IGNORECASE)
RE_OGARLA = re.compile(r'[-\u2014\u2013]{1,2}(?:ogarla|oga)(?:\s+|\.)?([0-9\.]+)?', re.IGNORECASE)
RE_WHITESPACE = re.compile(r'\s+')
RE_WILDCARD_BLOCKS = re.compile(r'\{([^{}]+)\}')

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

def truncate_prompt(prompt: str, max_len: int = 100) -> str:
    """Helper to truncate prompt text to avoid long walls of text in Discord messages."""
    if not prompt:
        return ""
    clean_p = prompt.strip()
    if len(clean_p) > max_len:
        return clean_p[:max_len-3].strip() + "..."
    return clean_p

def find_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, char in enumerate(shortest):
        for s in strings:
            if s[i] != char:
                return shortest[:i]
    return shortest

def find_common_suffix(strings: list[str]) -> str:
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i in range(1, len(shortest) + 1):
        char = shortest[-i]
        for s in strings:
            if s[-i] != char:
                return shortest[-i+1:] if i > 1 else ""
    return shortest

def clean_quadrant_prompts(prompts: list[str], raw_prompt: str = None) -> list[str]:
    """
    Cleans quadrant prompts for display. If raw_prompt containing {a|b|c} wildcards is provided,
    extracts ONLY the selected options inside { } for each quadrant.
    """
    if not prompts:
        return prompts
        
    if raw_prompt and "{" in raw_prompt and "}" in raw_prompt:
        blocks = re.findall(r'\{([^{}]+)\}', raw_prompt)
        if blocks:
            cleaned = []
            for p in prompts:
                picked = []
                for block in blocks:
                    options = [opt.strip() for opt in block.split('|')]
                    options.sort(key=len, reverse=True)
                    for opt in options:
                        if opt and opt in p:
                            picked.append(opt)
                            break
                if picked:
                    cleaned.append(", ".join(picked))
                else:
                    cleaned.append(p)
            return cleaned

    if len(prompts) < 2:
        return prompts

    prefix = find_common_prefix(prompts)
    suffix = find_common_suffix(prompts)
    
    cleaned = []
    for p in prompts:
        start = len(prefix)
        end = len(p) - len(suffix)
        val = p[start:end].strip().strip(",").strip()
        if not val:
            val = p
        cleaned.append(val)
    return cleaned

def parse_loras(prompt: str, is_flux: bool = False, target_arch: str = None):
    """
    Parses <lora:name:weight> tags, and also supports shorthand --sr.XX or --srXX
    mapping to Semi-realism_illustrious, and --ogarla shorthand.
    Returns (cleaned_prompt, list of (lora_name, weight_float)).
    """
    loras = []
    effective_arch = Architecture.FLUX if is_flux else (target_arch or Architecture.SDXL)
    
    # 1. Parse --sr or --semi-realism shorthand (e.g., --sr.85, --sr85, --sr 0.85, --semi-realism .75, --semi-realism)
    sr_match = re.search(r'[-—–]{1,2}(?:semi-realism|sr)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    sr_parsed = False
    if sr_match and (sr_match.group(1) or sr_match.group(0).startswith('-') or sr_match.group(0).startswith('—') or sr_match.group(0).startswith('–')):
        try:
            val_str = sr_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.70
            
            if not is_flux:
                loras.append(("Semi-realism_illustrious.safetensors", weight))
                sr_parsed = True
                if "semi-realism" not in prompt.lower():
                    prompt = f"semi-realism, {prompt}".strip()
            prompt = re.sub(r'[-—–]{1,2}(?:semi-realism|sr)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()
        except Exception as e:
            logger.error(f"Error parsing --sr/--semi-realism shorthand: {e}")

    # 2. Parse --oga/--ogarla shorthand (e.g., --oga.70, --ogarla.90, --ogarla)
    oga_match = re.search(r'[-—–]{1,2}(?:ogarla|oga)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    oga_parsed = False
    if oga_match and (oga_match.group(1) or oga_match.group(0).startswith('-') or oga_match.group(0).startswith('—') or oga_match.group(0).startswith('–')):
        try:
            val_str = oga_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85
            
            lora_name = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, weight))
            oga_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:ogarla|oga)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Harmonize trigger keywords to match model training dataset
            if is_flux:
                if "ogarlaflux" not in prompt.lower():
                    if "ogarla" in prompt.lower():
                        prompt = re.sub(r'\bogarla\b', 'ogarlaflux', prompt, flags=re.IGNORECASE)
                    else:
                        prompt = f"ogarlaflux, {prompt}".strip()
            else:
                if "ogarla" not in prompt.lower():
                    prompt = f"ogarla, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --oga/--ogarla shorthand: {e}")

    # 2.2 Keyword Fallback: Detect 'ogarla' / 'ogarlaflux' in prompt text even if user didn't write '--'
    if not oga_parsed:
        if is_flux and re.search(r'\b(?:ogarlaflux|ogarla)\b', prompt, flags=re.IGNORECASE):
            loras.append(("ogarlaflux_epoch_5.safetensors", 0.85))
            oga_parsed = True
            if "ogarlaflux" not in prompt.lower():
                prompt = re.sub(r'\bogarla\b', 'ogarlaflux', prompt, flags=re.IGNORECASE)
        elif not is_flux and re.search(r'\bogarla\b', prompt, flags=re.IGNORECASE):
            loras.append(("ogarla_epoch_5.safetensors", 0.85))
            oga_parsed = True

    # 2.3 Parse --valerie/--val shorthand (e.g., --valerie.75, --val.85, --valerie)
    val_match = re.search(r'[-—–]{1,2}(?:valerie|val)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    val_parsed = False
    if val_match and (val_match.group(1) or val_match.group(0).startswith('-') or val_match.group(0).startswith('—') or val_match.group(0).startswith('–')):
        try:
            val_str = val_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, weight))
            val_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:valerie|val)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Silently inject trained trigger 'jen'
            if "jen" not in prompt.lower():
                prompt = f"jen, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --valerie shorthand: {e}")

    # 2.4 Keyword Fallback: Detect 'valerie' / 'jen' in prompt text even if user didn't write '--'
    if not val_parsed:
        if re.search(r'\bvalerie\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            val_parsed = True
            # Silently substitute pseudonym with trained trigger for ComfyUI
            prompt = re.sub(r'\bvalerie\b', 'jen', prompt, flags=re.IGNORECASE)
        elif re.search(r'\bjen\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            val_parsed = True

    # 2.5 Parse --sully/--sul shorthand (e.g., --sully.80, --sul.85, --sully)
    sul_match = re.search(r'[-—–]{1,2}(?:sully|sul)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    sul_parsed = False
    if sul_match and (sul_match.group(1) or sul_match.group(0).startswith('-') or sul_match.group(0).startswith('—') or sul_match.group(0).startswith('–')):
        try:
            val_str = sul_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, weight))
            sul_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:sully|sul)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Silently inject trained trigger 'susa' and target traits
            susa_traits = "black hair, thin rim glasses"
            if "susa" not in prompt.lower():
                prompt = f"susa, {susa_traits}, {prompt}".strip()
            elif "thin rim glasses" not in prompt.lower():
                prompt = f"{prompt}, {susa_traits}".strip()
        except Exception as e:
            logger.error(f"Error parsing --sully shorthand: {e}")

    # 2.6 Keyword Fallback: Detect 'sully' / 'susa' in prompt text even if user didn't write '--'
    if not sul_parsed:
        susa_traits = "black hair, thin rim glasses"
        if re.search(r'\bsully\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            sul_parsed = True
            prompt = re.sub(r'\bsully\b', f'susa, {susa_traits}', prompt, flags=re.IGNORECASE)
        elif re.search(r'\bsusa\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            sul_parsed = True
            if "thin rim glasses" not in prompt.lower():
                prompt = f"{prompt}, {susa_traits}".strip()

    # 2.7 Parse --mageill / --mag shorthand with optional epoch (e.g. --mageill, --mag, --mageill3, --mag6, --mageill-e4, --mageill5.75)
    mag_match = re.search(r'[-—–]{1,2}(?:mageill|mag)(?:[-_]?e?([3-6]))?(?:(?:\s+|\.)([0-9\.]+))?', prompt, flags=re.IGNORECASE)
    mag_parsed = False
    if mag_match and (mag_match.group(1) or mag_match.group(2) or mag_match.group(0).startswith('-') or mag_match.group(0).startswith('—') or mag_match.group(0).startswith('–')):
        try:
            epoch_str = mag_match.group(1) if mag_match.group(1) else "5"
            val_str = mag_match.group(2)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_file = f"mageill_epoch_{epoch_str}.safetensors"
            lora_name = resolve_lora_for_architecture(lora_file, effective_arch)
            loras.append((lora_name, weight))
            mag_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:mageill|mag)(?:[-_]?e?[3-6])?(?:(?:\s+|\.)[0-9\.]+)?', '', prompt, flags=re.IGNORECASE).strip()

            if "mageill" not in prompt.lower():
                prompt = f"mageill, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --mageill shorthand: {e}")

    # 2.8 Keyword Fallback: Detect 'mageill' in prompt text even if user didn't write '--'
    if not mag_parsed:
        if re.search(r'\bmageill\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("mageill_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            mag_parsed = True

    # 2.9 Parse --cheri / --che shorthand with optional epoch (e.g. --cheri, --che, --cheri4, --che6, --cheri-e4, --cheri.80, --cheri4.75)
    che_match = re.search(r'[-—–]{1,2}(?:cheri|che)(?:[-_]?e?([46]))?(?:(?:\s+|\.)([0-9\.]+))?', prompt, flags=re.IGNORECASE)
    che_parsed = False
    if che_match and (che_match.group(1) or che_match.group(2) or che_match.group(0).startswith('-') or che_match.group(0).startswith('—') or che_match.group(0).startswith('–')):
        try:
            epoch_str = che_match.group(1) if che_match.group(1) else "6"
            val_str = che_match.group(2)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_file = f"cheri_epoch_{epoch_str}.safetensors"
            lora_name = resolve_lora_for_architecture(lora_file, effective_arch)
            loras.append((lora_name, weight))
            che_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:cheri|che)(?:[-_]?e?[46])?(?:(?:\s+|\.)[0-9\.]+)?', '', prompt, flags=re.IGNORECASE).strip()

            # Ensure trigger 'cheri' and required trait 'blonde hair'
            cheri_traits = "blonde hair"
            if "cheri" not in prompt.lower():
                prompt = f"cheri, {cheri_traits}, {prompt}".strip()
            elif "blonde hair" not in prompt.lower():
                prompt = f"{prompt}, {cheri_traits}".strip()
        except Exception as e:
            logger.error(f"Error parsing --cheri shorthand: {e}")

    # 2.10 Keyword Fallback: Detect 'cheri' in prompt text even if user didn't write '--'
    if not che_parsed:
        if re.search(r'\bcheri\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("cheri_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            che_parsed = True
            if "blonde hair" not in prompt.lower():
                prompt = f"{prompt}, blonde hair".strip()

    # 2.11 Keyword Fallback: Detect 'semi-realism' in prompt text even if user didn't write '--'
    if not sr_parsed and not is_flux:
        if re.search(r'\b(?:semi[- ]realism|semirealism)\b', prompt, flags=re.IGNORECASE):
            loras.append(("Semi-realism_illustrious.safetensors", 0.70))
            sr_parsed = True

    # 3. Parse standard <lora:name:weight> tags with architecture resolution
    pattern = r'<lora:([^>:]+)(?::([^>]+))?>'
    matches = re.findall(pattern, prompt)
    for name, weight in matches:
        try:
            w = float(weight) if weight else 1.0
        except ValueError:
            w = 1.0
        clean_lora_name = name.strip()
        resolved_name = resolve_lora_for_architecture(clean_lora_name, effective_arch)
        loras.append((resolved_name, w))
        
    cleaned_prompt = re.sub(pattern, '', prompt).strip()
    return cleaned_prompt, loras

def validate_workflow_loras(
    workflow: dict, 
    checkpoint_name: str, 
    loras: list
) -> tuple[bool, list[str]]:
    """
    Validates that all parsed LoRAs and active workflow LoRAs match the architecture
    of the selected checkpoint model.
    
    Returns:
        (is_valid: bool, messages: list[str])
    """
    messages = []
    is_valid = True
    
    for lora_item in (loras or []):
        lora_name = lora_item[0] if isinstance(lora_item, (list, tuple)) else str(lora_item)
        compat, msg, suggested = validate_architecture_compatibility(checkpoint_name, lora_name)
        if not compat and not suggested:
            is_valid = False
            messages.append(f"❌ {msg}")
        elif msg:
            messages.append(f"ℹ️ {msg}")

    return is_valid, messages

def apply_loras_to_workflow(workflow, loras):
    """
    Configures pre-wired static LoraLoader nodes (Node 75 Semi-realism, Node 76 Ogarla)
    and dynamically chains any extra custom LoRAs into the workflow.
    Ensures LoRA nodes are ALWAYS present in the workflow graph, setting strengths to 0.0
    when not prompted/disabled.
    """
    is_flux = ("12" in workflow and "1" in workflow and "4" not in workflow) or ("76" in workflow and workflow["76"].get("class_type") == "LoraLoaderModelOnly")
    target_arch = Architecture.FLUX if is_flux else Architecture.SDXL
    
    # 1. Reset pre-wired static LoRA nodes to 0.0 (disabled) by default
    if "75" in workflow and workflow["75"].get("class_type") == "LoraLoader":
        workflow["75"]["inputs"]["strength_model"] = 0.0
        workflow["75"]["inputs"]["strength_clip"] = 0.0
        workflow["75"]["inputs"]["lora_name"] = "Semi-realism_illustrious.safetensors"
        
    if "76" in workflow:
        if workflow["76"].get("class_type") == "LoraLoaderModelOnly":
            workflow["76"]["inputs"]["strength_model"] = 0.0
            workflow["76"]["inputs"]["lora_name"] = "ogarlaflux_epoch_5.safetensors"
        elif workflow["76"].get("class_type") == "LoraLoader":
            workflow["76"]["inputs"]["strength_model"] = 0.0
            workflow["76"]["inputs"]["strength_clip"] = 0.0
            workflow["76"]["inputs"]["lora_name"] = "ogarla_epoch_5.safetensors"

    extra_loras = []
    
    if loras:
        for lora_name, weight in loras:
            resolved_lora = resolve_lora_for_architecture(lora_name, target_arch)
            lname_clean = resolved_lora.lower()
            if "semi-realism" in lname_clean and "75" in workflow:
                workflow["75"]["inputs"]["strength_model"] = weight
                workflow["75"]["inputs"]["strength_clip"] = weight
                workflow["75"]["inputs"]["lora_name"] = "Semi-realism_illustrious.safetensors"
            elif "ogarla" in lname_clean and "76" in workflow:
                target_file = "ogarlaflux_epoch_5.safetensors" if is_flux else "ogarla_epoch_5.safetensors"
                workflow["76"]["inputs"]["strength_model"] = weight
                if "strength_clip" in workflow["76"]["inputs"]:
                    workflow["76"]["inputs"]["strength_clip"] = weight
                workflow["76"]["inputs"]["lora_name"] = target_file
            else:
                extra_loras.append((resolved_lora, weight))

    # 2. If workflow has no pre-wired Node 75/76 or there are extra custom LoRAs, dynamically chain them
    current_model_source = ["76", 0] if "76" in workflow else (["1", 0] if is_flux else ["4", 0])
    current_clip_source = ["76", 1] if ("76" in workflow and not is_flux) else (["12", 0] if is_flux else ["4", 1])

    start_node_id = 100
    for idx, (lora_name, weight) in enumerate(extra_loras):
        node_id = str(start_node_id + idx)
        if not lora_name.endswith(".safetensors") and not lora_name.endswith(".ckpt"):
            lora_name += ".safetensors"
            
        strength_clip = weight
        
        if is_flux:
            workflow[node_id] = {
                "inputs": {
                    "model": current_model_source,
                    "lora_name": lora_name,
                    "strength_model": weight
                },
                "class_type": "LoraLoaderModelOnly"
            }
            current_model_source = [node_id, 0]
        else:
            workflow[node_id] = {
                "inputs": {
                    "model": current_model_source,
                    "clip": current_clip_source,
                    "lora_name": lora_name,
                    "strength_model": weight,
                    "strength_clip": strength_clip
                },
                "class_type": "LoraLoader"
            }
            current_model_source = [node_id, 0]
            current_clip_source = [node_id, 1]

    # 3. Connect IPAdapter if present
    ipadapter_out_node_id = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "IPAdapterUnifiedLoader":
            node["inputs"]["model"] = current_model_source
        if node.get("class_type") in ["IPAdapter", "IPAdapterAdvanced"]:
            ipadapter_out_node_id = node_id

    model_after_ipadapter = [ipadapter_out_node_id, 0] if ipadapter_out_node_id else current_model_source

    # 4. Connect FreeU if present
    freeu_node_id = None
    for node_id, node in workflow.items():
        if node.get("class_type") in ["FreeU", "FreeU_V2"]:
            node["inputs"]["model"] = model_after_ipadapter
            freeu_node_id = node_id
            break

    final_sampler_model_source = [freeu_node_id, 0] if freeu_node_id else model_after_ipadapter

    # 5. Re-route KSamplers and CLIPTextEncodes
    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSampler (Efficient)"]:
            node["inputs"]["model"] = final_sampler_model_source
        elif node.get("class_type") == "CLIPTextEncode" and not is_flux:
            node["inputs"]["clip"] = current_clip_source
            
    return workflow

def apply_face_detailer_to_workflow(
    workflow: dict,
    seed: int = 123456789,
    cfg: float = 4.0,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    steps: int = 20,
    denoise: float = 0.40,
    guide_size: int = 512,
    max_size: int = 768,
    detector_model: str = "bbox/face_yolov8m.pt"
) -> dict:
    """
    Injects ComfyUI-Impact-Pack UltralyticsDetectorProvider and FaceDetailer nodes
    into an SDXL workflow, routing decoded images through face enhancement before display.
    Optimized for 8GB VRAM cards with guide_size=512 and moderate steps/denoise.
    """
    # 1. Locate VAE Decode node and Preview/Save node
    vae_decode_node_id = None
    save_or_preview_node_id = None
    for node_id, node in workflow.items():
        ctype = node.get("class_type")
        if ctype == "VAEDecode":
            vae_decode_node_id = node_id
        elif ctype in ["PreviewImage", "SaveImage"]:
            save_or_preview_node_id = node_id

    if not vae_decode_node_id:
        return workflow

    # 2. Locate active Model, Clip, and VAE sources
    model_source = None
    clip_source = None
    vae_source = None

    for node_id, node in workflow.items():
        if node.get("class_type") == "CheckpointLoaderSimple":
            vae_source = [node_id, 2]
            if not model_source:
                model_source = [node_id, 0]
            if not clip_source:
                clip_source = [node_id, 1]

    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSampler (Efficient)"]:
            model_source = node["inputs"].get("model", model_source)
            break

    if "6" in workflow and "clip" in workflow["6"].get("inputs", {}):
        clip_source = workflow["6"]["inputs"]["clip"]

    # 3. Add Detector Provider (Node "80")
    detector_node_id = "80"
    workflow[detector_node_id] = {
        "inputs": {
            "model_name": detector_model
        },
        "class_type": "UltralyticsDetectorProvider",
        "_meta": {
            "title": "Ultralytics Detector Provider (Face)"
        }
    }

    # 4. Add FaceDetailer (Node "85")
    detailer_node_id = "85"
    workflow[detailer_node_id] = {
        "inputs": {
            "image": [vae_decode_node_id, 0],
            "model": model_source,
            "clip": clip_source,
            "vae": vae_source,
            "guide_size": guide_size,
            "guide_size_for": True,
            "max_size": max_size,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "bbox_detector": [detector_node_id, 0],
            "wildcard": "",
            "cycle": 1,
            "positive": ["6", 0] if "6" in workflow else None,
            "negative": ["7", 0] if "7" in workflow else None
        },
        "class_type": "FaceDetailer",
        "_meta": {
            "title": "Face Detailer (Impact Pack - 8GB Safe)"
        }
    }

    # 5. Route PreviewImage / SaveImage to receive FaceDetailer output
    if save_or_preview_node_id and save_or_preview_node_id in workflow:
        workflow[save_or_preview_node_id]["inputs"]["images"] = [detailer_node_id, 0]

    return workflow

def parse_seed(prompt: str):
    """Parses --seed <number> from prompt. Returns (cleaned_prompt, seed_or_None)."""
    match = re.search(r'[-\u2014\u2013]{1,2}seed\s+(\d+)', prompt, flags=re.IGNORECASE)
    if match:
        seed = int(match.group(1))
        prompt = re.sub(r'[-\u2014\u2013]{1,2}seed\s+\d+', '', prompt, flags=re.IGNORECASE).strip()
        return prompt, seed
    return prompt, None

def parse_stylize(prompt: str):
    """
    Parses --stylize/--s (0-1000) and --raw from prompt.
    Returns (cleaned_prompt, cfg_scale, prepend_quality_tags).
    
    Mapping: --stylize 0 -> CFG 1.0, --stylize 500 -> CFG 4.0, --stylize 1000 -> CFG 12.0
    --raw disables quality tag prepend and sets CFG to 3.0
    """
    cfg = 4.0
    prepend_quality = True
    
    # Check --raw first
    raw_match = re.search(r'[-\u2014\u2013]{1,2}raw\b', prompt, flags=re.IGNORECASE)
    if raw_match:
        prepend_quality = False
        cfg = 3.0
        prompt = re.sub(r'[-\u2014\u2013]{1,2}raw\b', '', prompt, flags=re.IGNORECASE).strip()
    
    # Check --stylize / --s
    stylize_match = re.search(r'[-\u2014\u2013]{1,2}(?:stylize|s)\s+(\d+)', prompt, flags=re.IGNORECASE)
    if stylize_match:
        val = min(1000, max(0, int(stylize_match.group(1))))
        if val <= 500:
            cfg = 1.0 + (val / 500.0) * 3.0
        else:
            cfg = 4.0 + ((val - 500) / 500.0) * 8.0
        prompt = re.sub(r'[-\u2014\u2013]{1,2}(?:stylize|s)\s+\d+', '', prompt, flags=re.IGNORECASE).strip()
        
        # High stylize enables quality tags; low disables them
        prepend_quality = val >= 250
    
    return prompt, cfg, prepend_quality

def generate_dynamic_style(code: int):
    """
    Deterministically generates a unique style configuration based on a numeric code.
    Provides over 4.2 million possible unique style combinations.
    """
    code_int = int(code)
    rng = random.Random(code_int)
    medium = rng.choice(SREF_MEDIUMS)
    style = rng.choice(SREF_STYLES)
    lighting = rng.choice(SREF_LIGHTING)
    palette = rng.choice(SREF_PALETTES)
    texture = rng.choice(SREF_TEXTURES)
    
    style_name = f"{style.title()} {medium.title()}"
    prompt_str = f"{medium}, {style} aesthetic, {lighting}, {palette}, {texture}"
    
    return {
        "code": code_int,
        "name": style_name,
        "prompt": prompt_str
    }

def parse_sref(prompt: str):
    """
    Parses --sref <url|random|number> and optional --sw / --sref-weight <float> from prompt.
    Returns (cleaned_prompt, sref_url_or_None, sref_weight, sref_info_or_None).
    """
    sref_url = None
    sref_weight = 0.6
    sref_info = None
    
    # 1. Parse --sw or --sref-weight (e.g. --sw 0.9, --sw 0.85, --sw.9, --sw 90, --sref-weight 0.9)
    sw_match = re.search(r'[-\u2014\u2013]{1,2}(?:sw|sref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', prompt, flags=re.IGNORECASE)
    if sw_match:
        try:
            val_str = sw_match.group(1)
            if val_str.startswith('.'):
                val = float(val_str)
            elif val_str.isdigit() and float(val_str) > 1.0:
                val = float(val_str) / 100.0
            else:
                val = float(val_str)
            sref_weight = min(1.0, max(0.0, val))
            prompt = re.sub(r'[-\u2014\u2013]{1,2}(?:sw|sref[-_]?weight)(?:\s+|\.)?[0-9\.]+', '', prompt, flags=re.IGNORECASE).strip()
        except Exception as e:
            logger.error(f"Error parsing sref weight: {e}")
    
    # 2. Parse --sref <url|random|number|label>
    sref_match = re.search(r'[-\u2014\u2013]{1,2}sref\s+(.+?)(?=\s+[-\u2014\u2013]{1,2}[a-z]+|$)', prompt, flags=re.IGNORECASE)
    if sref_match:
        val = sref_match.group(1).strip()
        prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+.+?(?=\s+[-\u2014\u2013]{1,2}[a-z]+|$)', '', prompt, flags=re.IGNORECASE).strip()
        
        if val.startswith("http://") or val.startswith("https://"):
            sref_url = val
        elif "random" in val.lower() or "batch" in val.lower():
            batch_count = 1
            b_match = re.search(r'(?:random|batch)[:\s]*(\d+)', val, flags=re.IGNORECASE)
            if b_match:
                try:
                    batch_count = int(b_match.group(1))
                except ValueError:
                    batch_count = 1

            code = random.randint(100000, 999999)
            preset = generate_dynamic_style(code)
            sref_info = {"code": code, "name": preset["name"], "prompt": preset["prompt"], "batch_count": batch_count}
            prompt = f"{prompt}, {preset['prompt']}"
        else:
            digit_match = re.search(r'\b(\d{5,7})\b', val)
            if digit_match:
                code = int(digit_match.group(1))
                preset = generate_dynamic_style(code)
                sref_info = {"code": code, "name": preset["name"], "prompt": preset["prompt"]}
                prompt = f"{prompt}, {preset['prompt']}"
            elif val.isdigit():
                code = int(val)
                preset = generate_dynamic_style(code)
                sref_info = {"code": code, "name": preset["name"], "prompt": preset["prompt"]}
                prompt = f"{prompt}, {preset['prompt']}"
    
    return prompt, sref_url, sref_weight, sref_info


def clean_midjourney_flags(prompt: str) -> str:
    """
    Removes unsupported Midjourney parameter flags (--cw, --sref, --niji, --s, --stylize, --v, --c, --weird, --tile, --q, etc.)
    while preserving supported flags like --ar and --cref.
    """
    patterns = [
        r'[-\u2014\u2013]{1,2}cw(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}sref(?:\s+|<)[^\s>]+>?',
        r'[-\u2014\u2013]{1,2}sw(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}sv(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}niji(?:\s+[0-9\.]+)?',
        r'[-\u2014\u2013]{1,2}s(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}stylize(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}v(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}version(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}c(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}chaos(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}weird(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}w(?:\s+|\.)?\d+',
        r'[-\u2014\u2013]{1,2}tile\b',
        r'[-\u2014\u2013]{1,2}q(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}quality(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}iw(?:\s+|\.)?[0-9\.]+',
        r'[-\u2014\u2013]{1,2}(?:fast|relax|turbo)\b',
    ]

    cleaned = prompt
    for pat in patterns:
        cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def parse_cref(prompt: str):
    """
    Parses --cref <url> and optional --cw / --cref-weight <float> from prompt.
    Returns (cleaned_prompt, cref_url_or_None, cref_weight).
    """
    cref_url = None
    cref_weight = 0.20

    # 1. Parse --cw or --cref-weight (e.g. --cw 0.8, --cw 0.85, --cw.8, --cw 80, --cref-weight 0.8)
    cw_match = re.search(r'[-\u2014\u2013]{1,2}(?:cw|cref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', prompt, flags=re.IGNORECASE)
    if cw_match:
        try:
            val_str = cw_match.group(1)
            if val_str.startswith('.'):
                val = float(val_str)
            elif val_str.isdigit() and float(val_str) > 1.0:
                val = float(val_str) / 100.0
            else:
                val = float(val_str)
            cref_weight = min(1.0, max(0.0, val))
            prompt = re.sub(r'[-\u2014\u2013]{1,2}(?:cw|cref[-_]?weight)(?:\s+|\.)?[0-9\.]+', '', prompt, flags=re.IGNORECASE).strip()
        except Exception as e:
            logger.error(f"Error parsing cref weight: {e}")

    # 2. Parse --cref <url>
    cref_match = re.search(r'[-\u2014\u2013]{1,2}cref\s+(\S+)', prompt, flags=re.IGNORECASE)
    if cref_match:
        val = cref_match.group(1).strip()
        prompt = re.sub(r'[-\u2014\u2013]{1,2}cref\s+\S+', '', prompt, flags=re.IGNORECASE).strip()
        if val.startswith("http://") or val.startswith("https://"):
            cref_url = val

    return prompt, cref_url, cref_weight

def apply_ipadapter_to_workflow(workflow: dict, image_name: str, weight: float = 0.20, preset: str = None, node_prefix: str = "cref"):
    """
    Dynamically injects an IP-Adapter node chain into a workflow dict for a reference image (e.g. --cref or --sref).
    """
    # Guard against FLUX workflows where SDXL IPAdapter is not supported and Node 4 doesn't exist
    is_flux = ("12" in workflow and "1" in workflow and "4" not in workflow) or ("76" in workflow and workflow["76"].get("class_type") == "LoraLoaderModelOnly")
    if is_flux:
        return workflow

    # Locate the KSampler node
    ksampler_node_id = None
    if "3" in workflow and workflow["3"].get("class_type") in ["KSampler", "KSampler (Efficient)", "KSamplerAdvanced"]:
        ksampler_node_id = "3"
    else:
        for nid, node in workflow.items():
            if node.get("class_type") in ["KSampler", "KSampler (Efficient)", "KSamplerAdvanced"]:
                ksampler_node_id = nid
                break

    if not ksampler_node_id or "inputs" not in workflow[ksampler_node_id]:
        return workflow

    # Determine current model source safely without blindly pointing to non-existent nodes
    current_model = workflow[ksampler_node_id]["inputs"].get("model")
    if not current_model:
        if "76" in workflow:
            current_model = ["76", 0]
        elif "4" in workflow:
            current_model = ["4", 0]
        else:
            for nid, node in workflow.items():
                if node.get("class_type") in ["CheckpointLoaderSimple", "UNETLoader"]:
                    current_model = [nid, 0]
                    break

    if not current_model:
        return workflow

    loader_id = f"{node_prefix}_loader_20"
    load_img_id = f"{node_prefix}_img_21"
    ip_node_id = f"{node_prefix}_ip_23"

    actual_preset = preset or "PLUS (high strength)"

    workflow[loader_id] = {
        "inputs": {
            "model": current_model,
            "preset": actual_preset
        },
        "class_type": "IPAdapterUnifiedLoader",
        "_meta": {"title": f"IPAdapter Loader ({node_prefix.upper()})"}
    }

    workflow[load_img_id] = {
        "inputs": {
            "image": image_name,
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": f"Load Image ({node_prefix.upper()})"}
    }

    workflow[ip_node_id] = {
        "inputs": {
            "model": [loader_id, 0],
            "ipadapter": [loader_id, 1],
            "image": [load_img_id, 0],
            "weight": weight,
            "weight_type": "ease in-out" if node_prefix == "cref" else "linear",
            "combine_embeds": "concat",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "K+V"
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {"title": f"IPAdapter ({node_prefix.upper()})"}
    }

    workflow[ksampler_node_id]["inputs"]["model"] = [ip_node_id, 0]
    return workflow

def expand_dynamic_prompt(text: str, rng: random.Random = None) -> str:
    """
    Expands dynamic prompt wildcards like {a|b|c} recursively using provided RNG.
    Example: 'a {comfy forest nook|city street|steak house|butcher shop}' -> 'a city street'
    """
    _rng = rng or random
    pattern = re.compile(r'\{([^{}]+)\}')
    
    while True:
        match = pattern.search(text)
        if not match:
            break
        options = match.group(1).split('|')
        choice = _rng.choice(options).strip()
        text = text[:match.start()] + choice + text[match.end():]
        
    return text

def parse_magic_prompt(prompt: str):
    """
    Parses --magic / --mp flag from prompt string.
    Returns (cleaned_prompt, is_magic_enabled).
    """
    is_magic = False
    magic_match = re.search(r'[-\u2014\u2013]{1,2}(?:magic|mp)\b', prompt, flags=re.IGNORECASE)
    if magic_match:
        is_magic = True
        prompt = re.sub(r'[-\u2014\u2013]{1,2}(?:magic|mp)\b', '', prompt, flags=re.IGNORECASE).strip()
    
    return prompt, is_magic

def parse_smart_prompt(prompt: str):
    """
    Parses --smart / --sm flag from prompt string.
    Returns (cleaned_prompt, is_smart_enabled).
    """
    is_smart = False
    smart_match = re.search(r'[-\u2014\u2013]{1,2}(?:smart|sm)\b', prompt, flags=re.IGNORECASE)
    if smart_match:
        is_smart = True
        prompt = re.sub(r'[-\u2014\u2013]{1,2}(?:smart|sm)\b', '', prompt, flags=re.IGNORECASE).strip()
    return prompt, is_smart

def apply_smart_magic_and_sref(prompt: str, is_flux: bool = False):
    """
    Smart Art Director Engine:
    Analyzes prompt subject keywords to generate a subject-harmonized Smart Magic expansion
    and pairs it with a matching Smart Sref style code.
    Returns (smart_prompt, recommended_sref_code_or_None).
    """
    p_lower = prompt.lower()
    
    # 1. Cyberpunk / Sci-Fi / Mecha
    if any(k in p_lower for k in ["cyberpunk", "robot", "mecha", "futuristic", "neon", "spaceship", "cyber", "tech", "sci-fi", "android"]):
        enhancement = "futuristic neon reflections, cinematic anamorphic lens flare, dark wet pavement, high contrast volumetric lighting"
        sref_code = "113408"
    # 2. Epic Fantasy / Mythical / Medieval
    elif any(k in p_lower for k in ["dragon", "knight", "castle", "magic", "wizard", "elf", "sword", "fantasy", "enchanted", "mythical", "dungeon"]):
        enhancement = "intricate ornate detail, ethereal morning mist, volumetric golden light rays, atmospheric cinematic depth"
        sref_code = "405912"
    # 3. Cozy / Whimsical / Cute / Anime
    elif any(k in p_lower for k in ["cute", "ghost", "cat", "dog", "plushie", "cozy", "chibi", "sweet", "pastel", "sunflower", "kawaii"]):
        enhancement = "soft warm ambient lighting, cozy atmosphere, delicate pastel tones, gentle depth of field"
        sref_code = "772109"
    # 4. Photorealistic / Portrait / Character
    elif any(k in p_lower for k in ["photo", "portrait", "photorealistic", "ogarla", "woman", "man", "model", "cinematic", "person", "girl", "guy"]):
        enhancement = "shot on 35mm lens, natural rembrandt lighting, subtle catchlight in eyes, shallow depth of field, 8k professional portrait"
        sref_code = "884210"
    # 5. Retro / 80s / Synthwave
    elif any(k in p_lower for k in ["80s", "retro", "arcade", "synthwave", "vaporwave", "vintage", "pixel art", "90s"]):
        enhancement = "retro 80s aesthetic, glowing neon grid, nostalgic chromatic aberration, vibrant synthwave contrast"
        sref_code = "552104"
    # 6. Default / General Art
    else:
        enhancement = "masterpiece, volumetric studio lighting, rich color palette, ultra-detailed composition"
        sref_code = "123456"

    enhanced_prompt = f"{prompt}, {enhancement}"
    recommended_sref = None if is_flux else sref_code
    return enhanced_prompt, recommended_sref

def apply_magic_enhancement(prompt: str, seed: int) -> str:
    """Enhances prompt with magic artistic descriptors deterministically based on seed."""
    rng = random.Random(seed)
    enhancement = rng.choice(MAGIC_ENHANCEMENTS)
    return f"{prompt}, {enhancement}"

def calculate_wan_dimensions(orig_w: int, orig_h: int, target_area: int = 399360) -> tuple:
    """
    Calculates 8GB VRAM optimized width and height for Wan 2.2 Image-to-Video generation.
    Strictly preserves original source image aspect ratio while scaling to target pixel area (~400k pixels, e.g. 832x480).
    Dimensions are rounded to nearest multiple of 16 for VAE/DiT compatibility.
    """
    import math
    if orig_w <= 0 or orig_h <= 0:
        return 832, 480
    
    aspect_ratio = orig_w / orig_h
    target_h = math.sqrt(target_area / aspect_ratio)
    target_w = target_h * aspect_ratio
    
    w = max(256, min(896, int(round(target_w / 16.0) * 16)))
    h = max(256, min(896, int(round(target_h / 16.0) * 16)))
    
    return w, h


def _clean_extracted_prompt(text: str) -> str:
    """Cleans and unwraps extracted prompt string."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def _is_negative_only(text: str) -> bool:
    """Returns True if the extracted text is exclusively a negative prompt."""
    if not text:
        return True
    lower = text.lower().strip()
    if lower.startswith("negative prompt:"):
        return True
    neg_words = {"blurry", "low quality", "worst quality", "ugly", "bad anatomy", "deformed", "disfigured", "bad hands", "mutated"}
    words = [w.strip() for w in lower.split(",")]
    if len(words) > 0 and all(w in neg_words for w in words):
        return True
    return False


def _parse_a1111_parameters(params: str) -> str:
    """Parses A1111 / WebUI / standard parameters chunk for positive prompt."""
    if not params or not isinstance(params, str):
        return ""
    params = params.strip()
    
    match = re.split(r'\n?\s*Negative prompt:\s*', params, flags=re.IGNORECASE)
    if len(match) > 1:
        positive_part = match[0].strip()
        positive_part = re.split(r'\n?\s*Steps:\s*\d+', positive_part, flags=re.IGNORECASE)[0].strip()
        return _clean_extracted_prompt(positive_part)
        
    match_steps = re.split(r'\n?\s*Steps:\s*\d+', params, flags=re.IGNORECASE)
    if len(match_steps) > 1:
        positive_part = match_steps[0].strip()
        return _clean_extracted_prompt(positive_part)
        
    return _clean_extracted_prompt(params)


def _resolve_comfy_node_text(node_id: str, prompt_dict: dict, visited: set = None) -> str:
    """Recursively resolves text from ComfyUI API prompt graph node."""
    if visited is None:
        visited = set()
    node_id_str = str(node_id)
    if node_id_str in visited:
        return ""
    visited.add(node_id_str)
    
    node = prompt_dict.get(node_id_str)
    if not isinstance(node, dict):
        return ""
        
    class_type = node.get("class_type", "")
    inputs = node.get("inputs", {})
    if not isinstance(inputs, dict):
        return ""
        
    for text_key in ["text", "string", "prompt"]:
        val = inputs.get(text_key)
        if isinstance(val, str) and val.strip():
            return _clean_extracted_prompt(val)
        elif isinstance(val, list) and len(val) == 2:
            resolved = _resolve_comfy_node_text(val[0], prompt_dict, visited)
            if resolved:
                return resolved
                
    text_g = inputs.get("text_g")
    text_l = inputs.get("text_l")
    parts = []
    if isinstance(text_g, str) and text_g.strip():
        parts.append(text_g.strip())
    elif isinstance(text_g, list) and len(text_g) == 2:
        res = _resolve_comfy_node_text(text_g[0], prompt_dict, visited)
        if res:
            parts.append(res)
        
    if isinstance(text_l, str) and text_l.strip():
        parts.append(text_l.strip())
    elif isinstance(text_l, list) and len(text_l) == 2:
        res = _resolve_comfy_node_text(text_l[0], prompt_dict, visited)
        if res and res not in parts:
            parts.append(res)
        
    if parts:
        if len(parts) == 2 and parts[0] == parts[1]:
            return _clean_extracted_prompt(parts[0])
        return _clean_extracted_prompt(", ".join(parts))
        
    for input_k, input_v in inputs.items():
        if isinstance(input_v, list) and len(input_v) == 2:
            resolved = _resolve_comfy_node_text(input_v[0], prompt_dict, visited)
            if resolved and not _is_negative_only(resolved):
                return resolved
                
    return ""


def _parse_comfy_api_prompt(prompt_dict: dict) -> str:
    """Extracts positive prompt from ComfyUI API prompt JSON dictionary."""
    if not isinstance(prompt_dict, dict):
        return ""
        
    sampler_nodes = []
    for nid, ndata in prompt_dict.items():
        if isinstance(ndata, dict):
            ctype = ndata.get("class_type", "")
            if any(k in ctype for k in ["KSampler", "SamplerCustom", "WanImageToVideo", "Sampler"]):
                sampler_nodes.append((nid, ndata))
                
    for nid, ndata in sampler_nodes:
        inputs = ndata.get("inputs", {})
        if isinstance(inputs, dict):
            pos_link = inputs.get("positive") or inputs.get("positive_prompt") or inputs.get("positive_conditioning")
            if isinstance(pos_link, list) and len(pos_link) == 2:
                text = _resolve_comfy_node_text(pos_link[0], prompt_dict)
                if text and not _is_negative_only(text):
                    return text
            elif isinstance(pos_link, str) and pos_link.strip():
                return _clean_extracted_prompt(pos_link)
                
    candidates = []
    for nid, ndata in prompt_dict.items():
        if isinstance(ndata, dict):
            ctype = ndata.get("class_type", "")
            if any(k in ctype for k in ["CLIPTextEncode", "TextEncode", "Wildcard", "Prompt"]):
                text = _resolve_comfy_node_text(nid, prompt_dict)
                if text and not _is_negative_only(text):
                    candidates.append(text)
                    
    if candidates:
        return candidates[0]
        
    return ""


def _parse_comfy_workflow(wf_dict: dict) -> str:
    """Extracts positive prompt from ComfyUI UI workflow JSON dictionary."""
    if not isinstance(wf_dict, dict):
        return ""
    nodes = wf_dict.get("nodes")
    if not isinstance(nodes, list):
        return ""
        
    positive_candidates = []
    other_candidates = []
    
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ntype = node.get("type", "")
        title = (node.get("title") or "").lower()
        widgets = node.get("widgets_values")
        
        if any(k in ntype for k in ["CLIPTextEncode", "TextEncode", "Wildcard", "Prompt", "Text"]):
            if isinstance(widgets, list):
                for w in widgets:
                    if isinstance(w, str) and w.strip():
                        cleaned = _clean_extracted_prompt(w)
                        if cleaned and not _is_negative_only(cleaned):
                            if "positive" in title:
                                positive_candidates.append(cleaned)
                            elif "negative" not in title:
                                other_candidates.append(cleaned)
                                
    if positive_candidates:
        return positive_candidates[0]
    if other_candidates:
        return other_candidates[0]
    return ""


def _parse_comment_or_description(val: str) -> str:
    """Parses JSON or parameter text from Comment/Description metadata."""
    if not val or not isinstance(val, str):
        return ""
    val = val.strip()
    if val.startswith("{") and val.endswith("}"):
        try:
            data = json.loads(val)
            if isinstance(data, dict):
                for key in ["prompt", "Positive Prompt", "positive_prompt", "description", "caption"]:
                    if key in data and isinstance(data[key], str) and data[key].strip():
                        return _clean_extracted_prompt(data[key])
        except Exception:
            pass
    return _parse_a1111_parameters(val)


def extract_positive_prompt(image_bytes: bytes) -> str:
    """
    Extracts the positive prompt used to generate an image from embedded metadata.
    Supports Automatic1111/WebUI, ComfyUI (prompt JSON & workflow JSON), NovelAI,
    Fooocus, InvokeAI, SwarmUI, EXIF tags, and standard PNG parameter chunks.
    Returns the extracted positive prompt string, or "NOT FOUND" if no positive prompt is found.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = img.info or {}
    except Exception as e:
        logger.error(f"Error opening image for prompt extraction: {e}")
        return "NOT FOUND"

    # 1. Check A1111 / WebUI / Standard PNG 'parameters' chunk
    if "parameters" in info and isinstance(info["parameters"], str) and info["parameters"].strip():
        params = info["parameters"].strip()
        positive = _parse_a1111_parameters(params)
        if positive and not _is_negative_only(positive):
            return positive

    # 2. Check ComfyUI API Prompt graph JSON ('prompt' key)
    if "prompt" in info:
        prompt_val = info["prompt"]
        try:
            if isinstance(prompt_val, str):
                prompt_dict = json.loads(prompt_val)
            elif isinstance(prompt_val, dict):
                prompt_dict = prompt_val
            else:
                prompt_dict = None

            if isinstance(prompt_dict, dict):
                positive = _parse_comfy_api_prompt(prompt_dict)
                if positive and not _is_negative_only(positive):
                    return positive
        except Exception as e:
            logger.debug(f"Error parsing ComfyUI prompt JSON: {e}")

    # 3. Check ComfyUI UI Workflow graph JSON ('workflow' key)
    if "workflow" in info:
        wf_val = info["workflow"]
        try:
            if isinstance(wf_val, str):
                wf_dict = json.loads(wf_val)
            elif isinstance(wf_val, dict):
                wf_dict = wf_val
            else:
                wf_dict = None

            if isinstance(wf_dict, dict):
                positive = _parse_comfy_workflow(wf_dict)
                if positive and not _is_negative_only(positive):
                    return positive
        except Exception as e:
            logger.debug(f"Error parsing ComfyUI workflow JSON: {e}")

    # 4. Check NovelAI / SwarmUI / Comment JSON ('Comment' or 'comment' key)
    for comment_key in ["Comment", "comment", "DESCRIPTION", "description"]:
        if comment_key in info:
            val = info[comment_key]
            if isinstance(val, str) and val.strip():
                positive = _parse_comment_or_description(val)
                if positive and not _is_negative_only(positive):
                    return positive

    # 5. Check direct keys ('Positive Prompt', 'positive_prompt', 'prompt_text', 'Dream')
    for direct_key in ["Positive Prompt", "positive_prompt", "prompt_text", "Prompt", "prompt", "Dream", "Software"]:
        if direct_key in info and direct_key not in ("prompt", "workflow"):
            val = info[direct_key]
            if isinstance(val, str) and val.strip():
                positive = _clean_extracted_prompt(val)
                if positive and not _is_negative_only(positive):
                    return positive

    # 6. Check EXIF tags (UserComment, ImageDescription)
    try:
        exif = img.getexif()
        if exif:
            for tag_id in (0x9286, 0x010e):
                val = exif.get(tag_id)
                if val:
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8", errors="ignore")
                        except Exception:
                            val = ""
                    if isinstance(val, str) and val.strip():
                        val = re.sub(r'^(ASCII|UNICODE|JIS)\x00*', '', val, flags=re.IGNORECASE).strip()
                        positive = _parse_comment_or_description(val) or _parse_a1111_parameters(val)
                        if positive and not _is_negative_only(positive):
                            return positive
    except Exception as e:
        logger.debug(f"Error checking EXIF tags: {e}")

    return "NOT FOUND"


# ---------------------------------------------------------------------------
# LOCKED STYLE PRESETS & SCAPES GENERATOR
# ---------------------------------------------------------------------------

LOCKED_STYLE_PRESETS = {
    "junji_ito": {
        "name": "Junji Ito (Horror Manga Ink)",
        "positive": "Japanese horror manga illustration by Junji Ito style, clinical anatomical line precision, fine G-nib pen strokes, dense mechanical parallel cross-hatching, stark high-contrast black ink fills, visceral body horror, hypnotic Uzumaki spiral motifs, clinical beauty juxtaposed with uncanny psychological dread, heavy black shadow pools, 1990s vintage manga screentone texture",
        "negative": "color, vibrant hues, 3d render, photo, realistic, smooth airbrush gradients, digital glow",
    },
    "martine_johanna": {
        "name": "Martine Johanna (Pastel Surreal Portraiture)",
        "positive": "figurative contemporary portrait in Martine Johanna style, acrylic and oil on raw linen canvas, prismatic pastel color spectrum, unmixed small color strokes, warm and cool tone contrast, delicate fluid linework interrupting polished facial features, dreamy ethereal female gaze, light prisms, soft muted color blocking, pop surrealism",
        "negative": "harsh black manga lines, dark horror, monochromatic grayscale, 3d CGI, photorealistic render",
    },
    "ito_johanna_fusion": {
        "name": "Junji Ito + Martine Johanna Hybrid",
        "positive": "masterful fusion of Junji Ito horror manga line art and Martine Johanna prismatic pastel portraiture, fine G-nib ink cross-hatching combined with unmixed acrylic pastel strokes on raw linen, hypnotic Uzumaki spirals rendered in delicate lavender and mint color blocking, clinical anatomical precision meets dreamy pop surrealism, stark ink shadow pools softened by light prism reflections",
        "negative": "flat 2d cartoon, glossy 3d render, plastic texture, low resolution, blurry, harsh neon",
    },
    "dark_fantasy_landscape": {
        "name": "Dark Fantasy Landscape",
        "positive": "epic moody gothic landscape, towering spires, dramatic volumetric fog, dark fantasy artwork, detailed matte painting, rich chiaroscuro lighting, cinematic scale, atmospheric depth",
        "negative": "bright happy sunshine, cartoon, flat, low detail, saturated neon",
    },
    "cyberpunk_cityscape": {
        "name": "Cyberpunk Cityscape",
        "positive": "futuristic cyberpunk metropolis, rain-slicked streets, towering neon monoliths, holographic signs, high contrast dark cinematic lighting, dense sci-fi city architecture, reflections",
        "negative": "pastoral, medieval, natural foliage, bright daylight, sepia",
    },
    "ethereal_portrait": {
        "name": "Ethereal Portrait",
        "positive": "ethereal fine art portrait, soft dreamy focus, gentle pastel tones, luminous ambient glow, delicate features, graceful composition, high fashion magazine aesthetic",
        "negative": "harsh shadows, gritty realism, grotesque, heavy black lines, noise",
    },
}


def build_scapes_prompt(
    user_prompt: str,
    style: str,
    secondary_style: str = None,
    mode: str = "landscape",
    subject_type: str = "scenery",
    sref_url: str = None,
) -> dict:
    """
    Builds an enriched prompt for the /scapes command, locking into specific artist styles or blends.
    Returns dict with keys: 'final_prompt', 'style_name', 'aspect_ratio_flag', 'positive_additions', 'negative_additions'.
    """
    clean_user_prompt = (user_prompt or "").strip()
    primary_info = LOCKED_STYLE_PRESETS.get(style, LOCKED_STYLE_PRESETS.get("junji_ito"))

    style_names = [primary_info["name"]]
    pos_modifiers = [primary_info["positive"]]
    neg_modifiers = [primary_info["negative"]]

    if secondary_style and secondary_style in LOCKED_STYLE_PRESETS and secondary_style != style:
        sec_info = LOCKED_STYLE_PRESETS[secondary_style]
        style_names.append(sec_info["name"])
        pos_modifiers.append(sec_info["positive"])
        neg_modifiers.append(sec_info["negative"])

    # Aspect ratio flag
    ar_map = {
        "ultrawide": "--ar 21:9",
        "21:9": "--ar 21:9",
        "landscape": "--ar 16:9",
        "16:9": "--ar 16:9",
        "taskbar": "--ar 1920:1032",
        "1920:1032": "--ar 1920:1032",
        "ipad": "--ar 10:7",
        "10:7": "--ar 10:7",
        "portrait_3_5": "--ar 3:5",
        "3:5": "--ar 3:5",
        "portrait": "--ar 9:16",
        "9:16": "--ar 9:16",
    }
    mode_str = (mode or "").lower()
    ar_flag = ar_map.get(mode_str, "--ar 16:9") if mode_str else None

    # Subject type modifier
    subject_modifier = ""
    if subject_type == "character":
        subject_modifier = "character focus, striking figure composition"
    elif subject_type == "scenery":
        subject_modifier = "wide panoramic vista, immersive environment"

    # Assemble final prompt text
    prompt_parts = [clean_user_prompt]
    if subject_modifier:
        prompt_parts.append(subject_modifier)
    prompt_parts.extend(pos_modifiers)

    combined_positive = ", ".join([p for p in prompt_parts if p])
    
    # Append --ar flag if provided and not already specified by user
    if ar_flag and "--ar" not in combined_positive.lower() and not re.search(r'[-—–]{1,2}(?:ar|at)?\s*\d+', combined_positive, re.I):
        combined_positive = f"{combined_positive} {ar_flag}"

    # Append --sref if provided
    sref_added = None
    if sref_url and sref_url.strip().startswith("http"):
        sref_added = sref_url.strip()
        if "--sref" not in combined_positive.lower():
            combined_positive = f"{combined_positive} --sref {sref_added}"

    combined_negative = ", ".join(set(neg_modifiers))
    display_style_name = " + ".join(style_names)

    return {
        "final_prompt": combined_positive,
        "style_name": display_style_name,
        "aspect_ratio_flag": ar_flag,
        "negative_additions": combined_negative,
        "sref_url": sref_added,
    }



