"""
Style Databases, SREF Parsers, and Aesthetic Generators for Shallot-CUI Bot.
"""

import re
import random
import logging

logger = logging.getLogger("DiscordBot.Parsers.Styles")

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

RE_RAW = re.compile(r'[-\u2014\u2013]{1,2}raw\b', re.IGNORECASE)
RE_STYLIZE = re.compile(r'[-\u2014\u2013]{1,2}(?:stylize|s)\s+(\d+)', re.IGNORECASE)
RE_SW = re.compile(r'[-\u2014\u2013]{1,2}(?:sw|sref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', re.IGNORECASE)
RE_SREF = re.compile(r'[-\u2014\u2013]{1,2}sref\s+(.+?)(?=\s+[-\u2014\u2013]{1,2}[a-z]+|$)', re.IGNORECASE)


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
    raw_match = RE_RAW.search(prompt)
    if raw_match:
        prepend_quality = False
        cfg = 3.0
        prompt = RE_RAW.sub('', prompt).strip()
    
    # Check --stylize / --s
    stylize_match = RE_STYLIZE.search(prompt)
    if stylize_match:
        val = min(1000, max(0, int(stylize_match.group(1))))
        if val <= 500:
            cfg = 1.0 + (val / 500.0) * 3.0
        else:
            cfg = 4.0 + ((val - 500) / 500.0) * 8.0
        prompt = RE_STYLIZE.sub('', prompt).strip()
        
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
    sw_match = RE_SW.search(prompt)
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
            prompt = RE_SW.sub('', prompt).strip()
        except Exception as e:
            logger.error(f"Error parsing sref weight: {e}")
    
    # 2. Parse --sref <url|random|number|label>
    sref_match = RE_SREF.search(prompt)
    if sref_match:
        val = sref_match.group(1).strip()
        prompt = RE_SREF.sub('', prompt).strip()
        
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
