"""
Prompt Parsing, Cleaning, Decorators, and Metadata Extractors for Shallot-CUI Bot.
"""

import io
import re
import json
import random
import logging
from PIL import Image

from parsers.styles import MAGIC_ENHANCEMENTS

logger = logging.getLogger("DiscordBot.Parsers.Prompts")

# Regular expressions
RE_SEED = re.compile(r'[-\u2014\u2013]{1,2}seed\s+(\d+)', re.IGNORECASE)
RE_POWERHOUSE = re.compile(r'[-\u2014\u2013]{1,2}(?:powerhouse|ph|refine)\b', re.IGNORECASE)
RE_FREEU = re.compile(r'[-\u2014\u2013]{1,2}(?:no[-_]?freeu|disable[-_]?freeu|raw)\b', re.IGNORECASE)
RE_CW = re.compile(r'[-\u2014\u2013]{1,2}(?:cw|cref[-_]?weight)(?:\s+|\.)?([0-9\.]+)', re.IGNORECASE)
RE_CREF = re.compile(r'[-\u2014\u2013]{1,2}cref\s+(\S+)', re.IGNORECASE)
RE_WHITESPACE = re.compile(r'\s+')
RE_WILDCARD_BLOCKS = re.compile(r'\{([^{}]+)\}')
RE_GAMBLE = re.compile(r'[-\u2014\u2013]{1,2}(?:gamble|poetic)\b', re.IGNORECASE)

RE_FLORENCE_BOILERPLATE = re.compile(
    r"^(the image shows|the image depicts|the photo shows|the photo depicts|this is an image of|this image features|the image features|in this image,|in this photo,|a photo of|an image of|a picture of|this picture shows|this picture depicts|close-up photo of|close-up shot of)\s*",
    re.IGNORECASE
)
RE_PROSE_FILLER = re.compile(
    r"\b(there is|there are|we can see|one can see|can be seen|the image captures|captures a|depicting|showing)\b",
    re.IGNORECASE
)
RE_AN_OVERALL = re.compile(r"\b(an)\s+overall\b", re.IGNORECASE)
RE_OVERALL = re.compile(r"\boverall\b", re.IGNORECASE)


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


def deduplicate_intro_quality_tags(prompt: str) -> str:
    """
    Cleans up redundant quality tags and semi-realism phrases in prompt intros,
    preventing duplicate strings like 'masterpiece, best quality, absurdres. Semi-realism, masterpiece, best quality.'
    """
    if not prompt:
        return prompt

    p = prompt.strip()

    # Clean up duplicate 'Semi-realism, masterpiece, best quality.' -> 'Semi-realism,'
    p = re.sub(r'\bSemi-realism,\s*masterpiece,\s*best quality\.?\s*', 'Semi-realism, ', p, flags=re.IGNORECASE)

    # If 'masterpiece, best quality' appears multiple times, keep only the first occurrence
    matches = list(re.finditer(r'\bmasterpiece,\s*best quality\b', p, flags=re.IGNORECASE))
    if len(matches) > 1:
        first_end = matches[0].end()
        head = p[:first_end]
        tail = p[first_end:]
        tail = re.sub(r',?\s*absurdres\.?', '', tail, flags=re.IGNORECASE)
        tail = re.sub(r',?\s*masterpiece,\s*best quality\.?', '', tail, flags=re.IGNORECASE)
        p = head + tail

    # Clean up double punctuation / spaces
    p = re.sub(r'\.\s*\.', '.', p)
    p = re.sub(r',\s*,+', ',', p)
    p = re.sub(r'\s+', ' ', p).strip()
    return p


def parse_seed(prompt: str):
    """
    Parses --seed <number> from prompt.
    Returns (cleaned_prompt, seed_int_or_None).
    """
    match = RE_SEED.search(prompt)
    if match:
        seed = int(match.group(1))
        prompt = RE_SEED.sub('', prompt).strip()
        return prompt, seed
    return prompt, None


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
    cw_match = RE_CW.search(prompt)
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
            prompt = RE_CW.sub('', prompt).strip()
        except Exception as e:
            logger.error(f"Error parsing cref weight: {e}")

    # 2. Parse --cref <url>
    cref_match = RE_CREF.search(prompt)
    if cref_match:
        val = cref_match.group(1).strip()
        prompt = RE_CREF.sub('', prompt).strip()
        if val.startswith("http://") or val.startswith("https://"):
            cref_url = val

    return prompt, cref_url, cref_weight


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


def parse_powerhouse_prompt(prompt: str):
    """
    Parses --powerhouse / --ph / --refine flag from prompt string.
    Returns (cleaned_prompt, is_powerhouse_enabled).
    """
    is_ph = False
    if RE_POWERHOUSE.search(prompt):
        is_ph = True
        prompt = RE_POWERHOUSE.sub('', prompt).strip()
    return prompt, is_ph


def parse_freeu_prompt(prompt: str):
    """
    Parses --nofreeu / --no-freeu / --raw / --disable-freeu flag from prompt string.
    Returns (cleaned_prompt, is_no_freeu_enabled).
    """
    is_no_freeu = False
    if RE_FREEU.search(prompt):
        is_no_freeu = True
        prompt = RE_FREEU.sub('', prompt).strip()
    return prompt, is_no_freeu


def parse_gamble_prompt(prompt: str, engine: str = "sdxl", rng_seed: int = None):
    """
    Parses --gamble or --poetic flag from prompt string.
    If present, extracts the seed text and synthesizes a cohesive poetic visual allegory.
    Returns (expanded_prompt, is_gamble_enabled, poetic_result_or_None).
    """
    is_gamble = False
    if RE_GAMBLE.search(prompt):
        is_gamble = True
        cleaned = RE_GAMBLE.sub('', prompt).strip()
        from parsers.poetic import synthesize_poetic_prompt
        poetic_res = synthesize_poetic_prompt(seed=cleaned, engine=engine, rng_seed=rng_seed)
        return poetic_res.prompt, True, poetic_res
    return prompt, False, None


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

    subject_modifier = ""
    if subject_type == "character":
        subject_modifier = "character focus, striking figure composition"
    elif subject_type == "scenery":
        subject_modifier = "wide panoramic vista, immersive environment"

    prompt_parts = [clean_user_prompt]
    if subject_modifier:
        prompt_parts.append(subject_modifier)
    prompt_parts.extend(pos_modifiers)

    combined_positive = ", ".join([p for p in prompt_parts if p])
    
    if ar_flag and "--ar" not in combined_positive.lower() and not re.search(r'[-—–]{1,2}(?:ar|at)?\s*\d+', combined_positive, re.I):
        combined_positive = f"{combined_positive} {ar_flag}"

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


def parse_video_motion_flags(prompt: str) -> tuple:
    """
    Parses camera and motion directive shorthand flags from a prompt.
    Returns:
      tuple: (cleaned_prompt: str, badges: list[str], augmented_prompt: str)
    """
    if not prompt or not isinstance(prompt, str):
        return "", [], ""

    text = prompt
    badges = []
    cues = []

    FLAG_RULES = [
        (r'[-—–]{1,2}zoom(?:-in)?\b', "🎥 Zoom In", "slow cinematic camera zoom in, forward push in"),
        (r'[-—–]{1,2}zoom-out\b', "🎥 Zoom Out", "slow cinematic camera zoom out, backward pull out"),
        (r'[-—–]{1,2}pan-left\b', "🎥 Pan Left", "smooth cinematic camera pan to the left"),
        (r'[-—–]{1,2}pan-right\b', "🎥 Pan Right", "smooth cinematic camera pan to the right"),
        (r'[-—–]{1,2}(?:pan-up|tilt-up)\b', "🎥 Tilt Up", "smooth camera tilt upwards, upward crane motion"),
        (r'[-—–]{1,2}(?:pan-down|tilt-down)\b', "🎥 Tilt Down", "smooth camera tilt downwards"),
        (r'[-—–]{1,2}(?:orbit|rotate)\b', "🎥 Orbit", "slow orbital camera movement, 360 rotation around subject"),
        (r'[-—–]{1,2}cinematic\b', "✨ Cinematic", "cinematic steadycam motion, high production value, dramatic lighting"),
        (r'[-—–]{1,2}subtle\b', "🍃 Subtle", "subtle gentle movement, delicate breathing, calm steady shot"),
        (r'[-—–]{1,2}(?:dynamic|fast-motion)\b', "⚡ Dynamic", "energetic dynamic motion, fast action, dramatic camera movement"),
        (r'[-—–]{1,2}(?:realtime|natural|normal-speed)\b', "⏱️ Real-Time", "real-time motion, natural speed playback, authentic lifelike movement"),
    ]

    for pattern, badge_label, cue_text in FLAG_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            badges.append(badge_label)
            cues.append(cue_text)

    cleaned = re.sub(r'\s*,\s*,+', ',', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip(',').strip()

    if cues:
        cue_suffix = ", ".join(cues)
        augmented = f"{cleaned}, {cue_suffix}" if cleaned else cue_suffix
    else:
        augmented = cleaned

    return cleaned, badges, augmented


def sanitize_describe_text(text: str) -> str:
    """
    Replaces the word 'overall' with 'general' to prevent diffusion models
    from misinterpreting the word as clothing (overalls) on the subject.
    Also handles 'an overall' -> 'a general' for grammatical fluency.
    Preserves case capitalization (Overall -> General, OVERALL -> GENERAL, overall -> general).
    """
    if not text:
        return ""

    def _replace_an(match: re.Match) -> str:
        an_word = match.group(1)
        if an_word.isupper():
            return "A GENERAL"
        elif an_word[0].isupper():
            return "A general"
        return "a general"

    def _replace_overall(match: re.Match) -> str:
        w = match.group(0)
        if w.isupper():
            return "GENERAL"
        elif w[0].isupper():
            return "General"
        return "general"

    text = RE_AN_OVERALL.sub(_replace_an, text)
    return RE_OVERALL.sub(_replace_overall, text)


def format_sdxl_prompt(raw_text: str) -> str:
    """
    Formats a raw vision description (from JoyCaption, Qwen, or Florence-2)
    into an optimized, tag-dense prompt for SDXL / Illustrious models.
    Converts descriptive prose into clean comma-separated tokens and removes filler.
    """
    if not raw_text:
        return ""
    
    text = sanitize_describe_text(raw_text.strip())
    
    while True:
        cleaned = RE_FLORENCE_BOILERPLATE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned

    text = re.sub(r"[;\n]+", ", ", text)
    text = re.sub(r"\.\s+", ", ", text)
    text = RE_PROSE_FILLER.sub("", text)
    text = re.sub(r",\s*,+", ",", text)
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    
    seen = set()
    deduped = []
    for t in tokens:
        low = t.lower()
        if low not in seen and len(low) > 1:
            seen.add(low)
            deduped.append(t)
            
    final_sdxl = ", ".join(deduped)
    return final_sdxl.strip(",. ")


def format_flux_prompt(raw_text: str) -> str:
    """
    Formats a raw vision description into high-fidelity natural language prose
    ideal for Flux.1 models. Preserves detailed spatial and physical descriptions
    while stripping robotic boilerplate and generic quality buzzwords.
    """
    if not raw_text:
        return ""
    
    text = sanitize_describe_text(raw_text.strip())
    
    while True:
        cleaned = RE_FLORENCE_BOILERPLATE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned
        
    buzzwords = [
        r"\bmasterpiece\b", r"\bbest quality\b", r"\bultra high res\b",
        r"\b8k resolution\b", r"\bphotorealistic\b", r"\bhyperrealistic\b"
    ]
    for bw in buzzwords:
        text = re.sub(bw, "", text, flags=re.IGNORECASE)
        
    text = text.lstrip(":,.- ")
    if text:
        text = text[0].upper() + text[1:]
        
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_krea2_prompt(raw_text: str) -> str:
    """
    Formats a raw vision description (e.g. from Qwen2.5-VL, JoyCaption, or Florence-2)
    into an optimized natural prose prompt for Krea 2 Turbo flow-matching and Qwen3-VL text encoder.
    Strips robotic prefixes and cleans composition phrasing.
    """
    if not raw_text:
        return ""

    text = sanitize_describe_text(raw_text.strip())

    while True:
        cleaned = RE_FLORENCE_BOILERPLATE.sub("", text).strip()
        if cleaned == text:
            break
        text = cleaned

    text = text.lstrip(":,.- ")
    if text:
        text = text[0].upper() + text[1:]

    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuse_krea2_blend_prompt(vision_prompt: str, remix_prompt: str = None) -> str:
    """
    Fuses vision analysis (Qwen2.5-VL / JoyCaption / Florence-2) with user remix
    instructions into an optimized natural prose prompt for Krea 2 Turbo flow-matching.
    """
    cleaned_vision = format_krea2_prompt(vision_prompt or "")
    if not remix_prompt:
        return cleaned_vision

    remix = remix_prompt.strip().strip(",").strip()
    if not remix:
        return cleaned_vision

    if not cleaned_vision:
        return remix

    if not remix.endswith(('.', '!', '?')):
        remix += ","
    fused = f"{remix} {cleaned_vision}"
    return fused.strip()
