import os
import io
import json
import zipfile
import logging
import random
import re
from PIL import Image
import db

logger = logging.getLogger("DiscordBot.LoraDataset")

DATASET_ROOT_DIR = "datasets"

# =========================================================================
# LoRA Training Shot Matrix & Prompt Ideas for SDXL Character Training
# =========================================================================

def is_anime_checkpoint(checkpoint_name: str = None) -> bool:
    """Detects whether a checkpoint model is an anime/illustrious/danbooru architecture."""
    if not checkpoint_name:
        return True # Default to anime for Wai Illustrious
    name = str(checkpoint_name).lower()
    return any(k in name for k in ["illustrious", "wai", "hyphoria", "nai", "furry", "anime", "pony"])

LORA_SHOT_MATRIX_REALISTIC = [
    {
        "name": "📸 Studio Face Portrait",
        "category": "close_up",
        "template": "extreme close-up portrait of {trigger_word}, neutral relaxed expression, looking directly at camera, soft studio rim lighting, neutral clean minimalist backdrop, sharp focus, 8k uhd, highly detailed skin texture, photorealistic"
    },
    {
        "name": "😊 Cheerful Smiling Bust",
        "category": "portrait",
        "template": "upper body portrait of {trigger_word} with a warm genuine smile, bright eyes, daytime soft outdoor sunlight, gentle depth of field, sharp face details, masterwork, beautiful composition"
    },
    {
        "name": "⚔️ Dynamic Combat Action",
        "category": "action_full_body",
        "template": "dynamic action full-body shot of {trigger_word}, mid-motion combat action pose, holding adventurer gear, intense focused expression, wind blowing hair, cinematic dramatic lighting, particles in air, epic atmosphere"
    },
    {
        "name": "🌲 Palworld Lush Wilderness",
        "category": "environment",
        "template": "medium shot of {trigger_word} exploring a lush green fantasy wilderness, vibrant flora and rolling hills, soft golden sunbeams filtering through trees, adventurer attire, looking slightly to the side, rich colors"
    },
    {
        "name": "🌅 Golden Hour Side Profile",
        "category": "side_profile",
        "template": "side profile portrait of {trigger_word}, gazing at the distant horizon, golden hour sunset rim light, cinematic warm lens flare, contemplative expression, sharp silhouette and facial contours"
    },
    {
        "name": "⛺ Night Campfire Glow",
        "category": "lighting",
        "template": "waist-up shot of {trigger_word} resting near a glowing campfire at night, warm orange firelight illuminating face and outfit, dark starry sky in background, cozy relaxed ambiance, subtle depth"
    },
    {
        "name": "👑 Heroic 3/4 Turn",
        "category": "three_quarter",
        "template": "heroic three-quarter turn full-body shot of {trigger_word}, standing confident with hands on hips, tactical adventure outfit, high contrast rim lighting, grand fantasy temple ruins in background, sharp 8k"
    },
    {
        "name": "🌧️ Moody Rain & Fog",
        "category": "mood",
        "template": "cinematic medium close-up of {trigger_word} in light rain and misty fog, wet hair strands, serious determined gaze, cool blue cinematic lighting, water droplets on clothes, shallow depth of field"
    },
    {
        "name": "👀 Over-The-Shoulder View",
        "category": "angle",
        "template": "over-the-shoulder view of {trigger_word} turning back to look at the camera with an intriguing expression, dynamic camera perspective, soft bokeh background, highly detailed features"
    },
    {
        "name": "🏛️ Ancient Stone Ruins",
        "category": "environment",
        "template": "full-body shot of {trigger_word} standing beside ancient mossy stone pillars, fantasy architecture, sunbeams cutting through clouds, curious expression, detailed clothing folds and textures"
    },
    {
        "name": "✨ Cozy Indoor Room",
        "category": "indoor",
        "template": "waist-up shot of {trigger_word} sitting inside a cozy wooden cabin, warm lantern light, gentle smile, casual comfortable posture, domestic peaceful atmosphere, rich interior details"
    },
    {
        "name": "⚡ Low-Angle Power Stance",
        "category": "low_angle",
        "template": "dramatic low-angle shot of {trigger_word}, looking down with a confident smirk, majestic sky and clouds behind, powerful presence, vibrant colors, clean details"
    }
]

LORA_SHOT_MATRIX_ANIME = [
    {
        "name": "📸 Studio Face Portrait",
        "category": "close_up",
        "template": "masterpiece, best quality, {trigger_word}, close-up portrait, looking at viewer, neutral expression, soft lighting, simple background, clean illustration, sharp detailed eyes"
    },
    {
        "name": "😊 Cheerful Smiling Bust",
        "category": "portrait",
        "template": "masterpiece, best quality, {trigger_word}, upper body, smiling, gentle smile, looking at viewer, soft outdoor lighting, natural colors, depth of field"
    },
    {
        "name": "⚔️ Dynamic Combat Action",
        "category": "action_full_body",
        "template": "masterpiece, best quality, {trigger_word}, full body, dynamic combat pose, holding adventurer weapon, intense expression, determined, wind blown hair, dramatic lighting"
    },
    {
        "name": "🌲 Palworld Lush Wilderness",
        "category": "environment",
        "template": "masterpiece, best quality, {trigger_word}, exploring, lush grass, rolling green hills, fantasy meadow, bright daylight, gentle smile, adventurer outfit, beautiful background"
    },
    {
        "name": "🌅 Golden Hour Side Profile",
        "category": "side_profile",
        "template": "masterpiece, best quality, {trigger_word}, profile, looking away, sunset, golden hour, rim lighting, lens flare, calm serene expression, detailed hair"
    },
    {
        "name": "⛺ Night Campfire Glow",
        "category": "lighting",
        "template": "masterpiece, best quality, {trigger_word}, sitting near campfire, night, warm firelight illumination, starry sky, cozy atmosphere, relaxed expression"
    },
    {
        "name": "👑 Heroic 3/4 Turn",
        "category": "three_quarter",
        "template": "masterpiece, best quality, {trigger_word}, three-quarter view, standing confident, hands on hips, fantasy ancient ruins, sunbeams, confident smirk"
    },
    {
        "name": "🌧️ Moody Rain & Fog",
        "category": "mood",
        "template": "masterpiece, best quality, {trigger_word}, light rain, mist, wet hair, serious gaze, cool blue lighting, atmospheric, high quality anime artwork"
    },
    {
        "name": "👀 Over-The-Shoulder View",
        "category": "angle",
        "template": "masterpiece, best quality, {trigger_word}, from behind, looking back, looking at viewer, dynamic perspective, intriguing expression, soft background"
    },
    {
        "name": "🏛️ Ancient Stone Ruins",
        "category": "environment",
        "template": "masterpiece, best quality, {trigger_word}, full body, standing beside mossy stone pillars, ancient fantasy architecture, sunlight, detailed clothing"
    },
    {
        "name": "✨ Cozy Indoor Room",
        "category": "indoor",
        "template": "masterpiece, best quality, {trigger_word}, sitting inside wooden cabin, warm lantern light, gentle smile, comfortable casual clothes, cozy indoor"
    },
    {
        "name": "⚡ Low-Angle Power Stance",
        "category": "low_angle",
        "template": "masterpiece, best quality, {trigger_word}, from below, low angle view, confident smirk, dramatic sky, clouds, powerful stance, vibrant"
    }
]

# Default alias
LORA_SHOT_MATRIX = LORA_SHOT_MATRIX_ANIME

ANIME_CAMERA_ANGLES = [
    "close-up face portrait",
    "upper body portrait",
    "waist-up medium shot",
    "three-quarter view",
    "full body standing",
    "from below low angle",
    "profile view looking away",
    "from behind looking back",
]

ANIME_EXPRESSIONS = [
    "neutral expression",
    "gentle smile",
    "confident smirk",
    "intense determined gaze",
    "serious battle-ready",
    "cheerful happy expression",
]

ANIME_ENVIRONMENTS = [
    "simple clean background",
    "lush green fantasy meadow, rolling hills, wild flowers",
    "ancient mossy stone ruins",
    "cozy wooden cabin interior, warm lantern light",
    "misty pine forest, glowing flora",
]


def get_preset_shot_prompts(trigger_word: str, checkpoint_name: str = None) -> dict[str, str]:
    """Returns a dictionary of named preset shot prompts populated with the trigger word, adapted for the model type."""
    tw = trigger_word.strip() or "character"
    is_anime = is_anime_checkpoint(checkpoint_name)
    matrix = LORA_SHOT_MATRIX_ANIME if is_anime else LORA_SHOT_MATRIX_REALISTIC
    
    presets = {}
    for item in matrix:
        presets[item["name"]] = item["template"].format(trigger_word=tw)
    return presets


def generate_suggested_prompts(session_id: str, trigger_word: str, count: int = 5, checkpoint_name: str = None) -> list[str]:
    """
    Generates intelligent, diverse prompt suggestions for the character LoRA dataset
    adapted for anime (Wai Illustrious) or realistic checkpoints.
    """
    tw = trigger_word.strip() or "character"
    is_anime = is_anime_checkpoint(checkpoint_name)
    suggestions = []
    
    existing_images = db.get_dataset_images(session_id)
    num_existing = len(existing_images)

    if num_existing == 0:
        if is_anime:
            priorities = [
                f"masterpiece, best quality, {tw}, close-up portrait, looking at viewer, neutral expression, soft lighting, simple background, sharp detailed eyes",
                f"masterpiece, best quality, {tw}, upper body, smiling, gentle smile, soft outdoor daylight, detailed face",
                f"masterpiece, best quality, {tw}, full body, 3/4 turn angle, adventurer outfit, lush fantasy meadow, vibrant",
                f"masterpiece, best quality, {tw}, profile, looking away, sunset, golden hour rim lighting, calm serene expression",
                f"masterpiece, best quality, {tw}, dynamic combat pose, holding weapon, intense expression, wind, dramatic lighting"
            ]
        else:
            priorities = [
                f"extreme close-up portrait of {tw}, neutral expression, facing camera, studio lighting, clean background, sharp focus, 8k uhd",
                f"upper body portrait of {tw} with a warm smile, soft natural outdoor daylight, highly detailed face and eyes",
                f"full-body standing shot of {tw}, adventurer gear, 3/4 turn angle, fantasy meadow background, clear full character design",
                f"side profile view of {tw} looking into distance, golden hour rim lighting, sharp silhouette",
                f"dynamic action full body shot of {tw}, combat ready pose, dramatic cinematic lighting, epic motion"
            ]
        return priorities[:count]

    # Generate diverse randomized combinations
    used_combos = set()
    attempts = 0
    while len(suggestions) < count and attempts < 50:
        attempts += 1
        if is_anime:
            angle = random.choice(ANIME_CAMERA_ANGLES)
            expr = random.choice(ANIME_EXPRESSIONS)
            env = random.choice(ANIME_ENVIRONMENTS)
            prompt = f"masterpiece, best quality, {tw}, {angle}, {expr}, {env}"
        else:
            angle = random.choice(CAMERA_ANGLES)
            expr = random.choice(EXPRESSIONS)
            light = random.choice(LIGHTING_STYLES)
            env = random.choice(ENVIRONMENTS)
            outfit = random.choice(OUTFITS)
            prompt = f"{angle} of {tw}, {expr}, wearing {outfit}, {env}, {light}, photorealistic, sharp focus, 8k uhd"

        if prompt not in used_combos:
            used_combos.add(prompt)
            suggestions.append(prompt)

    return suggestions


# =========================================================================
# Dataset Session & Image File Management
# =========================================================================

def get_session_dir(session_id: str) -> str:
    """Returns the absolute path to the session's image storage folder."""
    path = os.path.join(DATASET_ROOT_DIR, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def prepare_image_for_dataset(image_bytes: bytes, target_size: tuple[int, int] = (1024, 1024)) -> bytes:
    """
    Preprocesses an image for SDXL LoRA training:
    - Crops center to square aspect ratio (1:1).
    - Resizes cleanly with LANCZOS to 1024x1024.
    - Saves as optimized PNG.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    
    # Calculate square center crop
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(target_size, Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img_resized.save(buf, format="PNG", quality=95)
    return buf.getvalue()


def save_image_to_dataset(session_id: str, image_bytes: bytes, caption: str = "", filename_hint: str = None) -> tuple[int, str]:
    """
    Crops, resizes to 1024x1024, saves the PNG + TXT file, and inserts the record in SQLite.
    Returns (image_id, image_path).
    """
    session_dir = get_session_dir(session_id)
    
    # Determine unique index for image
    existing = db.get_dataset_images(session_id)
    next_idx = len(existing) + 1
    
    base_name = f"img_{next_idx:03d}"
    img_filename = f"{base_name}.png"
    txt_filename = f"{base_name}.txt"
    
    img_path = os.path.join(session_dir, img_filename)
    txt_path = os.path.join(session_dir, txt_filename)
    
    # Preprocess & save PNG
    processed_bytes = prepare_image_for_dataset(image_bytes)
    with open(img_path, "wb") as f:
        f.write(processed_bytes)
        
    # Save caption TXT
    clean_caption = caption.strip() if caption else ""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(clean_caption)
        
    # Record in DB
    img_id = db.add_image_to_dataset(session_id, img_path, clean_caption)
    logger.info(f"Saved dataset image #{img_id} to {img_path} (session={session_id})")
    return img_id, img_path


# =========================================================================
# Florence-2 Batch Auto-Captioning
# =========================================================================

async def auto_caption_dataset_image(comfy_client, image_path: str, trigger_word: str) -> str:
    """
    Uploads an image to ComfyUI and runs Florence-2 detailed captioning using DESCRIBE_cuibot.json.
    Injects the trigger word and returns the refined caption.
    """
    if not os.path.exists(image_path):
        return ""
        
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        
    filename = os.path.basename(image_path)
    upload_res = await comfy_client.upload_image(img_bytes, filename)
    uploaded_name = upload_res.get("name")
    if not uploaded_name:
        logger.error(f"Failed to upload {filename} to ComfyUI for captioning.")
        return ""
        
    # Load describe workflow
    workflow_path = "workflows/DESCRIBE_cuibot.json"
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
        
    workflow["1"]["inputs"]["image"] = uploaded_name
    
    # Execute workflow
    results = await comfy_client.generate(workflow, timeout=180)
    
    detailed_caption = ""
    if isinstance(results, dict):
        if "10" in results and "text" in results["10"]:
            text_list = results["10"]["text"]
            if text_list:
                detailed_caption = text_list[0].strip()
        elif "9" in results and "text" in results["9"]:
            text_list = results["9"]["text"]
            if text_list:
                detailed_caption = text_list[0].strip()
                
    if not detailed_caption:
        detailed_caption = "character in detailed scenery"
        
    # Format and inject character trigger word
    tw = trigger_word.strip()
    # Clean generic prefixes from Florence-2 like "The image shows..."
    cleaned = re.sub(r'^(The image depicts|The image shows|An image of|A photo of)\s+', '', detailed_caption, flags=re.IGNORECASE).strip()
    
    if tw:
        # Prepend trigger word formatted for LoRA training
        final_caption = f"{tw}, {cleaned}"
    else:
        final_caption = cleaned
        
    return final_caption


async def batch_caption_session(comfy_client, session_id: str, trigger_word: str, progress_callback=None) -> dict:
    """
    Runs Florence-2 on all images in the session, writing `.txt` files and updating SQLite.
    Returns stats dict: {"total": int, "processed": int, "failed": int}
    """
    images = db.get_dataset_images(session_id)
    total = len(images)
    processed = 0
    failed = 0
    
    for idx, img_entry in enumerate(images):
        img_id = img_entry["id"]
        img_path = img_entry["image_path"]
        
        try:
            caption = await auto_caption_dataset_image(comfy_client, img_path, trigger_word)
            if caption:
                # Update DB
                db.update_image_caption(img_id, caption)
                
                # Update TXT file
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error captioning image {img_path}: {e}")
            failed += 1
            
        if progress_callback:
            await progress_callback(idx + 1, total)
            
    return {"total": total, "processed": processed, "failed": failed}


# =========================================================================
# Dataset Packaging & Export (Kohya_ss / OneTrainer / Civitai Format)
# =========================================================================

def export_dataset_zip(session_id: str, repeats: int = 10) -> tuple[str, int]:
    """
    Creates a standardized Kohya_ss / OneTrainer folder structure inside a ZIP archive:
    Structure:
      {repeats}_{trigger_word}/
        img_001.png
        img_001.txt
        img_002.png
        img_002.txt
        ...
    Returns (zip_file_path, image_count).
    """
    session = db.get_dataset_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")
        
    trigger_word = session.get("trigger_word", "character").replace(" ", "_")
    folder_name = f"{repeats}_{trigger_word}"
    
    images = db.get_dataset_images(session_id)
    if not images:
        raise ValueError("Dataset has no images to export.")
        
    export_dir = os.path.join(DATASET_ROOT_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    clean_session_name = re.sub(r'[^a-zA-Z0-9_\-]', '', session.get("name", "lora_dataset"))
    zip_filename = f"{clean_session_name}_{session_id[:8]}_SDXL_dataset.zip"
    zip_path = os.path.join(export_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, img_entry in enumerate(images, start=1):
            img_path = img_entry["image_path"]
            caption = img_entry.get("caption", "").strip()
            
            if not os.path.exists(img_path):
                continue
                
            arc_img_name = f"{folder_name}/img_{idx:03d}.png"
            arc_txt_name = f"{folder_name}/img_{idx:03d}.txt"
            
            # Write image to zip
            zf.write(img_path, arcname=arc_img_name)
            
            # Write txt caption to zip
            if not caption and session.get("trigger_word"):
                caption = session.get("trigger_word")
            zf.writestr(arc_txt_name, caption)
            
    return zip_path, len(images)

def cleanup_old_dataset_exports(days_threshold: int = 14) -> int:
    """
    Deletes exported dataset zip packages older than days_threshold from datasets/exports.
    Returns the number of deleted zip files.
    """
    export_dir = os.path.join(DATASET_ROOT_DIR, "exports")
    if not os.path.exists(export_dir):
        return 0

    import time
    now = time.time()
    cutoff = now - (days_threshold * 86400)
    cleaned_count = 0

    try:
        for fname in os.listdir(export_dir):
            if fname.lower().endswith(".zip"):
                fpath = os.path.join(export_dir, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        os.remove(fpath)
                        cleaned_count += 1
                except Exception as file_err:
                    logger.debug(f"Could not remove old dataset zip {fpath}: {file_err}")
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old LoRA dataset export zip(s).")
    except Exception as e:
        logger.error(f"Error cleaning old dataset exports: {e}")

    return cleaned_count
