import io
import logging
import os
import asyncio
from datetime import datetime
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

import re

logger = logging.getLogger("DiscordBot.ImageUtils")

def get_checkpoint_abbrev(checkpoint: str) -> str:
    """
    Returns a short, clean abbreviation for a checkpoint model filename.
    Example: 'waiIllustriousSDXL_v170.safetensors' -> 'wai'
             'RealVisXL_V4.0.safetensors' -> 'realvis'
             'illustriousRealismBy_v10VAE.safetensors' -> 'illuReal'
             'juggernautXL_ragnarok.safetensors' -> 'juggernaut'
             'CopaxTimeLessXL.safetensors' -> 'copax'
             'ultraRealisticByStable_v25.safetensors' -> 'ultra'
             'hyphoriaIlluNAI_v001.safetensors' -> 'hyphoria'
             'novaFurryXL_ilV180A.safetensors' -> 'nova'
    """
    if not checkpoint:
        return "wai"
    base = os.path.splitext(os.path.basename(str(checkpoint)))[0]
    ckpt_lower = base.lower()

    if "waiillustrious" in ckpt_lower or "wai" in ckpt_lower:
        return "wai"
    if "illustriousrealism" in ckpt_lower:
        return "illuReal"
    if "realvis" in ckpt_lower:
        return "realvis"
    if "juggernaut" in ckpt_lower:
        return "juggernaut"
    if "copax" in ckpt_lower:
        return "copax"
    if "ultrarealistic" in ckpt_lower or "ultra" in ckpt_lower:
        return "ultra"
    if "hyphoria" in ckpt_lower:
        return "hyphoria"
    if "nova" in ckpt_lower:
        return "nova"
    if "flux" in ckpt_lower:
        return "flux"
    if "pony" in ckpt_lower:
        return "pony"
    if "animagine" in ckpt_lower:
        return "animagine"

    clean = re.sub(r'[^a-zA-Z0-9]', '', base.split('_')[0])
    return clean[:10] if clean else "sdxl"


def format_image_filename(prefix: str = "grid", seed: int = None, ext: str = "png", sref: str = None) -> str:
    """
    Generates a clean descriptive filename with YYYY-MM-DD date, seed, and sref code.
    Example: grid_2026-08-14_175319_seed12345678_sref905471.jpg
             isolated_1_2026-08-14_175319_seed12345678.png
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    seed_str = f"_seed{seed}" if seed is not None else ""
    sref_str = f"_sref{sref}" if sref else ""
    clean_ext = ext.lstrip(".")
    clean_prefix = prefix.replace("DiscordBot_", "").replace("DiscordBot", "").strip("_")
    return f"{clean_prefix}_{date_str}{seed_str}{sref_str}.{clean_ext}"


def get_dated_save_prefix(subfolder: str = "") -> str:
    """
    Constructs ComfyUI output save prefix inside Discord Bot/<MM>/<DD>/<subfolder>/.
    Example: 'Discord Bot/08/14/highres/'
    """
    now = datetime.now()
    mm = now.strftime("%m")
    dd = now.strftime("%d")
    base = f"Discord Bot/{mm}/{dd}"
    if subfolder:
        clean_sub = subfolder.strip("/")
        return f"{base}/{clean_sub}/"
    return f"{base}/"

def crop_to_aspect_ratio(image_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """Crops and resizes image_bytes from center to match target aspect ratio (target_w/target_h)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    
    target_ratio = target_w / target_h
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Current image is wider than target. Crop the sides.
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:
        # Current image is taller than target. Crop the top/bottom.
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
        
    # Resize to exact target dimensions
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    out_io = io.BytesIO()
    img.save(out_io, format="PNG")
    return out_io.getvalue()

def upscale_isolated_image(image_bytes: bytes, target_w: int = 1024, target_h: int = 1024) -> tuple[bytes, int, int]:
    """Upscales isolated quadrant image bytes to full HD resolution (2x upscale using Lanczos)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    curr_w, curr_h = img.size
    
    if curr_w < target_w or curr_h < target_h:
        new_w = max(curr_w * 2, target_w)
        new_h = max(curr_h * 2, target_h)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        new_w, new_h = curr_w, curr_h
        
    out_io = io.BytesIO()
    img.save(out_io, format="PNG")
    return out_io.getvalue(), new_w, new_h

from PIL import ImageEnhance

def boost_image_vibrancy_and_contrast(image_bytes: bytes, saturation: float = 1.22, contrast: float = 1.08) -> bytes:
    """Enhances color saturation and contrast of image_bytes to prevent SDXL/FLUX washed out output."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Boost saturation
        converter = ImageEnhance.Color(img)
        img = converter.enhance(saturation)
        
        # Boost contrast slightly
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(contrast)
        
        out_io = io.BytesIO()
        img.save(out_io, format="PNG")
        return out_io.getvalue()
    except Exception as e:
        logger.error(f"Error boosting image vibrancy: {e}")
        return image_bytes

def embed_metadata(image_bytes, prompt, neg_prompt, seed, width, height):
    """Embeds standard generation metadata into PNG info chunk."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        metadata = PngInfo()
        metadata.add_text("parameters", f"{prompt}\nNegative prompt: {neg_prompt}\nSteps: 28, Sampler: dpmpp_2m, Scheduler: karras, CFG scale: 4.0, Seed: {seed}, Size: {width}x{height}")
        out_io = io.BytesIO()
        img.save(out_io, format="PNG", pnginfo=metadata)
        out_io.seek(0)
        return out_io
    except Exception as e:
        logger.error(f"Failed to embed metadata: {e}")
        return io.BytesIO(image_bytes)

def create_grid(image_bytes_list, prompt, neg_prompt, seed, width, height):
    """Stitches 4 images into a 2x2 grid and saves as JPEG to keep file size small."""
    images = [Image.open(io.BytesIO(b)) for b in image_bytes_list]
    if len(images) != 4:
        raise ValueError("Grid generation requires exactly 4 images.")
    
    w, h = images[0].size
    grid = Image.new('RGB', (w * 2, h * 2))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (w, 0))
    grid.paste(images[2], (0, h))
    grid.paste(images[3], (w, h))
    
    out_io = io.BytesIO()
    grid.save(out_io, format="JPEG", quality=90, optimize=True)
    out_io.seek(0)
    return out_io

def calculate_outpaint_padding(image_bytes: bytes, mode_or_ratio: str):
    """
    Rescales image to SDXL base resolution (max side 1024) if needed, 
    and calculates (left, top, right, bottom, rescaled_image_bytes, target_w, target_h).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    # Upscale to SDXL native base resolution (max side 1024) if smaller
    max_side = max(orig_w, orig_h)
    if max_side < 1024:
        scale = 1024 / max_side
        w = int(round(orig_w * scale))
        h = int(round(orig_h * scale))
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    else:
        w, h = orig_w, orig_h

    if mode_or_ratio == "16:9":
        target_w = int(round((h * 16 / 9) / 64) * 64) if w / h < 16 / 9 else w
        target_h = h if w / h < 16 / 9 else int(round((w * 9 / 16) / 64) * 64)
    elif mode_or_ratio == "21:9":
        target_w = int(round((h * 21 / 9) / 64) * 64) if w / h < 21 / 9 else w
        target_h = h if w / h < 21 / 9 else int(round((w * 9 / 21) / 64) * 64)
    elif mode_or_ratio == "9:16":
        target_w = w if w / h > 9 / 16 else int(round((h * 9 / 16) / 64) * 64)
        target_h = int(round((w * 16 / 9) / 64) * 64) if w / h > 9 / 16 else h
    elif mode_or_ratio == "3:5":
        target_w = w if w / h > 3 / 5 else int(round((h * 3 / 5) / 64) * 64)
        target_h = int(round((w * 5 / 3) / 64) * 64) if w / h > 3 / 5 else h
    elif mode_or_ratio == "10:7":
        target_w = int(round((h * 10 / 7) / 64) * 64) if w / h < 10 / 7 else w
        target_h = h if w / h < 10 / 7 else int(round((w * 7 / 10) / 64) * 64)
    elif mode_or_ratio == "1.5x":
        target_w = int(round((w * 1.5) / 64) * 64)
        target_h = int(round((h * 1.5) / 64) * 64)
    elif mode_or_ratio == "2.0x":
        target_w = int(round((w * 2.0) / 64) * 64)
        target_h = int(round((h * 2.0) / 64) * 64)
    else:
        target_w = int(round((h * 16 / 9) / 64) * 64)
        target_h = h

    total_pad_w = max(0, target_w - w)
    total_pad_h = max(0, target_h - h)

    left = total_pad_w // 2
    right = total_pad_w - left
    top = total_pad_h // 2
    bottom = total_pad_h - top

    out_io = io.BytesIO()
    img.save(out_io, format="PNG")

    return left, top, right, bottom, out_io.getvalue(), target_w, target_h


QUADRANT_CACHE_DIR = os.getenv("QUADRANT_CACHE_DIR", r"C:\ComfyUI\ComfyUI\output\Discord Bot\scratch")

def save_quadrant_images(generation_id: str, images: list):
    """Saves the generated quadrant image bytes to disk for variation lookup."""
    try:
        os.makedirs(QUADRANT_CACHE_DIR, exist_ok=True)
        for idx, img_bytes in enumerate(images):
            path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_{idx + 1}.png")
            with open(path, "wb") as f:
                f.write(img_bytes)
    except Exception as e:
        logger.error(f"Failed to save quadrant images: {e}")

async def save_quadrant_images_async(generation_id: str, images: list):
    """Non-blocking async variant of save_quadrant_images."""
    await asyncio.to_thread(save_quadrant_images, generation_id, images)

def get_quadrant_bytes(generation_id: str, index: int):
    """Retrieves PNG bytes for quadrant index (1-4)."""
    path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_{index}.png")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read quadrant image {path}: {e}")
    return None

async def get_quadrant_bytes_async(generation_id: str, index: int):
    """Non-blocking async variant of get_quadrant_bytes."""
    return await asyncio.to_thread(get_quadrant_bytes, generation_id, index)

async def embed_metadata_async(image_bytes: bytes, prompt: str, neg_prompt: str = "", seed: int = 0, width: int = 1024, height: int = 1024):
    """Non-blocking async variant of embed_metadata."""
    return await asyncio.to_thread(embed_metadata, image_bytes, prompt, neg_prompt, seed, width, height)

async def create_grid_async(image_bytes_list: list, prompt: str, neg_prompt: str = "", seed: int = 0, width: int = 1024, height: int = 1024):
    """Non-blocking async variant of create_grid."""
    return await asyncio.to_thread(create_grid, image_bytes_list, prompt, neg_prompt, seed, width, height)

def crop_quadrant_from_grid_bytes(grid_bytes: bytes, index: int) -> bytes:
    """Crops a 2x2 grid image (bytes) into quadrant index (1-4) PNG bytes."""
    img = Image.open(io.BytesIO(grid_bytes)).convert("RGB")
    w, h = img.size
    half_w, half_h = w // 2, h // 2
    if index == 1:
        box = (0, 0, half_w, half_h)
    elif index == 2:
        box = (half_w, 0, w, half_h)
    elif index == 3:
        box = (0, half_h, half_w, h)
    elif index == 4:
        box = (half_w, half_h, w, h)
    else:
        box = (0, 0, w, h)
    
    cropped = img.crop(box)
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()

ICO_OUTPUT_DIR = r"C:\ComfyUI\ComfyUI\output\Discord Bot\ico"
WINDOWS_ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

def apply_rounded_corners(img: Image.Image, radius_ratio: float = 0.18) -> Image.Image:
    """Applies smooth anti-aliased curved edges (rounded squircle corners) to an image."""
    img = img.convert("RGBA")
    w, h = img.size
    radius = int(min(w, h) * radius_ratio)
    
    # 4x supersampled mask for ultra-smooth anti-aliased curved edges
    scale = 4
    mask_w, mask_h = w * scale, h * scale
    mask_radius = radius * scale
    
    mask = Image.new("L", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, mask_w - 1, mask_h - 1), radius=mask_radius, fill=255)
    
    # Resize mask down to original size using LANCZOS for super smooth curved edges
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)
    
    # Merge with any existing alpha channel in the image
    current_alpha = img.split()[3]
    final_alpha = Image.composite(current_alpha, Image.new("L", (w, h), 0), mask)
    
    img.putalpha(final_alpha)
    return img

def apply_rounded_corners_to_bytes(img_bytes: bytes, radius_ratio: float = 0.18) -> bytes:
    """Helper to apply rounded corners to image bytes and return PNG bytes."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        rounded_img = apply_rounded_corners(img, radius_ratio=radius_ratio)
        out_io = io.BytesIO()
        rounded_img.save(out_io, format="PNG")
        return out_io.getvalue()
    except Exception as e:
        logger.error(f"Error applying rounded corners to bytes: {e}")
        return img_bytes

def create_windows_ico_bytes(img_bytes: bytes, rounded_corners: bool = True, radius_ratio: float = 0.18) -> bytes:
    """
    Converts input image bytes into a fully compliant Windows 11 multi-resolution .ico file.
    Includes RGBA layers: 16x16, 24x24, 32x32, 48x48, 64x64, 128x128, 256x256.
    If rounded_corners is True, applies anti-aliased curved edges for clean Windows 11 icons.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        if rounded_corners:
            img = apply_rounded_corners(img, radius_ratio=radius_ratio)
        out_io = io.BytesIO()
        img.save(out_io, format="ICO", sizes=WINDOWS_ICO_SIZES)
        return out_io.getvalue()
    except Exception as e:
        logger.error(f"Error generating Windows ICO bytes: {e}")
        return None

def save_ico_file(ico_bytes: bytes, filename: str) -> str:
    """Saves ICO bytes to ComfyUI output inside Discord Bot/<MM>/<DD>/ico/."""
    try:
        now = datetime.now()
        mm = now.strftime("%m")
        dd = now.strftime("%d")
        comfy_output_path = os.getenv("COMFYUI_OUTPUT_PATH", "C:/ComfyUI/ComfyUI/output")
        target_dir = os.path.join(comfy_output_path, "Discord Bot", mm, dd, "ico")
        os.makedirs(target_dir, exist_ok=True)
        full_path = os.path.join(target_dir, filename)
        with open(full_path, "wb") as f:
            f.write(ico_bytes)
        logger.info(f"Saved Windows ICO file to {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Failed to save ICO file: {e}")
        return None

def convert_image_to_ico(image_bytes: bytes, rounded_corners: bool = True, radius_ratio: float = 0.18) -> tuple:
    """
    Converts any arbitrary input image bytes into a 1:1 square Windows 11 multi-resolution .ico container
    and a preview PNG bytes tuple: (png_bytes, ico_bytes).
    Center-crops to 1:1 square if input image is non-square.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        w, h = img.size
        
        # Center crop to 1:1 square if non-square
        if w != h:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))

        # Ensure high quality 1024x1024 base square resolution
        if img.size != (1024, 1024):
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

        # Apply smooth anti-aliased curved edge mask if requested
        if rounded_corners:
            img = apply_rounded_corners(img, radius_ratio=radius_ratio)

        # Save PNG preview
        png_io = io.BytesIO()
        img.save(png_io, format="PNG")
        png_bytes = png_io.getvalue()

        # Save ICO container
        ico_io = io.BytesIO()
        img.save(ico_io, format="ICO", sizes=WINDOWS_ICO_SIZES)
        ico_bytes = ico_io.getvalue()

        return png_bytes, ico_bytes
    except Exception as e:
        logger.error(f"Error converting image to ICO: {e}")
        return None, None
