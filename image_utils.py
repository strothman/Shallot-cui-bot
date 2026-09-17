import io
import logging
import os
import asyncio
from datetime import datetime
import math
from PIL import Image, ImageDraw, ImageChops, ImageFilter
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

    # Normalize input image to SDXL/Krea native base resolution (max side 1024)
    # Upscale if smaller than 1024, or clamp if oversized (> 1280px / > 1.5M pixels) to prevent canvas explosion
    max_side = max(orig_w, orig_h)
    if max_side < 1024:
        scale = 1024 / max_side
        w = int(round(orig_w * scale))
        h = int(round(orig_h * scale))
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    elif max_side > 1280 or (orig_w * orig_h > 1_500_000):
        scale = 1024 / max_side
        w = int(round((orig_w * scale) / 64) * 64)
        h = int(round((orig_h * scale) / 64) * 64)
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    else:
        w, h = orig_w, orig_h

    mode = str(mode_or_ratio).lower().strip()

    # Directional pan expansion (384px extension aligned to 64px boundary)
    if mode in ["up", "pan_up"]:
        pad_step = 384
        top = pad_step
        bottom = 0
        left = 0
        right = 0
        target_w = w
        target_h = h + pad_step
    elif mode in ["down", "pan_down"]:
        pad_step = 384
        top = 0
        bottom = pad_step
        left = 0
        right = 0
        target_w = w
        target_h = h + pad_step
    elif mode in ["left", "pan_left"]:
        pad_step = 384
        top = 0
        bottom = 0
        left = pad_step
        right = 0
        target_w = w + pad_step
        target_h = h
    elif mode in ["right", "pan_right"]:
        pad_step = 384
        top = 0
        bottom = 0
        left = 0
        right = pad_step
        target_w = w + pad_step
        target_h = h
    elif mode == "16:9":
        target_w = int(round((h * 16 / 9) / 64) * 64) if w / h < 16 / 9 else w
        target_h = h if w / h < 16 / 9 else int(round((w * 9 / 16) / 64) * 64)
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "21:9":
        target_w = int(round((h * 21 / 9) / 64) * 64) if w / h < 21 / 9 else w
        target_h = h if w / h < 21 / 9 else int(round((w * 9 / 21) / 64) * 64)
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "9:16":
        target_w = w if w / h > 9 / 16 else int(round((h * 9 / 16) / 64) * 64)
        target_h = int(round((w * 16 / 9) / 64) * 64) if w / h > 9 / 16 else h
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "3:5":
        target_w = w if w / h > 3 / 5 else int(round((h * 3 / 5) / 64) * 64)
        target_h = int(round((w * 5 / 3) / 64) * 64) if w / h > 3 / 5 else h
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "10:7":
        target_w = int(round((h * 10 / 7) / 64) * 64) if w / h < 10 / 7 else w
        target_h = h if w / h < 10 / 7 else int(round((w * 7 / 10) / 64) * 64)
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "1.5x":
        target_w = int(round((w * 1.5) / 64) * 64)
        target_h = int(round((h * 1.5) / 64) * 64)
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
    elif mode == "2.0x":
        target_w = int(round((w * 2.0) / 64) * 64)
        target_h = int(round((h * 2.0) / 64) * 64)
        total_pad_w = max(0, target_w - w)
        total_pad_h = max(0, target_h - h)
        left = total_pad_w // 2
        right = total_pad_w - left
        top = total_pad_h // 2
        bottom = total_pad_h - top
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
    """Retrieves PNG bytes for quadrant index (1-4) or single-image generation."""
    path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_{index}.png")
    if not os.path.exists(path):
        path_single = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}.png")
        if os.path.exists(path_single):
            path = path_single
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

async def crop_to_aspect_ratio_async(image_bytes: bytes, target_w: int, target_h: int) -> bytes:
    """Non-blocking async variant of crop_to_aspect_ratio."""
    return await asyncio.to_thread(crop_to_aspect_ratio, image_bytes, target_w, target_h)

async def upscale_isolated_image_async(image_bytes: bytes, target_w: int = 1024, target_h: int = 1024) -> tuple[bytes, int, int]:
    """Non-blocking async variant of upscale_isolated_image."""
    return await asyncio.to_thread(upscale_isolated_image, image_bytes, target_w, target_h)

async def calculate_outpaint_padding_async(image_bytes: bytes, mode_or_ratio: str):
    """Non-blocking async variant of calculate_outpaint_padding."""
    return await asyncio.to_thread(calculate_outpaint_padding, image_bytes, mode_or_ratio)

def composite_outpaint_seamless(
    original_img_bytes: bytes,
    generated_img_bytes: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    feather_radius: int = 36
) -> bytes:
    """
    Seamlessly blends the pristine original image content onto the outpainted canvas
    using an S-curve cosine alpha feather along padded boundaries. Eliminates harsh
    rectangular VAE decoding box seams and exposure/lighting cutoffs.
    """
    try:
        gen_img = Image.open(io.BytesIO(generated_img_bytes)).convert("RGBA")
        gen_w, gen_h = gen_img.size

        orig_img = Image.open(io.BytesIO(original_img_bytes)).convert("RGBA")
        orig_w, orig_h = orig_img.size

        inner_w = gen_w - left - right
        inner_h = gen_h - top - bottom

        if (orig_w != inner_w or orig_h != inner_h) and inner_w > 0 and inner_h > 0:
            orig_img = orig_img.resize((inner_w, inner_h), Image.Resampling.LANCZOS)
            orig_w, orig_h = inner_w, inner_h

        # Construct smooth cosine-feathered alpha mask using pure Pillow (zero numpy dependency)
        mask_h = Image.new("L", (orig_w, orig_h), 255)
        mask_v = Image.new("L", (orig_w, orig_h), 255)

        if (left > 0 or right > 0) and feather_radius > 0:
            r = min(feather_radius, orig_w // 2)
            if r > 0:
                ramp_bytes = bytes([int(255.0 * 0.5 * (1.0 - math.cos(math.pi * i / r))) for i in range(r)])
                ramp_left = Image.frombytes("L", (r, 1), ramp_bytes).resize((r, orig_h), Image.Resampling.BILINEAR)
                if left > 0:
                    mask_h.paste(ramp_left, (0, 0))
                if right > 0:
                    ramp_right = ramp_left.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    mask_h.paste(ramp_right, (orig_w - r, 0))

        if (top > 0 or bottom > 0) and feather_radius > 0:
            r = min(feather_radius, orig_h // 2)
            if r > 0:
                ramp_bytes = bytes([int(255.0 * 0.5 * (1.0 - math.cos(math.pi * i / r))) for i in range(r)])
                ramp_top = Image.frombytes("L", (1, r), ramp_bytes).resize((orig_w, r), Image.Resampling.BILINEAR)
                if top > 0:
                    mask_v.paste(ramp_top, (0, 0))
                if bottom > 0:
                    ramp_bottom = ramp_top.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    mask_v.paste(ramp_bottom, (0, orig_h - r))

        alpha = ImageChops.darker(mask_h, mask_v)
        orig_img.putalpha(alpha)

        gen_img.paste(orig_img, (left, top), orig_img)
        final_img = gen_img.convert("RGB")

        out_buf = io.BytesIO()
        final_img.save(out_buf, format="PNG")
        return out_buf.getvalue()
    except Exception as e:
        logger.error(f"Error in composite_outpaint_seamless: {e}")
        return generated_img_bytes

async def composite_outpaint_seamless_async(
    original_img_bytes: bytes,
    generated_img_bytes: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    feather_radius: int = 36
) -> bytes:
    """Non-blocking async variant of composite_outpaint_seamless."""
    return await asyncio.to_thread(
        composite_outpaint_seamless,
        original_img_bytes,
        generated_img_bytes,
        left,
        top,
        right,
        bottom,
        feather_radius
    )

def create_outpaint_edge_bleed_canvas(
    image_bytes: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    blur_radius: int = 16,
    feather_radius: int = 36
) -> bytes:
    """
    Creates an enlarged pre-filled RGBA canvas where margin pixels are directionally
    replicated and blurred from the original image boundaries, preserving the scene's
    ambient lighting, contrast, and color palette. The Alpha channel acts as the
    exact inpaint/outpaint mask (255 opaque in the center, 0 transparent in margins).
    """
    try:
        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = orig.size
        new_w = orig_w + left + right
        new_h = orig_h + top + bottom

        # Create enlarged canvas with replicated edge pixels
        canvas = Image.new("RGB", (new_w, new_h))
        canvas.paste(orig, (left, top))

        # Replicate horizontal strips
        if left > 0:
            left_strip = orig.crop((0, 0, 1, orig_h)).resize((left, orig_h))
            canvas.paste(left_strip, (0, top))
        if right > 0:
            right_strip = orig.crop((orig_w - 1, 0, orig_w, orig_h)).resize((right, orig_h))
            canvas.paste(right_strip, (left + orig_w, top))

        # Replicate vertical strips (including corners)
        if top > 0:
            top_strip = canvas.crop((0, top, new_w, top + 1)).resize((new_w, top))
            canvas.paste(top_strip, (0, 0))
        if bottom > 0:
            bot_strip = canvas.crop((0, top + orig_h - 1, new_w, top + orig_h)).resize((new_w, bottom))
            canvas.paste(bot_strip, (0, top + orig_h))

        # Softly blur the canvas to blend directional streaks into smooth ambient color
        blurred = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Keep center pristine
        blurred.paste(orig, (left, top))

        # Build Alpha channel: 255 in center (with smooth cosine ramp near edges), 0 in margins
        # ComfyUI's LoadImage interprets RGBA alpha as: Mask = 1.0 - (Alpha / 255.0)
        # So Alpha = 255 -> Mask = 0.0 (Preserve), Alpha = 0 -> Mask = 1.0 (Outpaint)
        alpha_canvas = Image.new("L", (new_w, new_h), 0)

        # Create feathered alpha for original image box
        mask_h = Image.new("L", (orig_w, orig_h), 255)
        mask_v = Image.new("L", (orig_w, orig_h), 255)

        if (left > 0 or right > 0) and feather_radius > 0:
            r = min(feather_radius, orig_w // 2)
            if r > 0:
                ramp_bytes = bytes([int(255.0 * 0.5 * (1.0 - math.cos(math.pi * i / r))) for i in range(r)])
                ramp_left = Image.frombytes("L", (r, 1), ramp_bytes).resize((r, orig_h), Image.Resampling.BILINEAR)
                if left > 0:
                    mask_h.paste(ramp_left, (0, 0))
                if right > 0:
                    ramp_right = ramp_left.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    mask_h.paste(ramp_right, (orig_w - r, 0))

        if (top > 0 or bottom > 0) and feather_radius > 0:
            r = min(feather_radius, orig_h // 2)
            if r > 0:
                ramp_bytes = bytes([int(255.0 * 0.5 * (1.0 - math.cos(math.pi * i / r))) for i in range(r)])
                ramp_top = Image.frombytes("L", (1, r), ramp_bytes).resize((orig_w, r), Image.Resampling.BILINEAR)
                if top > 0:
                    mask_v.paste(ramp_top, (0, 0))
                if bottom > 0:
                    ramp_bottom = ramp_top.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    mask_v.paste(ramp_bottom, (0, orig_h - r))

        inner_alpha = ImageChops.darker(mask_h, mask_v)
        alpha_canvas.paste(inner_alpha, (left, top))

        rgba_img = blurred.convert("RGBA")
        rgba_img.putalpha(alpha_canvas)

        out_buf = io.BytesIO()
        rgba_img.save(out_buf, format="PNG")
        return out_buf.getvalue()
    except Exception as e:
        logger.error(f"Error in create_outpaint_edge_bleed_canvas: {e}")
        return image_bytes

async def create_outpaint_edge_bleed_canvas_async(
    image_bytes: bytes,
    left: int,
    top: int,
    right: int,
    bottom: int,
    blur_radius: int = 16,
    feather_radius: int = 36
) -> bytes:
    """Non-blocking async variant of create_outpaint_edge_bleed_canvas."""
    return await asyncio.to_thread(
        create_outpaint_edge_bleed_canvas,
        image_bytes,
        left,
        top,
        right,
        bottom,
        blur_radius,
        feather_radius
    )


async def boost_image_vibrancy_and_contrast_async(image_bytes: bytes, saturation: float = 1.22, contrast: float = 1.08) -> bytes:
    """Non-blocking async variant of boost_image_vibrancy_and_contrast."""
    return await asyncio.to_thread(boost_image_vibrancy_and_contrast, image_bytes, saturation, contrast)

async def crop_quadrant_from_grid_bytes_async(grid_bytes: bytes, index: int) -> bytes:
    """Non-blocking async variant of crop_quadrant_from_grid_bytes."""
    return await asyncio.to_thread(crop_quadrant_from_grid_bytes, grid_bytes, index)

async def create_thumbnail_bytes_async(image_bytes: bytes, max_dim: int = 512) -> bytes:
    """Non-blocking async variant of create_thumbnail_bytes."""
    return await asyncio.to_thread(create_thumbnail_bytes, image_bytes, max_dim)

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

async def convert_image_to_ico_async(image_bytes: bytes, rounded_corners: bool = True, radius_ratio: float = 0.18) -> tuple:
    """Non-blocking async variant of convert_image_to_ico."""
    return await asyncio.to_thread(convert_image_to_ico, image_bytes, rounded_corners, radius_ratio)


def detect_closest_aspect_ratio(width: int, height: int) -> str:
    """
    Finds the closest supported aspect ratio for an uploaded image based on its width and height.
    Supported: '21:9', '16:9', '10:7', '1:1', '3:5', '9:16'.
    """
    if not width or not height or width <= 0 or height <= 0:
        return "16:9"
    ratio = float(width) / float(height)
    candidates = {
        "21:9": 21.0 / 9.0,
        "16:9": 16.0 / 9.0,
        "10:7": 10.0 / 7.0,
        "1:1": 1.0,
        "3:5": 3.0 / 5.0,
        "9:16": 9.0 / 16.0,
    }
    return min(candidates.keys(), key=lambda ar: abs(candidates[ar] - ratio))


def detect_closest_krea_aspect_ratio(width: int, height: int) -> str:
    """
    Finds the closest supported aspect ratio for Bert's Krea 2 (Bertflow) workflow based on image dimensions.
    Supported Krea 2 options: '21:9', '16:9', '1:1', '3:4', '9:16'.
    Orientation-aware matching ensures landscape images map to landscape ratios and portrait to portrait.
    """
    if not width or not height or width <= 0 or height <= 0:
        return "16:9"
    ratio = float(width) / float(height)

    # Orientation-aware mapping for Krea 2 ratios:
    # 21:9 = ~2.333, 16:9 = ~1.778, 1:1 = 1.0, 3:4 = 0.75, 9:16 = 0.5625
    if ratio >= 2.0:
        return "21:9"
    elif ratio >= 1.2:
        return "16:9"
    elif ratio <= 0.625:
        return "9:16"
    elif ratio <= 0.85:
        return "3:4"
    else:
        return "1:1"


def create_thumbnail_bytes(image_bytes: bytes, max_dim: int = 512) -> bytes:
    """Creates a fast, lightweight JPEG thumbnail of an uploaded image for Discord embeds."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode in ("RGBA", "LA", "P"):
                # Paste onto white background to avoid black borders on transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            else:
                img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
    except Exception as e:
        logger.debug(f"Could not resize thumbnail: {e}")
        return image_bytes

