"""
Batch Sensual & Passionate Krea 2 Turbo Remaster Runner for Star Board Pictures.

Processes images in C:\\Users\\strot\\Pictures\\star board\\picture:
1. Discovers source images (.png, .jpg, .jpeg, .webp).
2. Calculates exact 1:1 proportional render dimensions (zero distortion / no smooshing).
3. Performs Smart Dual-Pass Human & Scene Detection:
   - Filename keyword scan (woman, girl, lady, model, portrait, etc.)
   - Non-blocking visual CV scan (OpenCV Haar cascades for frontal/profile face & body)
   - When human detected: Applies character LoRA if requested (e.g. Ogarla or Loveless) or natural sensual aesthetic.
4. Synthesizes high-fidelity sensual, intimate, passionate remaster via Krea 2 Turbo Diffusion:
   - Intensely passionate, seductive, erotic, sensual mood, romantic warm lighting, flushed dewy skin.
   - Glistening skin highlights via Krea 2 wetness LoRA (configurable wetness_strength).
   - Preserves pose, anatomy, and composition at configurable denoise (default: 0.52).
5. Super-resolves output using 4x-UltraSharp (1080p target, zero aspect-ratio drift).
6. Saves directly to C:\\Users\\strot\\Pictures\\star board\\krea2_sensual with persistent JSON resume tracking.
"""

import os
import sys
import io
import re
import math
import json
import time
import glob
import random
import argparse
import asyncio
import logging
import subprocess
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS, COMFYUI_BATCH_PATH
from comfy_client import ComfyClient
from parsers.workflows import prepare_bertflow_workflow
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchSensualStarBoardKrea2")

DEFAULT_SOURCE_DIR = r"C:\Users\strot\Pictures\star board\picture"
DEFAULT_OUTPUT_DIR = r"C:\Users\strot\Pictures\star board\krea2_sensual"
PROGRESS_FILE_NAME = "krea2_sensual_progress.json"
UPSCALE_MODEL = "4x-UltraSharp.pth"
KREA2_UNET = "museByStableYogi_v35Int8Extended.safetensors"

DEFAULT_SENSUAL_PREFIX = (
    "masterpiece, highly passionate, intensely sensual, seductive intimate atmosphere, "
    "deep romantic chemistry, sultry gaze, soft warm bedroom lighting, erotic aesthetic, "
    "flushed dewy glistening skin texture, passionate embrace, breathtaking beauty, "
    "cinematic lighting, crisp textures, exquisite fidelity, ultra-detailed"
)

STYLE_VARIATIONS = [
    {
        "name": "Midnight Velvet & Candlelight",
        "flavor": "midnight velvet ambiance, deep indigo and warm amber candlelight, soft shadows, sultry gaze, romantic intimate glow, seductive warmth, cinematic chiaroscuro",
        "wetness": 0.55,
    },
    {
        "name": "Golden Hour Radiance",
        "flavor": "golden hour sunset rim-light, sun-drenched amber glow, soft lens diffusion, warm cinematic backlight, captivating radiance, rich skin tones, breathtaking beauty",
        "wetness": 0.40,
    },
    {
        "name": "Neon Noir Rimlight",
        "flavor": "neon noir cinematic lighting, moody cyan and magenta rim light, rain-kissed reflections, sultry dramatic atmosphere, deep contrast, artistic allure",
        "wetness": 0.60,
    },
    {
        "name": "High-Fashion Editorial",
        "flavor": "high-fashion editorial photography, pristine studio lighting, silky drape textures, dramatic sharp focus, elegant sensual poses, vogue magazine aesthetic",
        "wetness": 0.35,
    },
    {
        "name": "Ethereal Dreamscape",
        "flavor": "ethereal dreamscape, soft luminous bloom, delicate twilight hues, enchanting hazy atmosphere, romantic fantasy aesthetic, fine art portraiture",
        "wetness": 0.45,
    },
    {
        "name": "35mm Vintage Analog",
        "flavor": "35mm vintage film photograph, warm authentic grain, nostalgic Kodachrome color grading, intimate flash aesthetic, candid allure, timeless cinematic feel",
        "wetness": 0.30,
    },
    {
        "name": "Silk & Satin Boudoir",
        "flavor": "luxurious silk and satin boudoir, warm diffused lighting, opulent fabric textures, seductive softness, lavish aesthetic, exquisite lighting nuances",
        "wetness": 0.50,
    },
    {
        "name": "Moonlit Twilight",
        "flavor": "cool moonlit twilight glow, silver rim lighting, deep midnight shadows, mystical quiet intimacy, soft atmospheric mist, enchanting elegance",
        "wetness": 0.45,
    },
]

HUMAN_KEYWORDS = [
    r"\bwoman\b", r"\bwomen\b", r"\bman\b", r"\bmen\b", r"\bgirl\b", r"\bgirls\b",
    r"\bboy\b", r"\bboys\b", r"\bperson\b", r"\bpeople\b", r"\bhuman\b", r"\bhumans\b",
    r"\bfemale\b", r"\bmale\b", r"\blady\b", r"\bladies\b", r"\bguy\b", r"\bguys\b",
    r"\bportrait\b", r"\bface\b", r"\bmodel\b", r"\bcharacters?\b", r"\bfigure\b",
    r"\bdancer\b", r"\bbride\b", r"\bqueen\b", r"\bking\b", r"\bprincess\b", r"\bprince\b"
]
HUMAN_REGEX = re.compile("|".join(HUMAN_KEYWORDS), re.IGNORECASE)

# Fast OpenCV cascades for visual human/face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
UPPER_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


def detect_human_visually(image_path: str) -> bool:
    """Fast non-blocking CV check for face or upper body in < 15ms."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        max_dim = 640
        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Frontal faces
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            return True

        # 2. Profile faces
        profiles = PROFILE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(profiles) > 0:
            return True

        # 3. Upper body
        uppers = UPPER_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
        if len(uppers) > 0:
            return True

        return False
    except Exception as e:
        logger.debug(f"Visual human detection error on {image_path}: {e}")
        return False


def load_json(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")
    return {}


def save_json(filepath: str, data: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    temp = f"{filepath}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp, filepath)


def extract_keywords_from_filename(filename: str) -> Tuple[str, bool]:
    """Cleans filename into scene keywords and checks for human regex indicators."""
    base = os.path.splitext(filename)[0]
    cleaned = re.sub(
        r'^\d+_stars_|\b(?:t2i|txt2img|img2img|img2watercoal)\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        ' ',
        base,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r'-(?:waiIllustrious|hyphoria|illustrij|Copax|cyberrealistic|illustrious)[^-\s]*',
        ' ',
        cleaned,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r'-(?:dpmpp|euler|ddim)[^-\s]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    has_human_kw = bool(HUMAN_REGEX.search(cleaned)) or "donana" in base.lower() or "sophil" in base.lower()
    return cleaned, has_human_kw


def calculate_zero_distortion_dimensions(orig_w: int, orig_h: int, base_area: int = 1498176) -> Tuple[int, int, int, int]:
    """
    Calculates Krea 2 render dimensions and proportional upscale dimensions
    that maintain exact 1:1 aspect ratio with the source image, eliminating distortion.
    """
    aspect = (orig_w / orig_h) if orig_h > 0 else 1.0

    # 1. Krea 2 ~1.5M pixel render dimensions (snapped to multiples of 8)
    target_render_h = math.sqrt(base_area / aspect)
    target_render_w = aspect * target_render_h
    render_w = max(512, min(2048, int(round(target_render_w / 8) * 8)))
    render_h = max(512, min(2048, int(round(target_render_h / 8) * 8)))

    # 2. High-res upscale target (strictly proportional to render dimensions)
    r_aspect = render_w / render_h
    if r_aspect >= 1.0:
        target_h = 1080
        target_w = int(round((1080 * r_aspect) / 8) * 8)
    else:
        target_w = 1080
        target_h = int(round((1080 / r_aspect) / 8) * 8)

    return render_w, render_h, target_w, target_h


async def extract_image_bytes(client: ComfyClient, outputs: Any) -> Optional[bytes]:
    """Extracts raw image bytes from ComfyUI generation outputs."""
    if not outputs:
        return None
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
        return outputs[0]
    if isinstance(outputs, dict):
        for node_id, node_out in outputs.items():
            if isinstance(node_out, dict) and "images" in node_out:
                for img_info in node_out["images"]:
                    fname = img_info.get("filename")
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")
                    if fname:
                        return await client.get_image(fname, subfolder, img_type)
    return None


async def save_image_async(image_bytes: bytes, out_path: str, format_name: str = "JPEG", quality: int = 95):
    """Offloads PIL image saving and disk I/O to a worker thread."""
    def _save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_path, format=format_name, quality=quality, optimize=True)
    await asyncio.to_thread(_save)


async def check_comfyui_online(client: ComfyClient) -> bool:
    """Checks if ComfyUI instance is actively responding."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{COMFYUI_ADDRESS}/system_stats", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Batch Sensual & Passionate Krea 2 Runner for Star Board")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Source image folder")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output destination folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--files", nargs="*", default=None, help="Specific image filenames to process")
    parser.add_argument("--denoise", type=float, default=0.42, help="Denoising strength (default: 0.42 to preserve core composition)")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps for Krea 2 Turbo (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed (default: randomized)")
    parser.add_argument("--sensual-prompt", default=DEFAULT_SENSUAL_PREFIX, help="Sensual & passionate prompt prefix")
    parser.add_argument("--style-mode", default="cycle", choices=["cycle", "random", "none"],
                        help="Atmospheric style variation mode: 'cycle' (rotates profiles), 'random', or 'none'")
    parser.add_argument("--wetness", type=float, default=0.50, help="Base skin wetness/dewy sheen strength (default: 0.50)")
    parser.add_argument("--character", default="smart", choices=["smart", "ogarla", "loveless", "none"],
                        help="Character mode: 'smart' auto-applies Ogarla when humans detected, 'none' uses pure natural remaster, or force 'ogarla'/'loveless'")
    parser.add_argument("--char-weight", type=float, default=0.80, help="Character LoRA weight (default: 0.80)")
    parser.add_argument("--no-upscale", action="store_true", help="Skip 4x-UltraSharp 1080p upscale pass")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running generation")
    args = parser.parse_args()

    source_dir = args.source_dir
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    raw_files = [
        f for f in glob.glob(os.path.join(source_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
    ]
    if len(raw_files) == 0 and os.path.isdir(os.path.join(source_dir, "picture")):
        source_dir = os.path.join(source_dir, "picture")
        raw_files = [
            f for f in glob.glob(os.path.join(source_dir, "*"))
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
        ]

    output_dir = args.output_dir
    progress_file = os.path.join(output_dir, PROGRESS_FILE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(raw_files)
    progress = load_json(progress_file)

    if args.files:
        chosen_set = set(os.path.basename(f) for f in args.files)
        target_files = [f for f in all_files if os.path.basename(f) in chosen_set]
    else:
        target_files = all_files

    to_process = []
    for fp in target_files:
        fname = os.path.basename(fp)
        out_jpg = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_sensual_krea2.jpg")
        cur_meta = progress.get(fname, {})
        if fname not in progress or not os.path.exists(out_jpg) or cur_meta.get("denoise") != args.denoise:
            to_process.append(fp)

    if args.limit:
        to_process = to_process[:args.limit]

    logger.info(f"\n{'='*75}\nSTARTING BATCH SENSUAL & PASSIONATE KREA 2 REMASTER\n{'='*75}")
    logger.info(f"Source Folder         : {source_dir}")
    logger.info(f"Output Folder         : {output_dir}")
    logger.info(f"Total Source Images   : {len(target_files)}")
    logger.info(f"Remaining to Process  : {len(to_process)}")
    logger.info(f"Krea 2 UNET           : {KREA2_UNET}")
    logger.info(f"Denoise Level         : {args.denoise:.2f}")
    logger.info(f"Steps                 : {args.steps}")
    logger.info(f"Skin Sheen / Wetness  : {args.wetness:.2f}")
    logger.info(f"Character Mode        : {args.character} (Weight: {args.char_weight:.2f})")
    logger.info(f"Style Mode            : {args.style_mode}")
    logger.info(f"Upscale Enabled       : {not args.no_upscale} ({UPSCALE_MODEL})")
    logger.info(f"Sensual Prefix        : {args.sensual_prompt[:75]}...")
    logger.info(f"{'='*75}\n")

    if args.dry_run:
        logger.info("[Dry Run] Evaluating plan for target images (first 15 shown):")
        for idx, fp in enumerate(to_process[:15], 1):
            fn = os.path.basename(fp)
            kw, kw_human = extract_keywords_from_filename(fn)
            vis_human = await asyncio.to_thread(detect_human_visually, fp)
            is_human = kw_human or vis_human
            if args.character == "none":
                char_label = "none"
            elif args.character in ["ogarla", "loveless"]:
                char_label = f"{args.character}.{int(args.char_weight * 100)}"
            else:  # smart
                char_label = f"ogarla.{int(args.char_weight * 100)}" if is_human else "none"

            if args.style_mode == "cycle":
                st = STYLE_VARIATIONS[(idx - 1) % len(STYLE_VARIATIONS)]
            elif args.style_mode == "random":
                st = STYLE_VARIATIONS[random.randint(0, len(STYLE_VARIATIONS) - 1)]
            else:
                st = {"name": "Default Sensual"}

            with Image.open(fp) as im:
                rw, rh, tw, th = calculate_zero_distortion_dimensions(im.width, im.height)
                ar_val = im.width / im.height
            logger.info(f"  [{idx:2d}] {fn[:25]:<25} | AR: {ar_val:.2f} | Style: {st['name'][:24]:<24} | Human: {str(is_human):<5} | Char: {char_label:<10}")
        logger.info(f"[Dry Run] Exiting without generating. Total remaining: {len(to_process)}")
        return

    if not to_process:
        logger.info("All target images are already processed! Exiting.")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    # Verify ComfyUI is running
    is_online = await check_comfyui_online(client)
    if not is_online:
        logger.warning(f"ComfyUI is not currently online at {COMFYUI_ADDRESS}!")
        if os.path.exists(COMFYUI_BATCH_PATH):
            logger.info(f"Attempting to launch ComfyUI via {COMFYUI_BATCH_PATH}...")
            try:
                subprocess.Popen(["cmd.exe", "/c", COMFYUI_BATCH_PATH], creationflags=subprocess.CREATE_NEW_CONSOLE)
                logger.info("Launched ComfyUI console. Waiting 15s for server startup...")
                for attempt in range(15):
                    await asyncio.sleep(2)
                    if await check_comfyui_online(client):
                        logger.info("ComfyUI is now online and ready!")
                        is_online = True
                        break
            except Exception as e:
                logger.error(f"Failed to auto-launch ComfyUI: {e}")

        if not is_online:
            logger.error(
                f"Cannot connect to ComfyUI at {COMFYUI_ADDRESS}. "
                "Please start ComfyUI using run_nvidia_gpu.bat, then run this script again."
            )
            return

    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_start = time.time()

    try:
        for idx, fp in enumerate(to_process, 1):
            fname = os.path.basename(fp)
            base = os.path.splitext(fname)[0]
            out_jpg_path = os.path.join(output_dir, f"{base}_sensual_krea2.jpg")

            try:
                t0 = time.time()

                # 1. Read input dimensions & calculate zero-distortion dimensions
                def _get_info():
                    with Image.open(fp) as im:
                        return im.size
                orig_w, orig_h = await asyncio.to_thread(_get_info)

                render_w, render_h, target_w, target_h = calculate_zero_distortion_dimensions(orig_w, orig_h)

                # 2. Dual-pass human detection (Keyword + Visual Cascade)
                keywords, kw_human = extract_keywords_from_filename(fname)
                vis_human = await asyncio.to_thread(detect_human_visually, fp)
                is_human = kw_human or vis_human

                if args.character == "none":
                    char_choice = "none"
                elif args.character in ["ogarla", "loveless"]:
                    char_choice = f"{args.character}.{int(args.char_weight * 100)}"
                else:  # smart
                    char_choice = f"ogarla.{int(args.char_weight * 100)}" if is_human else "none"

                # 3. Atmospheric Style Variation Selection
                if args.style_mode == "cycle":
                    style = STYLE_VARIATIONS[(idx - 1) % len(STYLE_VARIATIONS)]
                elif args.style_mode == "random":
                    style = random.choice(STYLE_VARIATIONS)
                else:
                    style = {"name": "Default Sensual", "flavor": "", "wetness": args.wetness}

                style_name = style["name"]
                chosen_wetness = style.get("wetness", args.wetness) if args.style_mode != "none" else args.wetness

                prompt_parts = [args.sensual_prompt]
                if style.get("flavor"):
                    prompt_parts.append(style["flavor"])
                if keywords:
                    prompt_parts.append(keywords)
                fused_prompt = ", ".join(prompt_parts)

                # 4. Upload source image to ComfyUI
                def _read_bytes():
                    with open(fp, "rb") as f:
                        return f.read()
                img_bytes = await asyncio.to_thread(_read_bytes)

                upload_name = f"krea_sensual_init_{base[:30]}.png"
                upload_res = await client.upload_image(img_bytes, filename=upload_name, overwrite=True)
                uploaded_name = upload_res.get("name", upload_name)

                img_seed = args.seed if args.seed is not None else random.randint(1, 1125899906842624)

                # 5. Build Krea 2 Turbo Bertflow workflow
                wf = prepare_bertflow_workflow(
                    prompt=fused_prompt,
                    width=render_w,
                    height=render_h,
                    seed=img_seed,
                    steps=args.steps,
                    unet_model=KREA2_UNET,
                    wetness_strength=chosen_wetness,
                    init_image=uploaded_name,
                    comp_strength="medium",
                    character=char_choice,
                    celebrity="none",
                    filename_prefix=f"sb_sensual_{base[:20]}"
                )

                # Override start_at_step for exact denoise ratio
                start_step = max(1, min(args.steps - 1, round(args.steps * (1.0 - args.denoise))))
                if "599" in wf and "inputs" in wf["599"]:
                    wf["599"]["inputs"]["start_at_step"] = start_step

                aspect_str = f"{render_w / render_h:.2f}"
                logger.info(
                    f"\n[{idx}/{len(to_process)}] Rendering {fname} via Krea 2 ({render_w}x{render_h}, AR: {aspect_str}) | "
                    f"Style: {style_name} | Char: {char_choice} | Wetness: {chosen_wetness:.2f} | Denoise: {args.denoise:.2f} (step {start_step}/{args.steps})..."
                )

                # 5. Execute Krea 2 render
                render_outputs = await client.generate(
                    wf, command_type="blend-krea", description=f"Krea2 Sensual {base[:15]}"
                )
                render_bytes = await extract_image_bytes(client, render_outputs)

                if not render_bytes:
                    logger.error(f"Failed to retrieve Krea 2 render for {fname}")
                    continue

                t_render = time.time() - t0

                # 6. Super-resolve with 4x-UltraSharp if enabled
                final_bytes = render_bytes
                t_upscale = 0.0

                if not args.no_upscale:
                    t_up0 = time.time()
                    up_upload_name = f"krea_sensual_up_in_{base[:30]}.png"
                    up_upload_res = await client.upload_image(render_bytes, filename=up_upload_name, overwrite=True)
                    up_uploaded_name = up_upload_res.get("name", up_upload_name)

                    up_wf = build_fast_upscale_workflow(
                        image_filename=up_uploaded_name,
                        target_width=target_w,
                        target_height=target_h,
                        model_name=UPSCALE_MODEL,
                        filename_prefix=f"sb_sensual_1080p_{base[:20]}"
                    )

                    up_outputs = await client.generate(
                        up_wf, command_type="upscale-fast", description=f"Up {base[:15]}"
                    )
                    final_bytes = await extract_image_bytes(client, up_outputs) or render_bytes
                    t_upscale = time.time() - t_up0

                # 7. Save output JPEG
                await save_image_async(final_bytes, out_jpg_path, format_name="JPEG", quality=95)

                t_elapsed = time.time() - t0
                jpg_sz_mb = os.path.getsize(out_jpg_path) / (1024 * 1024)

                # 8. Record progress
                meta = {
                    "source_file": fname,
                    "output_file": os.path.basename(out_jpg_path),
                    "source_dims": [orig_w, orig_h],
                    "render_dims": [render_w, render_h],
                    "final_dims": [target_w, target_h] if not args.no_upscale else [render_w, render_h],
                    "is_human": is_human,
                    "character_lora": char_choice,
                    "style_variant": style_name,
                    "wetness": chosen_wetness,
                    "denoise": args.denoise,
                    "steps": args.steps,
                    "seed": img_seed,
                    "krea_time_s": round(t_render, 2),
                    "upscale_time_s": round(t_upscale, 2),
                    "total_time_s": round(t_elapsed, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                progress[fname] = meta
                save_json(progress_file, progress)

                success_count += 1
                logger.info(
                    f"SUCCESS Saved {out_jpg_path} ({jpg_sz_mb:.2f} MB in {t_elapsed:.1f}s | "
                    f"Krea2: {t_render:.1f}s, Upscale: {t_upscale:.1f}s)"
                )

            except Exception as e:
                logger.error(f"Error processing {fname}: {e}", exc_info=True)
                await asyncio.sleep(2)
    finally:
        await client.stop()

    total_time = time.time() - t_start
    logger.info(f"\nFinished batch pass. Processed {success_count} images in {total_time:.1f}s.")


if __name__ == "__main__":
    asyncio.run(main())
