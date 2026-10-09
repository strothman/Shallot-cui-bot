"""
Batch Halloween Decor & Spooky Season Makeover Runner for Star Board Pictures using Krea 2.

Processes images in C:\\Users\\strot\\Pictures\\star board\\picture:
1. Discovers source images (.png, .jpg, .jpeg, .webp).
2. Detects closest Krea 2 aspect ratio (preserving landscape, portrait, square, ultrawide).
3. Evaluates scene keywords and human presence for optional character LoRA styling (Ogarla Krea 2).
4. Synthesizes a festive Halloween makeover via Krea 2 Turbo Diffusion (museByStableYogi_v35Int8Extended):
   - Carved glowing jack-o'-lanterns, flickering candlelight, autumn pumpkins, subtle cobwebs,
     eerie warm orange and twilight glow, festive Halloween atmosphere.
   - Composition retention at configurable denoise (default: 0.60).
5. Optionally super-resolves each output using 4x-UltraSharp.
6. Saves results directly to C:\\Users\\strot\\Pictures\\star board\\halloween_krea2 with full resume tracking.
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
from typing import Optional, Dict, Any, Tuple
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from parsers.workflows import prepare_bertflow_workflow
from parsers import resolve_bertflow_dimensions
from image_utils import detect_closest_krea_aspect_ratio
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchHalloweenStarBoard")

DEFAULT_SOURCE_DIR = r"C:\Users\strot\Pictures\star board\picture"
DEFAULT_OUTPUT_DIR = r"C:\Users\strot\Pictures\star board\halloween_krea2"
PROGRESS_FILE_NAME = "halloween_krea2_progress.json"
UPSCALE_MODEL = "4x-UltraSharp.pth"
KREA2_UNET = "museByStableYogi_v35Int8Extended.safetensors"

DEFAULT_HALLOWEEN_PREFIX = (
    "festive Halloween decorations, carved glowing jack-o'-lanterns with warm candlelight in background, "
    "autumn pumpkins, subtle cobwebs, eerie warm orange ambient lighting, preserving original composition"
)

HUMAN_KEYWORDS = [
    r"\bwoman\b", r"\bwomen\b", r"\bman\b", r"\bmen\b", r"\bgirl\b", r"\bgirls\b",
    r"\bboy\b", r"\bboys\b", r"\bperson\b", r"\bpeople\b", r"\bhuman\b", r"\bhumans\b",
    r"\bfemale\b", r"\bmale\b", r"\blady\b", r"\bladies\b", r"\bguy\b", r"\bguys\b",
    r"\bportrait\b", r"\bface\b", r"\bmodel\b", r"\bcharacters?\b", r"\bfigure\b",
    r"\bdancer\b", r"\bbride\b", r"\bqueen\b", r"\bking\b", r"\bprincess\b", r"\bprince\b"
]
HUMAN_REGEX = re.compile("|".join(HUMAN_KEYWORDS), re.IGNORECASE)


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
    """Cleans filename into usable scene keywords and checks for human indicators."""
    base = os.path.splitext(filename)[0]
    cleaned = re.sub(r'^\d+_stars_|\b(?:t2i|txt2img|img2img|img2watercoal)\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', ' ', base, flags=re.IGNORECASE)
    cleaned = re.sub(r'-(?:waiIllustrious|hyphoria|illustrij|Copax|cyberrealistic|illustrious)[^-\s]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'-(?:dpmpp|euler|ddim)[^-\s]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    has_human = bool(HUMAN_REGEX.search(cleaned)) or "donana" in base.lower() or "sophil" in base.lower()
    return cleaned, has_human


def calculate_zero_distortion_dimensions(orig_w: int, orig_h: int, base_area: int = 1498176) -> Tuple[int, int, int, int]:
    """
    Calculates Krea 2 render dimensions and proportional upscale dimensions
    that maintain exact 1:1 aspect ratio with the source image, eliminating smooshing.
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
    """Helper to extract generated image bytes from ComfyUI response."""
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
        return outputs[0]
    elif isinstance(outputs, dict):
        for k, v in outputs.items():
            if isinstance(v, dict) and "images" in v:
                for item in v["images"]:
                    b = await client.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                    if b:
                        return b
    return None


async def save_image_async(image_bytes: bytes, output_path: str, format_name: str = "JPEG", quality: int = 92):
    """Saves image bytes off-thread to avoid blocking the asyncio event loop."""
    def _save():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with Image.open(io.BytesIO(image_bytes)) as img:
            if format_name.upper() == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")
            img.save(output_path, format=format_name, quality=quality, optimize=True)

    await asyncio.to_thread(_save)


async def main():
    parser = argparse.ArgumentParser(description="Batch Halloween Krea 2 Makeover for Star Board Picture Folder")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Source picture directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Destination directory for Halloween renders")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--files", nargs="*", default=None, help="Specific image filenames to process")
    parser.add_argument("--denoise", type=float, default=0.35, help="Denoising strength (default: 0.35 for pristine composition retention)")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps for Krea 2 Turbo (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed (default: randomized per image)")
    parser.add_argument("--halloween-theme", default=DEFAULT_HALLOWEEN_PREFIX, help="Custom Halloween decor prompt prefix")
    parser.add_argument("--force-ogarla", action="store_true", help="Force Ogarla Krea 2 LoRA on all images")
    parser.add_argument("--ogarla", action="store_true", help="Apply subtle Ogarla likeness LoRA when humans are detected")
    parser.add_argument("--ogarla-weight", type=float, default=0.45, help="Ogarla LoRA weight if enabled (default: 0.45)")
    parser.add_argument("--no-character", action="store_true", help="Disable character LoRA completely (preserves original character/art style)")
    parser.add_argument("--no-upscale", action="store_true", help="Skip 4x-UltraSharp 1080p upscale and save native Krea renders")
    parser.add_argument("--reprocess-distorted", action="store_true", help="Reprocess previously completed images that had aspect ratio distortion")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and target files without running generation")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    progress_file = os.path.join(output_dir, PROGRESS_FILE_NAME)

    os.makedirs(output_dir, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    raw_files = [
        f for f in glob.glob(os.path.join(source_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
    ]
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
        out_jpg = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_halloween_krea2.jpg")
        cur_meta = progress.get(fname, {})
        is_distorted = False
        if cur_meta and "render_dims" in cur_meta and "final_dims" in cur_meta:
            r_ar = cur_meta["render_dims"][0] / cur_meta["render_dims"][1]
            f_ar = cur_meta["final_dims"][0] / cur_meta["final_dims"][1]
            if abs(r_ar - f_ar) / r_ar > 0.01:
                is_distorted = True

        if fname not in progress or not os.path.exists(out_jpg) or cur_meta.get("denoise") != args.denoise or (args.reprocess_distorted and is_distorted):
            to_process.append(fp)

    if args.limit:
        to_process = to_process[:args.limit]

    logger.info(f"\n{'='*75}\nSTARTING BATCH HALLOWEEN KREA 2 MAKEOVER (STAR BOARD)\n{'='*75}")
    logger.info(f"Source Folder         : {source_dir}")
    logger.info(f"Output Folder         : {output_dir}")
    logger.info(f"Total Source Images   : {len(target_files)}")
    logger.info(f"Remaining to Process  : {len(to_process)}")
    logger.info(f"Krea 2 UNET           : {KREA2_UNET}")
    logger.info(f"Denoise Level         : {args.denoise:.2f}")
    logger.info(f"Steps                 : {args.steps}")
    logger.info(f"Upscale Enabled       : {not args.no_upscale} ({UPSCALE_MODEL})")
    logger.info(f"Reprocess Distorted   : {args.reprocess_distorted}")
    logger.info(f"Halloween Prefix      : {args.halloween_theme[:75]}...")
    logger.info(f"{'='*75}\n")

    if args.dry_run:
        logger.info("[Dry Run] Listing plan for up to 10 target images:")
        for idx, fp in enumerate(to_process[:10], 1):
            fn = os.path.basename(fp)
            keywords, has_human = extract_keywords_from_filename(fn)
            char_tag = "ogarla.85" if (has_human or args.force_ogarla) and not args.no_character else "none"
            with Image.open(fp) as im:
                rw, rh, tw, th = calculate_zero_distortion_dimensions(im.width, im.height)
                ar_val = im.width / im.height
            logger.info(f"  [{idx}] {fn[:35]:<35} | Orig: {ar_val:.3f} | Render: {rw}x{rh} | Up: {tw}x{th} | Char: {char_tag:<10}")
        logger.info(f"[Dry Run] Exiting without generating. Total remaining: {len(to_process)}")
        return

    if not to_process:
        logger.info("All target images are already processed! Exiting.")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_start = time.time()

    for idx, fp in enumerate(to_process, 1):
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        out_jpg_path = os.path.join(output_dir, f"{base}_halloween_krea2.jpg")

        try:
            t0 = time.time()

            # 1. Read input dimensions & calculate zero-distortion dimensions
            def _get_info():
                with Image.open(fp) as im:
                    return im.size
            orig_w, orig_h = await asyncio.to_thread(_get_info)

            render_w, render_h, target_w, target_h = calculate_zero_distortion_dimensions(orig_w, orig_h)

            # 2. Extract scene keywords & character selection
            keywords, has_human = extract_keywords_from_filename(fname)
            if args.force_ogarla:
                char_choice = f"ogarla.{int(args.ogarla_weight*100)}"
            elif (has_human or args.ogarla) and not args.no_character and args.ogarla:
                char_choice = f"ogarla.{int(args.ogarla_weight*100)}"
            else:
                char_choice = "none"

            # 3. Fuse prompt
            prompt_parts = [args.halloween_theme]
            if keywords:
                prompt_parts.append(keywords)
            fused_prompt = ", ".join(prompt_parts)

            # 4. Upload image to ComfyUI
            def _read_bytes():
                with open(fp, "rb") as f:
                    return f.read()
            img_bytes = await asyncio.to_thread(_read_bytes)

            upload_name = f"krea_sb_init_{base[:30]}.png"
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
                wetness_strength=-2.0,
                init_image=uploaded_name,
                comp_strength="medium",
                character=char_choice,
                celebrity="none",
                filename_prefix=f"halloween_sb_{base[:20]}"
            )

            # Override start_at_step for exact denoise ratio
            start_step = max(1, min(args.steps - 1, round(args.steps * (1.0 - args.denoise))))
            if "599" in wf and "inputs" in wf["599"]:
                wf["599"]["inputs"]["start_at_step"] = start_step

            aspect_str = f"{render_w / render_h:.2f}"
            logger.info(
                f"\n[{idx}/{len(to_process)}] Rendering {fname} via Krea 2 ({render_w}x{render_h}, AR: {aspect_str}) | "
                f"Char: {char_choice} | Denoise: {args.denoise:.2f} (step {start_step}/{args.steps})..."
            )

            # 6. Execute Krea 2 render
            render_outputs = await client.generate(
                wf, command_type="blend-krea", description=f"Halloween Krea2 {base[:15]}"
            )
            render_bytes = await extract_image_bytes(client, render_outputs)

            if not render_bytes:
                logger.error(f"Failed to retrieve Krea 2 render for {fname}")
                continue

            t_render = time.time() - t0

            # 7. Super-resolve with 4x-UltraSharp if enabled
            final_bytes = render_bytes
            t_upscale = 0.0

            if not args.no_upscale:
                t_up0 = time.time()
                up_upload_name = f"krea_sb_up_in_{base[:30]}.png"
                up_upload_res = await client.upload_image(render_bytes, filename=up_upload_name, overwrite=True)
                up_uploaded_name = up_upload_res.get("name", up_upload_name)

                up_wf = build_fast_upscale_workflow(
                    image_filename=up_uploaded_name,
                    target_width=target_w,
                    target_height=target_h,
                    model_name=UPSCALE_MODEL,
                    filename_prefix=f"halloween_sb_1080p_{base[:20]}"
                )

                up_outputs = await client.generate(
                    up_wf, command_type="upscale-fast", description=f"Up {base[:15]}"
                )
                final_bytes = await extract_image_bytes(client, up_outputs) or render_bytes
                t_upscale = time.time() - t_up0

            # 8. Save output JPEG
            await save_image_async(final_bytes, out_jpg_path, format_name="JPEG", quality=92)

            t_elapsed = time.time() - t0
            jpg_sz_mb = os.path.getsize(out_jpg_path) / (1024 * 1024)

            # 9. Record progress
            meta = {
                "source_file": fname,
                "output_file": os.path.basename(out_jpg_path),
                "source_dims": [orig_w, orig_h],
                "render_dims": [render_w, render_h],
                "final_dims": [target_w, target_h] if not args.no_upscale else [render_w, render_h],
                "aspect_ratio": round(orig_w / orig_h, 3),
                "character": char_choice,
                "has_human": has_human,
                "denoise": args.denoise,
                "seed": img_seed,
                "start_step": start_step,
                "steps": args.steps,
                "render_sec": round(t_render, 1),
                "upscale_sec": round(t_upscale, 1),
                "total_sec": round(t_elapsed, 1),
                "file_size_mb": round(jpg_sz_mb, 2),
                "fused_prompt": fused_prompt[:250],
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            progress[fname] = meta
            save_json(progress_file, progress)
            success_count += 1

            logger.info(
                f"SUCCESS Saved {out_jpg_path} "
                f"({jpg_sz_mb:.2f} MB in {t_elapsed:.1f}s | Krea2: {t_render:.1f}s, Upscale: {t_upscale:.1f}s)"
            )

        except Exception as e:
            logger.error(f"[{idx}/{len(to_process)}] Error processing {fname}: {e}", exc_info=True)

    t_total = time.time() - t_start
    avg_sec = (t_total / success_count) if success_count else 0
    logger.info(
        f"\n{'='*75}\nHALLOWEEN KREA 2 BATCH PASS COMPLETE: {success_count}/{len(to_process)} finished in "
        f"{t_total/60:.1f} mins ({avg_sec:.1f}s/img)!\nOutputs saved to: {output_dir}\n{'='*75}\n"
    )

    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass
    try:
        await client.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
