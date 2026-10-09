"""
Batch Re-Blend 21:9 Generator for Krea Blends.

Standardizes all 16:9 wallpapers in C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends
into authentic, native 21:9 (2528x1080) wallpapers:
1. Extracts original prompt, seed, and LoRA parameters embedded in each blend_*.png file.
2. Re-synthesizes the scene natively across a wide 1872x800 canvas (native 21:9) using Krea 2 Turbo Diffusion.
   - Zero seams, zero border cutoffs, zero perspective distortion.
3. Super-resolves each render to 2528x1080 using ComfyUI 4x-UltraSharp.
4. Outputs the final collection to C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends\\1080p_21x9.

Includes full resume support via reblend_21x9_progress.json.
"""

import os
import sys
import io
import time
import json
import glob
import shutil
import argparse
import asyncio
import logging
from typing import Optional, Dict, Any
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchReblend21x9")

DEFAULT_KREA_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_KREA_DIR, "1080p_21x9")
PROGRESS_FILE_NAME = "reblend_21x9_progress.json"
TARGET_WIDTH = 2528
TARGET_HEIGHT = 1080
NATIVE_21X9_W = 1872
NATIVE_21X9_H = 800
UPSCALE_MODEL = "4x-UltraSharp.pth"


def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress_file: str, prog: dict):
    try:
        temp = f"{progress_file}.tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2)
        os.replace(temp, progress_file)
    except Exception as e:
        logger.error(f"Error saving progress: {e}")


def prepare_21x9_workflow(src_png_path: str) -> Optional[Dict[str, Any]]:
    """Loads embedded prompt metadata from a blend_*.png file and updates dimensions to 21:9."""
    try:
        with Image.open(src_png_path) as im:
            if "prompt" not in im.info:
                return None
            wf = json.loads(im.info["prompt"])

        # Update empty latent dimensions to 1872x800
        if "698" in wf and "inputs" in wf["698"]:
            wf["698"]["inputs"]["width"] = NATIVE_21X9_W
            wf["698"]["inputs"]["height"] = NATIVE_21X9_H

        # Update init image scaling if present
        if "901" in wf and "inputs" in wf["901"]:
            wf["901"]["inputs"]["width"] = NATIVE_21X9_W
            wf["901"]["inputs"]["height"] = NATIVE_21X9_H

        base_name = os.path.splitext(os.path.basename(src_png_path))[0]
        if "830" in wf and "inputs" in wf["830"]:
            wf["830"]["inputs"]["filename_prefix"] = f"reblend_21x9_{base_name[:30]}"

        return wf
    except Exception as e:
        logger.error(f"Error extracting workflow from {src_png_path}: {e}")
        return None


async def extract_image_bytes(client: ComfyClient, outputs: Any) -> Optional[bytes]:
    """Helper to extract generated image bytes from ComfyUI response."""
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
        return outputs[0]
    elif isinstance(outputs, dict):
        for k, v in outputs.items():
            if isinstance(v, dict) and "images" in v:
                for item in v["images"]:
                    img_bytes = await client.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                    if img_bytes:
                        return img_bytes
    return None


async def render_and_upscale_21x9(
    client: ComfyClient,
    src_png_path: str,
    output_jpg_path: str,
    base_name: str
) -> dict:
    """Renders 1872x800 native 21:9 blend, then upscales to 2528x1080."""
    t0 = time.time()

    wf = prepare_21x9_workflow(src_png_path)
    if not wf:
        raise ValueError("Could not extract embedded workflow from source PNG")

    # Step 1: Render native 21:9 in Krea 2
    t_gen_start = time.time()
    outputs = await client.generate(wf, command_type="reblend-21x9", description=f"Re-blend {base_name[:20]}")
    gen_bytes = await extract_image_bytes(client, outputs)
    if not gen_bytes:
        raise RuntimeError("Failed to retrieve generated 21:9 image bytes from ComfyUI")
    t_gen = time.time() - t_gen_start

    # Step 2: AI Upscale with 4x-UltraSharp to 2528x1080
    t_up_start = time.time()
    upload_res = await client.upload_image(gen_bytes, filename=f"reblend_in_{base_name[:30]}.png")
    uploaded_name = upload_res.get("name", f"reblend_in_{base_name[:30]}.png")

    up_wf = build_fast_upscale_workflow(
        image_filename=uploaded_name,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        model_name=UPSCALE_MODEL,
        filename_prefix=f"up_{base_name[:25]}"
    )
    up_outputs = await client.generate(up_wf, command_type="upscale-reblend", description=f"Up {base_name[:20]}")
    up_bytes = await extract_image_bytes(client, up_outputs)
    if not up_bytes:
        raise RuntimeError("Failed to retrieve upscaled image bytes from ComfyUI")
    t_up = time.time() - t_up_start

    # Step 3: Save as high-quality JPEG
    def save_jpg():
        with Image.open(io.BytesIO(up_bytes)) as up_im:
            if up_im.mode != "RGB":
                up_im = up_im.convert("RGB")
            up_im.save(output_jpg_path, format="JPEG", quality=92, optimize=True)

    await asyncio.to_thread(save_jpg)

    elapsed = time.time() - t0
    jpg_size_mb = os.path.getsize(output_jpg_path) / (1024 * 1024)

    return {
        "dimensions": [TARGET_WIDTH, TARGET_HEIGHT],
        "render_time_sec": round(t_gen, 1),
        "upscale_time_sec": round(t_up, 1),
        "total_elapsed_sec": round(elapsed, 1),
        "jpg_size_mb": round(jpg_size_mb, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


async def main():
    parser = argparse.ArgumentParser(description="Batch 21:9 Re-Blend Generator (Option A)")
    parser.add_argument("--source-dir", default=DEFAULT_KREA_DIR, help="Base krea_blends directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Target folder for 21:9 images")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to re-render")
    parser.add_argument("--dry-run", action="store_true", help="List images without running generation")
    args = parser.parse_args()

    krea_dir = args.source_dir
    jpg_dir = os.path.join(krea_dir, "1080p_jpg")
    output_dir = args.output_dir
    progress_file = os.path.join(output_dir, PROGRESS_FILE_NAME)

    os.makedirs(output_dir, exist_ok=True)
    progress = load_progress(progress_file)

    all_jpgs = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))

    # Identify 16:9 images that need re-blending
    sixteen_nine_files = []
    already_21x9_files = []

    for fpath in all_jpgs:
        fname = os.path.basename(fpath)
        with Image.open(fpath) as im:
            w, h = im.size
        aspect = w / h
        if aspect >= 2.1:
            already_21x9_files.append(fpath)
        else:
            sixteen_nine_files.append(fpath)

    logger.info(f"\n{'='*70}\nBATCH 21:9 NATIVE RE-BLEND (OPTION A)\n{'='*70}")
    logger.info(f"Source Base Dir : {krea_dir}")
    logger.info(f"Target Output   : {output_dir}")
    logger.info(f"Total Wallpapers: {len(all_jpgs)}")
    logger.info(f"Already 21:9    : {len(already_21x9_files)} images")
    logger.info(f"Target 16:9     : {len(sixteen_nine_files)} images to re-render natively")
    logger.info(f"{'='*70}\n")

    # Step 1: Ensure pre-existing 21:9 images are copied to output_dir
    for fpath in already_21x9_files:
        dest_p = os.path.join(output_dir, os.path.basename(fpath))
        if not os.path.exists(dest_p) and not args.dry_run:
            shutil.copy2(fpath, dest_p)

    # Step 2: Identify which 16:9 blends still need re-rendering
    to_process = []
    for fpath in sixteen_nine_files:
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]
        src_png = os.path.join(krea_dir, f"{base}.png")
        if not os.path.exists(src_png):
            logger.warning(f"Could not find source PNG for {fname}, skipping")
            continue

        if fname not in progress:
            to_process.append((fname, base, src_png))

    logger.info(f"Remaining 16:9 wallpapers to re-blend: {len(to_process)}")

    if args.dry_run:
        logger.info("[Dry Run] Exiting without generating.")
        return

    if args.limit:
        to_process = to_process[:args.limit]
        logger.info(f"Applied limit: processing {len(to_process)} images.")

    if not to_process:
        logger.info(f"All images already re-blended in {output_dir}!")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)
    success_count = 0
    start_time = time.time()

    try:
        for idx, (fname, base, src_png) in enumerate(to_process, 1):
            dest_jpg = os.path.join(output_dir, fname)
            logger.info(f"[{idx}/{len(to_process)}] Re-blending 21:9: {fname}...")

            try:
                res = await render_and_upscale_21x9(client, src_png, dest_jpg, base)
                progress[fname] = res
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(
                    f"  -> SUCCESS in {res['total_elapsed_sec']}s "
                    f"(Krea2: {res['render_time_sec']}s, 4x-UltraSharp: {res['upscale_time_sec']}s) | "
                    f"{res['jpg_size_mb']} MB"
                )
            except Exception as e:
                logger.error(f"  -> Failed processing {fname}: {e}")
    finally:
        try:
            await client.close()
        except Exception:
            pass

    total_time = time.time() - start_time
    logger.info(f"\n{'='*70}\nBATCH 21:9 RE-BLEND COMPLETE!\n{'='*70}")
    logger.info(f"Processed : {success_count}/{len(to_process)} images in {total_time:.1f}s")
    logger.info(f"Destination: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
