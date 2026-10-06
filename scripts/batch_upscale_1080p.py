"""
Batch 1080p AI Upscaler for Krea Blends.
Processes all wallpapers in C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends
Produces two outputs per image simultaneously:
  1. High-Quality JPEG (Quality 92) -> 1080p_jpg/ (~300-450 KB)
  2. Lossless PNG                   -> 1080p_png/ (~2.3-2.6 MB)
Maintains exact aspect ratios (16:9 -> 1920x1080, 21:9 -> 2528x1080).
Includes resume support via upscale_1080p_progress.json.
"""

import os
import sys
import io
import time
import json
import glob
import asyncio
import logging
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchUpscale1080p")

SOURCE_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends"
PNG_DIR = os.path.join(SOURCE_DIR, "1080p_png")
JPG_DIR = os.path.join(SOURCE_DIR, "1080p_jpg")
PROGRESS_FILE = os.path.join(SOURCE_DIR, "upscale_1080p_progress.json")
UPSCALE_MODEL = "4x-UltraSharp.pth"


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(prog: dict):
    try:
        temp = f"{PROGRESS_FILE}.tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2)
        os.replace(temp, PROGRESS_FILE)
    except Exception as e:
        logger.error(f"Error saving progress: {e}")


async def main():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(JPG_DIR, exist_ok=True)

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    all_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "blend_*.png")))
    progress = load_progress()

    to_process = []
    for fp in all_files:
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        png_out = os.path.join(PNG_DIR, f"{base}.png")
        jpg_out = os.path.join(JPG_DIR, f"{base}.jpg")
        if fname not in progress or not os.path.exists(png_out) or not os.path.exists(jpg_out):
            to_process.append(fp)

    logger.info(f"\n{'='*65}\nSTARTING 1080p AI UPSCALE BATCH ({UPSCALE_MODEL})\n{'='*65}")
    logger.info(f"Total target images: {len(all_files)}")
    logger.info(f"Already completed  : {len(all_files) - len(to_process)}")
    logger.info(f"Remaining to upscale: {len(to_process)}")
    logger.info(f"JPEG folder: {JPG_DIR}")
    logger.info(f"PNG folder : {PNG_DIR}")
    logger.info(f"{'='*65}\n")

    if not to_process:
        logger.info("All images already upscaled!")
        return

    start_time = time.time()
    success_count = 0

    for idx, fp in enumerate(to_process, 1):
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        png_out = os.path.join(PNG_DIR, f"{base}.png")
        jpg_out = os.path.join(JPG_DIR, f"{base}.jpg")

        t0 = time.time()

        try:
            with Image.open(fp) as im:
                orig_w, orig_h = im.size

            aspect = orig_w / orig_h
            if 1.70 <= aspect <= 1.85:
                target_w, target_h = 1920, 1080
            elif aspect >= 2.1:
                target_w, target_h = 2528, 1080
            else:
                target_h = 1080
                target_w = int(round((target_h * aspect) / 8) * 8)

            with open(fp, "rb") as f:
                img_bytes = f.read()

            upload_name = f"upscale_in_{base[:40]}.png"
            upload_res = await client.upload_image(img_bytes, filename=upload_name)
            uploaded_name = upload_res.get("name", upload_name)

            wf = build_fast_upscale_workflow(
                image_filename=uploaded_name,
                target_width=target_w,
                target_height=target_h,
                model_name=UPSCALE_MODEL,
                filename_prefix=f"up_{base[:30]}"
            )

            outputs = await client.generate(wf, command_type="upscale-fast", description=f"Up {base[:20]}")

            gen_bytes = None
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
                gen_bytes = outputs[0]
            elif isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, dict) and "images" in v:
                        for item in v["images"]:
                            gen_bytes = await client.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                            if gen_bytes:
                                break
                    if gen_bytes:
                        break

            if not gen_bytes:
                logger.error(f"[{idx}/{len(to_process)}] Failed to retrieve image for {fname}")
                continue

            with open(png_out, "wb") as f_png:
                f_png.write(gen_bytes)

            with Image.open(io.BytesIO(gen_bytes)) as up_img:
                if up_img.mode != "RGB":
                    up_img = up_img.convert("RGB")
                up_img.save(jpg_out, format="JPEG", quality=92, optimize=True)

            png_sz_mb = os.path.getsize(png_out) / (1024 * 1024)
            jpg_sz_mb = os.path.getsize(jpg_out) / (1024 * 1024)
            elapsed = time.time() - t0

            progress[fname] = {
                "orig_dims": [orig_w, orig_h],
                "upscaled_dims": [target_w, target_h],
                "png_size_mb": round(png_sz_mb, 2),
                "jpg_size_mb": round(jpg_sz_mb, 2),
                "elapsed_sec": round(elapsed, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_progress(progress)
            success_count += 1

            logger.info(
                f"[{idx}/{len(to_process)}] OK ({elapsed:.1f}s) | {orig_w}x{orig_h} -> {target_w}x{target_h} | "
                f"PNG: {png_sz_mb:.2f}MB, JPG: {jpg_sz_mb:.2f}MB | {fname[:45]}"
            )

        except Exception as e:
            logger.error(f"[{idx}/{len(to_process)}] Error upscaling {fname}: {e}")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*65}\n1080p UPSCALE COMPLETE: {success_count}/{len(to_process)} upscaled in {total_time/60:.1f} mins!\n{'='*65}")

    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
