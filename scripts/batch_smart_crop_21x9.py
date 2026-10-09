"""
Batch Smart Cropper & 21:9 Standardizer for Krea Blends.

Standardizes all wallpapers in C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends to 21:9 (2528x1080):
1. Copies all existing 21:9 (2528x1080) wallpapers directly (no re-compression).
2. For 16:9 (1920x1080) wallpapers:
   - Performs content-aware smart cropping using facial detection (Haar cascade) and
     edge/gradient energy saliency to preserve focal subjects and headroom.
   - Upscales the cropped 1920x820 frame to pristine 2528x1080 using ComfyUI (4x-UltraSharp)
     or high-grade Lanczos.
3. Produces a unified folder containing 100% 21:9 (2528x1080) wallpapers.
Includes resume support via smart_crop_21x9_progress.json.
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
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartCrop21x9")

DEFAULT_KREA_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_KREA_DIR, "1080p_21x9")
TARGET_WIDTH = 2528
TARGET_HEIGHT = 1080
UPSCALE_MODEL = "4x-UltraSharp.pth"

# Load Haar cascades for face detection
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


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


def calculate_smart_crop_top(pil_image: Image.Image, target_crop_h: int) -> Tuple[int, dict]:
    """
    Calculates the optimal vertical offset (top) to crop an image to target_crop_h.
    Uses facial recognition and gradient energy density.
    """
    orig_w, orig_h = pil_image.size
    max_top = orig_h - target_crop_h
    if max_top <= 0:
        return 0, {"faces": 0, "strategy": "none"}

    # Convert to grayscale numpy array for OpenCV analysis
    np_img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # 1. Detect faces
    faces = list(_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)))
    if not faces:
        faces = list(_profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)))

    # 2. Compute edge / gradient energy map
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    row_energy = np.sum(mag, axis=1)  # shape: (orig_h,)

    # 3. Score every candidate top offset
    scores = []
    max_row = np.max(row_energy) if len(row_energy) > 0 else 1.0

    for top in range(max_top + 1):
        # Base energy within slice
        e = float(np.sum(row_energy[top : top + target_crop_h]))

        # Face preservation penalty:
        face_penalty = 0.0
        if faces:
            for (fx, fy, fw, fh) in faces:
                face_top = fy - int(0.2 * fh)  # Add 20% headroom
                face_bottom = fy + fh
                # Severe penalty if head is sliced
                if face_top < top:
                    face_penalty += (top - face_top) * 50000.0
                if face_bottom > (top + target_crop_h):
                    face_penalty += (face_bottom - (top + target_crop_h)) * 50000.0

        # Gentle center prior (slight rule-of-thirds upward tilt)
        center_target = max_top * 0.45
        center_dist = abs(top - center_target)
        center_bias = - (center_dist ** 2) * (max_row / (max_top ** 2 + 1)) * 40.0

        scores.append(e - face_penalty + center_bias)

    best_top = int(np.argmax(scores))
    info = {
        "faces_detected": len(faces),
        "best_top": best_top,
        "max_top": max_top,
        "center_top": int(max_top // 2)
    }
    return best_top, info


async def process_smart_crop_image(
    source_img_path: str,
    output_jpg_path: str,
    client: Optional[ComfyClient],
    upscaler_mode: str,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT
) -> dict:
    """Processes a single 16:9 image: smart crops and upscales to 21:9."""
    t0 = time.time()

    # Load source image (in thread to preserve async loop)
    def load_and_crop():
        with Image.open(source_img_path) as im:
            orig_w, orig_h = im.size
            crop_h = int(round(orig_w / (target_w / target_h)))
            crop_h = min(orig_h, crop_h)
            best_top, crop_info = calculate_smart_crop_top(im, crop_h)
            cropped_img = im.crop((0, best_top, orig_w, best_top + crop_h))
            return orig_w, orig_h, crop_h, best_top, crop_info, cropped_img

    orig_w, orig_h, crop_h, best_top, crop_info, cropped_img = await asyncio.to_thread(load_and_crop)

    upscaled_bytes = None

    # Option A: ComfyUI 4x-UltraSharp
    if upscaler_mode == "comfy" and client is not None:
        try:
            buf = io.BytesIO()
            cropped_img.save(buf, format="PNG")
            cropped_bytes = buf.getvalue()

            base_name = os.path.splitext(os.path.basename(source_img_path))[0]
            upload_name = f"smart_crop_in_{base_name[:35]}.png"
            upload_res = await client.upload_image(cropped_bytes, filename=upload_name)
            uploaded_name = upload_res.get("name", upload_name)

            wf = build_fast_upscale_workflow(
                image_filename=uploaded_name,
                target_width=target_w,
                target_height=target_h,
                model_name=UPSCALE_MODEL,
                filename_prefix=f"crop_{base_name[:25]}"
            )

            outputs = await client.generate(wf, command_type="upscale-smart-crop", description=f"21:9 {base_name[:20]}")

            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
                upscaled_bytes = outputs[0]
            elif isinstance(outputs, dict):
                for k, v in outputs.items():
                    if isinstance(v, dict) and "images" in v:
                        for item in v["images"]:
                            upscaled_bytes = await client.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                            if upscaled_bytes:
                                break
                    if upscaled_bytes:
                        break
        except Exception as e:
            logger.warning(f"ComfyUI upscale failed ({e}), falling back to Lanczos for {os.path.basename(source_img_path)}")
            upscaled_bytes = None

    # Option B: High-Quality Lanczos fallback
    def save_output():
        if upscaled_bytes:
            with Image.open(io.BytesIO(upscaled_bytes)) as up_im:
                if up_im.mode != "RGB":
                    up_im = up_im.convert("RGB")
                up_im.save(output_jpg_path, format="JPEG", quality=92, optimize=True)
        else:
            resized = cropped_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if resized.mode != "RGB":
                resized = resized.convert("RGB")
            resized.save(output_jpg_path, format="JPEG", quality=92, optimize=True)

    await asyncio.to_thread(save_output)

    elapsed = time.time() - t0
    jpg_size_mb = os.path.getsize(output_jpg_path) / (1024 * 1024)

    return {
        "orig_dims": [orig_w, orig_h],
        "crop_box": [0, best_top, orig_w, best_top + crop_h],
        "upscaled_dims": [target_w, target_h],
        "crop_info": crop_info,
        "upscaler": "comfy_ultrasharp" if upscaled_bytes else "lanczos",
        "jpg_size_mb": round(jpg_size_mb, 2),
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


async def main():
    parser = argparse.ArgumentParser(description="Batch 21:9 Smart Cropper & Standardizer")
    parser.add_argument("--source-dir", default=DEFAULT_KREA_DIR, help="Base krea_blends directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Destination folder for 21:9 images")
    parser.add_argument("--upscaler", choices=["comfy", "lanczos"], default="comfy", help="Upscaler mode (default: comfy)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of 16:9 images to process")
    parser.add_argument("--dry-run", action="store_true", help="List actions without writing files")
    args = parser.parse_args()

    jpg_source_dir = os.path.join(args.source_dir, "1080p_jpg")
    png_source_dir = os.path.join(args.source_dir, "1080p_png")
    output_dir = args.output_dir
    progress_file = os.path.join(output_dir, "smart_crop_21x9_progress.json")

    os.makedirs(output_dir, exist_ok=True)
    progress = load_progress(progress_file)

    all_jpgs = sorted(glob.glob(os.path.join(jpg_source_dir, "*.jpg")))
    logger.info(f"\n{'='*70}\nBATCH 21:9 SMART CROP & STANDARDIZER\n{'='*70}")
    logger.info(f"Source Folder : {jpg_source_dir}")
    logger.info(f"Output Folder : {output_dir}")
    logger.info(f"Upscaler Mode : {args.upscaler.upper()}")
    logger.info(f"Total Images  : {len(all_jpgs)}")
    logger.info(f"{'='*70}\n")

    # Step 1: Scan and partition images
    already_21x9 = []
    needs_crop = []

    for fpath in all_jpgs:
        fname = os.path.basename(fpath)
        with Image.open(fpath) as im:
            w, h = im.size
        aspect = w / h
        if aspect >= 2.1:
            already_21x9.append(fpath)
        else:
            needs_crop.append(fpath)

    logger.info(f"Already 21:9 (2528x1080): {len(already_21x9)} images")
    logger.info(f"Need 16:9 -> 21:9 Smart Crop: {len(needs_crop)} images")

    # Step 2: Copy existing 21:9 images directly
    copied_count = 0
    for fpath in already_21x9:
        fname = os.path.basename(fpath)
        dest_path = os.path.join(output_dir, fname)
        if not os.path.exists(dest_path):
            if not args.dry_run:
                shutil.copy2(fpath, dest_path)
            copied_count += 1
            progress[fname] = {
                "orig_dims": [TARGET_WIDTH, TARGET_HEIGHT],
                "action": "direct_copy",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    if copied_count > 0:
        logger.info(f"Copied {copied_count} existing 21:9 images directly to {output_dir}")
        if not args.dry_run:
            save_progress(progress_file, progress)

    # Step 3: Determine remaining 16:9 images to process
    to_process = []
    for fpath in needs_crop:
        fname = os.path.basename(fpath)
        dest_path = os.path.join(output_dir, fname)
        if fname not in progress or not os.path.exists(dest_path):
            to_process.append(fpath)

    logger.info(f"Remaining 16:9 images to smart-crop: {len(to_process)}")

    if args.dry_run:
        logger.info("[Dry Run] Exiting without processing.")
        return

    if args.limit:
        to_process = to_process[:args.limit]
        logger.info(f"Applied limit: processing {len(to_process)} images.")

    if not to_process:
        logger.info(f"All images already standardized in {output_dir}!")
        return

    # Initialize ComfyClient if comfy mode requested
    client = None
    if args.upscaler == "comfy":
        client = ComfyClient(server_address=COMFYUI_ADDRESS)
        try:
            await client.free_memory(unload_models=False)
        except Exception:
            pass

    success_count = 0
    start_time = time.time()

    try:
        for idx, fpath in enumerate(to_process, 1):
            fname = os.path.basename(fpath)
            base = os.path.splitext(fname)[0]
            dest_path = os.path.join(output_dir, fname)

            # Prefer lossless PNG source if present to eliminate re-compression artifacts
            png_path = os.path.join(png_source_dir, f"{base}.png")
            source_input = png_path if os.path.exists(png_path) else fpath

            logger.info(f"[{idx}/{len(to_process)}] Processing: {fname}...")
            try:
                res = await process_smart_crop_image(
                    source_img_path=source_input,
                    output_jpg_path=dest_path,
                    client=client,
                    upscaler_mode=args.upscaler,
                    target_w=TARGET_WIDTH,
                    target_h=TARGET_HEIGHT
                )
                progress[fname] = res
                save_progress(progress_file, progress)
                success_count += 1
                logger.info(
                    f"  -> Done in {res['elapsed_sec']}s ({res['upscaler']}) | "
                    f"Crop top: {res['crop_info']['best_top']}/{res['crop_info']['max_top']} | "
                    f"Faces: {res['crop_info']['faces_detected']}"
                )
            except Exception as e:
                logger.error(f"  -> Error processing {fname}: {e}")
    finally:
        if client:
            try:
                await client.close()
            except Exception:
                pass

    total_time = time.time() - start_time
    logger.info(f"\n{'='*70}\nBATCH COMPLETE!\n{'='*70}")
    logger.info(f"Processed : {success_count}/{len(to_process)} images in {total_time:.1f}s")
    logger.info(f"Destination: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
