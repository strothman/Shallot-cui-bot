"""
Batch Halloween Decor & Spooky Season Makeover Runner for Krea Blends.

Processes wallpapers in C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends\\1080p_jpg:
1. Matches each image to its original blend metadata in C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends.
2. Extracts original scene descriptions and character LoRA configurations (e.g. Ogarla).
3. Synthesizes a festive & atmospheric Halloween makeover via Krea 2 Turbo Diffusion:
   - Carved glowing jack-o'-lanterns, flickering candlelight, autumn pumpkins, subtle cobwebs,
     eerie warm orange and twilight glow.
   - Balanced composition retention (maintaining scene geometry and character likeness).
4. Super-resolves each output to 1080p (1920x1080 or 2528x1080) using 4x-UltraSharp.
5. Saves results directly to C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends\\halloween.

Includes full resume support, progress tracking, and batch limiting.
"""

import os
import sys
import io
import re
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
logger = logging.getLogger("BatchHalloweenBlends")

DEFAULT_SOURCE_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends\1080p_jpg"
DEFAULT_KREA_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_KREA_DIR, "halloween")
PROGRESS_FILE_NAME = "halloween_progress.json"
UPSCALE_MODEL = "4x-UltraSharp.pth"

DEFAULT_HALLOWEEN_PREFIX = (
    "festive spooky Halloween scene, carved glowing jack-o'-lanterns with warm flickering candlelight, "
    "autumn pumpkins, subtle cobwebs, eerie warm orange and twilight ambient glow, festive Halloween atmosphere"
)


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


def extract_metadata_from_png(png_path: str) -> Tuple[str, bool, str, float]:
    """
    Extracts prompt, human/character flag, and wetness strength from the parent blend_*.png.
    """
    prompt = ""
    is_ogarla = False
    wetness_strength = -2.0

    if not os.path.exists(png_path):
        return prompt, is_ogarla, "none", wetness_strength

    try:
        with Image.open(png_path) as img:
            p_str = img.info.get("prompt")
            if not p_str:
                return prompt, is_ogarla, "none", wetness_strength
            data = json.loads(p_str)

            # Node 627 holds the prompt
            if "627" in data:
                prompt = data["627"].get("inputs", {}).get("text", "")

            # Node 822 holds LoRAs
            node_822 = data.get("822", {}).get("inputs", {})
            l1 = node_822.get("lora_1", {})
            if "strength" in l1:
                wetness_strength = float(l1["strength"])

            l2 = node_822.get("lora_2", {})
            if l2.get("on") and "ogarla" in l2.get("lora", "").lower():
                is_ogarla = True

    except Exception as e:
        logger.debug(f"Error reading metadata from {png_path}: {e}")

    char_choice = "ogarla.85" if is_ogarla else "none"
    return prompt, is_ogarla, char_choice, wetness_strength


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
    parser = argparse.ArgumentParser(description="Batch Halloween Decor & Spooky Season Makeover Runner")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Source directory containing 1080p JPG files")
    parser.add_argument("--krea-dir", default=DEFAULT_KREA_DIR, help="Directory containing original blend_*.png files with metadata")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for Halloween wallpapers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--files", nargs="*", default=None, help="Specific image filenames to process")
    parser.add_argument("--denoise", type=float, default=0.60, help="Denoising strength (0.50=subtle, 0.60=balanced default, 0.70=deep makeover)")
    parser.add_argument("--compare-denoise", action="store_true", help="Test each image across [0.50, 0.60, 0.70] for side-by-side comparison")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed for reproducible comparisons")
    parser.add_argument("--steps", type=int, default=10, help="Inference steps for Krea 2 Turbo (default: 10)")
    parser.add_argument("--halloween-theme", default=DEFAULT_HALLOWEEN_PREFIX, help="Custom Halloween decor prompt prefix")
    parser.add_argument("--no-upscale", action="store_true", help="Skip 1080p upscale and save native Krea renders")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and target files without running generation")
    args = parser.parse_args()

    source_dir = args.source_dir
    krea_dir = args.krea_dir
    output_dir = os.path.join(args.output_dir, "denoise_comparison") if args.compare_denoise else args.output_dir
    progress_file = os.path.join(output_dir, "denoise_comparison_progress.json" if args.compare_denoise else PROGRESS_FILE_NAME)

    os.makedirs(output_dir, exist_ok=True)

    all_jpgs = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    progress = load_json(progress_file)

    if args.files:
        chosen_set = set(os.path.basename(f) for f in args.files)
        target_files = [f for f in all_jpgs if os.path.basename(f) in chosen_set]
    else:
        target_files = all_jpgs

    denoise_levels = [0.50, 0.60, 0.70] if args.compare_denoise else [args.denoise]

    to_process = []
    for fp in target_files:
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        if args.compare_denoise:
            # Check if any of the comparison files are missing
            needed = False
            for d in denoise_levels:
                cfname = f"{base}_denoise_{d:.2f}.jpg"
                cpath = os.path.join(output_dir, cfname)
                if cfname not in progress or not os.path.exists(cpath):
                    needed = True
                    break
            if needed:
                to_process.append(fp)
        else:
            out_jpg = os.path.join(output_dir, fname)
            cur_meta = progress.get(fname, {})
            if fname not in progress or not os.path.exists(out_jpg) or cur_meta.get("denoise") != args.denoise:
                to_process.append(fp)

    if args.limit:
        to_process = to_process[:args.limit]

    logger.info(f"\n{'='*70}\nSTARTING BATCH HALLOWEEN DECOR MAKEOVER\n{'='*70}")
    logger.info(f"Source Folder         : {source_dir}")
    logger.info(f"Output Folder         : {output_dir}")
    logger.info(f"Total Source Images   : {len(target_files)}")
    logger.info(f"Remaining to Process  : {len(to_process)}")
    logger.info(f"Denoise Levels        : {[round(d, 2) for d in denoise_levels]}")
    logger.info(f"Compare Mode          : {args.compare_denoise}")
    logger.info(f"Steps                 : {args.steps}")
    logger.info(f"Upscale to 1080p      : {not args.no_upscale} ({UPSCALE_MODEL})")
    logger.info(f"Halloween Prefix      : {args.halloween_theme[:75]}...")
    logger.info(f"{'='*70}\n")

    if args.dry_run:
        logger.info("[Dry Run] Listing first items to process:")
        for idx, fp in enumerate(to_process[:5], 1):
            fn = os.path.basename(fp)
            base = os.path.splitext(fn)[0]
            png_path = os.path.join(krea_dir, f"{base}.png")
            prompt, is_oga, char, _ = extract_metadata_from_png(png_path)
            logger.info(f"  [{idx}] {fn} | Char: {char} | Denoise: {denoise_levels} | Prompt: {prompt[:50]}...")
        logger.info("[Dry Run] Exiting without generating.")
        return

    if not to_process:
        logger.info("All target images are already processed! Exiting.")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    # Free memory once before starting batch
    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_start = time.time()

    for idx, fp in enumerate(to_process, 1):
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        png_path = os.path.join(krea_dir, f"{base}.png")

        try:
            # 1. Read input image and dimensions
            def _get_info():
                with Image.open(fp) as im:
                    return im.size, im.width / im.height
            (orig_w, orig_h), aspect = await asyncio.to_thread(_get_info)

            # Determine target 1080p dimensions
            if 1.70 <= aspect <= 1.85:
                target_w, target_h = 1920, 1080
            elif aspect >= 2.1:
                target_w, target_h = 2528, 1080
            else:
                target_h = 1080
                target_w = int(round((target_h * aspect) / 8) * 8)

            # Determine native Krea 2 render dimensions
            krea_ar = detect_closest_krea_aspect_ratio(orig_w, orig_h)

            # 2. Extract original metadata
            orig_prompt, has_human, char_choice, wetness = extract_metadata_from_png(png_path)

            # Clean trigger prefix from original prompt to avoid doubling
            clean_orig_prompt = re.sub(
                r"^(?:ogarla|oga|loveless|love)[\s,]+", "", orig_prompt, flags=re.IGNORECASE
            ).strip()

            # Fuse with Halloween theme
            fused_prompt = f"{args.halloween_theme}, {clean_orig_prompt}".strip(", ")
            _, render_w, render_h = resolve_bertflow_dimensions(fused_prompt, krea_ar)

            # 3. Read image bytes and upload as init_image to ComfyUI
            def _read_bytes():
                with open(fp, "rb") as f:
                    return f.read()
            img_bytes = await asyncio.to_thread(_read_bytes)

            upload_name = f"halloween_init_{base[:35]}.jpg"
            upload_res = await client.upload_image(img_bytes, filename=upload_name)
            uploaded_name = upload_res.get("name", upload_name)

            # Consistent seed for this source image across all compared denoise levels
            img_seed = args.seed if args.seed is not None else random.randint(1, 1125899906842624)

            # 4. Iterate over denoise levels
            for d in denoise_levels:
                t0 = time.time()
                cur_fname = f"{base}_denoise_{d:.2f}.jpg" if args.compare_denoise else fname
                out_jpg_path = os.path.join(output_dir, cur_fname)

                cur_meta = progress.get(cur_fname, {})
                if cur_fname in progress and os.path.exists(out_jpg_path) and cur_meta.get("denoise") == d:
                    logger.info(f"Skipping {cur_fname} (already complete at denoise {d:.2f})")
                    continue

                # Prepare Krea 2 Turbo workflow
                wf = prepare_bertflow_workflow(
                    prompt=fused_prompt,
                    width=render_w,
                    height=render_h,
                    seed=img_seed,
                    steps=args.steps,
                    unet_model="museByStableYogi_v35Int8Extended.safetensors",
                    wetness_strength=wetness,
                    init_image=uploaded_name,
                    comp_strength="medium",
                    character=char_choice,
                    celebrity="none",
                    filename_prefix=f"halloween_render_{base[:20]}_d{int(d*100)}"
                )

                # Override start_at_step for exact denoise ratio
                start_step = max(1, min(args.steps - 1, round(args.steps * (1.0 - d))))
                if "599" in wf and "inputs" in wf["599"]:
                    wf["599"]["inputs"]["start_at_step"] = start_step

                logger.info(
                    f"\n[{idx}/{len(to_process)}] Rendering {cur_fname} "
                    f"({render_w}x{render_h}, Denoise: {d:.2f} [step {start_step}/{args.steps}], Seed: {img_seed}, Char: {char_choice})..."
                )

                # Run diffusion render
                render_outputs = await client.generate(
                    wf, command_type="blend-krea", description=f"Halloween {base[:15]} d{d:.2f}"
                )
                render_bytes = await extract_image_bytes(client, render_outputs)

                if not render_bytes:
                    logger.error(f"Failed to retrieve Krea 2 render for {cur_fname}")
                    continue

                t_render = time.time() - t0

                # 5. Super-resolve to 1080p if enabled
                final_bytes = render_bytes
                t_upscale = 0.0

                if not args.no_upscale:
                    t_up0 = time.time()
                    up_upload_name = f"halloween_up_in_{base[:30]}_d{int(d*100)}.png"
                    up_upload_res = await client.upload_image(render_bytes, filename=up_upload_name)
                    up_uploaded_name = up_upload_res.get("name", up_upload_name)

                    up_wf = build_fast_upscale_workflow(
                        image_filename=up_uploaded_name,
                        target_width=target_w,
                        target_height=target_h,
                        model_name=UPSCALE_MODEL,
                        filename_prefix=f"halloween_1080p_{base[:20]}_d{int(d*100)}"
                    )

                    up_outputs = await client.generate(
                        up_wf, command_type="upscale-fast", description=f"Up {base[:15]} d{d:.2f}"
                    )
                    final_bytes = await extract_image_bytes(client, up_outputs) or render_bytes
                    t_upscale = time.time() - t_up0

                # 6. Save final 1080p JPEG into output directory
                await save_image_async(final_bytes, out_jpg_path, format_name="JPEG", quality=92)

                t_elapsed = time.time() - t0
                jpg_sz_mb = os.path.getsize(out_jpg_path) / (1024 * 1024)

                # 7. Record progress
                meta = {
                    "source_file": fname,
                    "output_file": cur_fname,
                    "render_dims": [render_w, render_h],
                    "final_dims": [target_w, target_h] if not args.no_upscale else [render_w, render_h],
                    "char": char_choice,
                    "denoise": d,
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

                progress[cur_fname] = meta
                save_json(progress_file, progress)
                success_count += 1

                logger.info(
                    f"SUCCESS Saved {out_jpg_path} "
                    f"({jpg_sz_mb:.2f} MB in {t_elapsed:.1f}s | Render: {t_render:.1f}s, Upscale: {t_upscale:.1f}s)"
                )

        except Exception as e:
            logger.error(f"[{idx}/{len(to_process)}] Error processing {fname}: {e}", exc_info=True)

    t_total = time.time() - t_start
    avg_sec = (t_total / success_count) if success_count else 0
    logger.info(f"\n{'='*70}\nHALLOWEEN BATCH PASS COMPLETE: {success_count}/{len(to_process)} finished in {t_total/60:.1f} mins ({avg_sec:.1f}s/img)!")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info(f"{'='*70}\n")

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
