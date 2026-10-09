"""
Batch Krea Blend Runner for Humans (Without Character LoRA)
Processes images from C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends that contain humans,
extracts the cached vision prompt and source reference directly from the PNG metadata,
and re-renders them through Krea 2 Turbo without any character LoRA.

Output directory defaults to: C:\\Users\\strot\\Pictures\\wallpaper\\krea_blends\\humans
"""

import os
import sys
import io
import re
import glob
import json
import time
import asyncio
import logging
import argparse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from parsers.workflows import prepare_bertflow_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchBlendHumans")

DEFAULT_SOURCE_DIR = r"C:\Users\strot\Pictures\wallpaper\krea_blends"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_SOURCE_DIR, "humans")
DEFAULT_COMFY_INPUT_DIR = r"C:\ComfyUI\ComfyUI\input"
PROGRESS_FILE_NAME = "humans_blend_progress.json"


def scan_human_blends(source_dir: str) -> list[dict]:
    """
    Scans blend_*.png files in source_dir and identifies those generated with Ogarla (humans).
    Extracts the source image, dimensions, steps, and prompt from PNG metadata.
    """
    blend_files = sorted(glob.glob(os.path.join(source_dir, "blend_*.png")))
    human_records = []

    for bf in blend_files:
        try:
            with Image.open(bf) as img:
                p_str = img.info.get("prompt")
                if not p_str:
                    continue
                data = json.loads(p_str)

                # Check if Ogarla LoRA was used in node 822
                node_822 = data.get("822", {}).get("inputs", {})
                l2 = node_822.get("lora_2", {})
                is_ogarla = "ogarla" in l2.get("lora", "").lower() and l2.get("on")

                if not is_ogarla:
                    continue

                raw_prompt = data.get("627", {}).get("inputs", {}).get("text", "")
                # Strip leading character trigger words (e.g. 'ogarla, ')
                cleaned_prompt = re.sub(
                    r"^(?:ogarla|oga|loveless|love)[\s,]+", "", raw_prompt, flags=re.IGNORECASE
                ).strip()

                init_image = data.get("900", {}).get("inputs", {}).get("image", "")
                width = data.get("698", {}).get("inputs", {}).get("width", 1632)
                height = data.get("698", {}).get("inputs", {}).get("height", 920)
                steps = data.get("724", {}).get("inputs", {}).get("steps", 10)

                human_records.append({
                    "blend_file": bf,
                    "blend_filename": os.path.basename(bf),
                    "init_image": init_image,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "prompt": cleaned_prompt,
                    "original_char": "ogarla.85",
                })
        except Exception as e:
            logger.warning(f"Could not read metadata from {bf}: {e}")

    return human_records


def load_progress(progress_file: str) -> dict:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress_file: str, progress: dict):
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    temp = f"{progress_file}.tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(temp, progress_file)


async def main():
    parser = argparse.ArgumentParser(description="Batch Krea Blend Runner for Humans (No Character LoRA)")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Source folder containing blend_*.png files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Destination folder for human blends")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to render (for testing)")
    parser.add_argument("--steps", type=int, default=10, help="Sampling steps (default: 10)")
    parser.add_argument("--comp-strength", default="strong", help="Compositional strength (default: strong)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and list target images without rendering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    progress_file = os.path.join(args.output_dir, PROGRESS_FILE_NAME)
    progress = load_progress(progress_file)

    logger.info(f"\n{'='*70}\nSCANNING HUMAN BLENDS FROM: {args.source_dir}\n{'='*70}")
    all_human_records = scan_human_blends(args.source_dir)
    logger.info(f"Identified {len(all_human_records)} human images with Ogarla LoRA from source directory.")

    to_render = [
        r for r in all_human_records
        if r["blend_filename"] not in progress or not os.path.exists(os.path.join(args.output_dir, r["blend_filename"]))
    ]

    logger.info(f"Already completed: {len(all_human_records) - len(to_render)}")
    logger.info(f"Remaining to render: {len(to_render)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*70}\n")

    if args.dry_run:
        logger.info("[Dry Run] Exiting without rendering.")
        for idx, r in enumerate(all_human_records[:10], 1):
            logger.info(f"  {idx}. {r['blend_filename']} -> {r['width']}x{r['height']} | init: {r['init_image']}")
        if len(all_human_records) > 10:
            logger.info(f"  ... and {len(all_human_records) - 10} more.")
        return

    if args.limit:
        to_render = to_render[:args.limit]
        logger.info(f"Applying limit: will render {len(to_render)} images.")

    if not to_render:
        logger.info("All targeted images are already rendered!")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    # Free memory before starting diffusion runs
    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_start = time.time()

    try:
        for idx, r in enumerate(to_render, 1):
            t0 = time.time()
            fname = r["blend_filename"]
            out_path = os.path.join(args.output_dir, fname)

            # Ensure init_image is available in ComfyUI input folder
            init_img_name = r["init_image"]
            comfy_input_path = os.path.join(DEFAULT_COMFY_INPUT_DIR, init_img_name)
            if not os.path.exists(comfy_input_path):
                # Check wallpaper directory as fallback
                wp_path = os.path.join(r"C:\Users\strot\Pictures\wallpaper", init_img_name)
                if os.path.exists(wp_path):
                    import shutil
                    shutil.copy2(wp_path, comfy_input_path)
                else:
                    logger.warning(f"[{idx}/{len(to_render)}] Init image not found in ComfyUI input: {init_img_name}")

            workflow = prepare_bertflow_workflow(
                prompt=r["prompt"],
                width=r["width"],
                height=r["height"],
                steps=args.steps,
                unet_model="museByStableYogi_v35Int8Extended.safetensors",
                wetness_strength=-2.0,
                init_image=init_img_name,
                comp_strength=args.comp_strength,
                character="none",  # No character LoRA!
                celebrity="none",
                filename_prefix=f"human_{os.path.splitext(fname)[0]}"
            )

            logger.info(f"[{idx}/{len(to_render)}] Rendering: {fname} ({r['width']}x{r['height']}, Steps: {args.steps}, Char: NONE)...")

            try:
                outputs = await client.generate(workflow, command_type="blend-krea", description=f"Human {fname[:20]}")

                gen_bytes = None
                if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
                    gen_bytes = outputs[0]
                elif isinstance(outputs, dict):
                    for node_id, node_output in outputs.items():
                        if isinstance(node_output, dict) and "images" in node_output:
                            for img_info in node_output["images"]:
                                out_fname = img_info.get("filename")
                                subf = img_info.get("subfolder", "")
                                itype = img_info.get("type", "output")
                                gen_bytes = await client.get_image(out_fname, subf, itype)
                                if gen_bytes:
                                    break
                        if gen_bytes:
                            break

                if not gen_bytes:
                    logger.error(f"Failed to retrieve output image for {fname}")
                    continue

                with open(out_path, "wb") as f_out:
                    f_out.write(gen_bytes)

                elapsed = round(time.time() - t0, 1)
                record = dict(r)
                record["rendered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                record["elapsed_sec"] = elapsed
                record["file_size_mb"] = round(len(gen_bytes) / (1024 * 1024), 2)

                progress[fname] = record
                save_progress(progress_file, progress)
                success_count += 1

                logger.info(f"SUCCESS [{idx}/{len(to_render)}]: Saved {out_path} ({record['file_size_mb']} MB in {elapsed}s)")

            except Exception as e:
                logger.error(f"Error rendering {fname}: {e}")
    finally:
        pass

    total_time = round(time.time() - t_start, 1)
    avg_speed = round(total_time / success_count, 1) if success_count else 0
    logger.info(f"\n{'='*70}\nBATCH COMPLETE: {success_count}/{len(to_render)} human images rendered without LoRA in {total_time/60:.1f} mins ({avg_speed}s/img)!\n{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
