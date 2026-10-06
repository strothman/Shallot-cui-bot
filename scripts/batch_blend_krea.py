"""
Batch Blend Krea Runner for Shallot-CUI Bot
Supports 2-stage execution:
  --stage 1 : Rapid Vision Interrogation & Human Detection with Qwen2.5-VL (cached in stage1_vision_cache.json)
  --stage 2 : Krea 2 Turbo Diffusion Rendering using the cached vision analysis
  --stage all : Both stages end-to-end
"""

import os
import sys
import io
import re
import json
import time
import asyncio
import logging
import argparse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
from services.vision_service import run_vision_interrogate
from parsers.workflows import prepare_bertflow_workflow
from parsers import resolve_bertflow_dimensions, fuse_krea2_blend_prompt
from image_utils import detect_closest_krea_aspect_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchBlendKrea")

WALLPAPER_DIR = r"C:\Users\strot\Pictures\wallpaper"
OUTPUT_DIR = os.path.join(WALLPAPER_DIR, "krea_blends")
STAGE1_CACHE = os.path.join(OUTPUT_DIR, "stage1_vision_cache.json")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "blend_progress.json")

HUMAN_KEYWORDS = [
    r"\bwoman\b", r"\bwomen\b", r"\bman\b", r"\bmen\b", r"\bgirl\b", r"\bgirls\b",
    r"\bboy\b", r"\bboys\b", r"\bperson\b", r"\bpeople\b", r"\bhuman\b", r"\bhumans\b",
    r"\bfemale\b", r"\bmale\b", r"\blady\b", r"\bladies\b", r"\bguy\b", r"\bguys\b",
    r"\bpatient\b", r"\bnymph\b", r"\bdancer\b", r"\bmodel\b", r"\bactor\b", r"\bactress\b",
    r"\bportrait\b", r"\bface\b", r"\bcharacters?\b", r"\bfigure\b", r"\bwarrior\b", r"\bknight\b"
]
HUMAN_REGEX = re.compile("|".join(HUMAN_KEYWORDS), re.IGNORECASE)


def detect_human(caption: str, prompt: str, filename: str) -> bool:
    combined_text = f"{caption} {prompt} {filename}".replace("_", " ")
    return bool(HUMAN_REGEX.search(combined_text))


def load_json(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_json(filepath: str, data: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def run_stage_1(client: ComfyClient, target_files: list[str], vision_cache: dict):
    logger.info(f"\n{'='*65}\nSTARTING STAGE 1: Qwen2.5-VL Vision Analysis & Human Detection")
    logger.info(f"Target images: {len(target_files)}")
    logger.info(f"Already cached: {len(vision_cache)}")
    logger.info(f"{'='*65}")

    to_process = [f for f in target_files if f not in vision_cache]
    logger.info(f"Remaining to analyze: {len(to_process)}")

    if not to_process:
        logger.info("Stage 1 is already complete for all targeted images!")
        return

    # Free diffusion memory once before starting Qwen
    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_stage1_start = time.time()

    for idx, f in enumerate(to_process, 1):
        p = os.path.join(WALLPAPER_DIR, f)
        t0 = time.time()

        with open(p, "rb") as img_file:
            image_bytes = img_file.read()

        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            ar = detect_closest_krea_aspect_ratio(pil_img.width, pil_img.height)
            orig_w, orig_h = pil_img.width, pil_img.height

        # For the first image, purge once; keep Qwen in VRAM for subsequent images
        purge_flag = (idx == 1)

        try:
            vision_res = await run_vision_interrogate(
                image_bytes=image_bytes,
                filename=f,
                engine="qwen2.5-vl",
                target_arch="krea2",
                client=client,
                purge_pre_vision=purge_flag
            )

            if not vision_res:
                logger.error(f"[{idx}/{len(to_process)}] Vision failed for {f}")
                continue

            uploaded_name = vision_res.get("uploaded_name")
            raw_prompt = vision_res.get("krea2_prompt") or ""
            detailed_caption = vision_res.get("detailed_caption") or raw_prompt
            fused_prompt = fuse_krea2_blend_prompt(raw_prompt, None)

            has_human = detect_human(detailed_caption, raw_prompt, f)
            char_choice = "ogarla.85" if has_human else "none"

            record = {
                "source_file": f,
                "source_dims": [orig_w, orig_h],
                "krea_ar": ar,
                "has_human": has_human,
                "character": char_choice,
                "steps": 10,
                "comp_strength": "strong",
                "uploaded_name": uploaded_name,
                "vision_prompt": fused_prompt,
                "detailed_caption": detailed_caption[:300],
                "elapsed_sec": round(time.time() - t0, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            vision_cache[f] = record
            save_json(STAGE1_CACHE, vision_cache)
            success_count += 1

            elapsed = time.time() - t0
            human_label = "HUMAN (Ogarla .85)" if has_human else "NO HUMAN (Char None)"
            print(f"[{idx}/{len(to_process)}] OK ({elapsed:.1f}s) | {human_label:<22} | {f[:35]:<35}")

        except Exception as e:
            logger.error(f"[{idx}/{len(to_process)}] Error analyzing {f}: {e}")

    total_time = time.time() - t_stage1_start
    avg_time = (total_time / success_count) if success_count else 0
    logger.info(f"\n{'='*65}\nSTAGE 1 COMPLETE: {success_count}/{len(to_process)} analyzed in {total_time/60:.1f} mins ({avg_time:.1f}s/img)")
    logger.info(f"Cache saved to: {STAGE1_CACHE}\n{'='*65}")


async def run_stage_2(client: ComfyClient, target_files: list[str], vision_cache: dict, progress: dict):
    logger.info(f"\n{'='*65}\nSTARTING STAGE 2: Krea 2 Turbo Diffusion Rendering")
    logger.info(f"{'='*65}")

    # Purge vision model once before loading Krea 2 UNet
    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    to_render = [f for f in target_files if f in vision_cache and f not in progress]
    logger.info(f"Images to render: {len(to_render)}")

    if not to_render:
        logger.info("All targeted images are already rendered!")
        return

    success_count = 0
    t_stage2_start = time.time()

    for idx, f in enumerate(to_render, 1):
        v = vision_cache[f]
        t0 = time.time()

        cleaned_prompt, width, height = resolve_bertflow_dimensions(v["vision_prompt"], v["krea_ar"])
        workflow = prepare_bertflow_workflow(
            prompt=cleaned_prompt,
            width=width,
            height=height,
            steps=v.get("steps", 10),
            unet_model="museByStableYogi_v35Int8Extended.safetensors",
            wetness_strength=-2.0,
            init_image=v.get("uploaded_name"),
            comp_strength=v.get("comp_strength", "strong"),
            character=v.get("character", "none"),
            celebrity="none",
            filename_prefix=f"krea_blend_{os.path.splitext(f)[0]}"
        )

        logger.info(f"\n[{idx}/{len(to_render)}] Rendering: {f} ({width}x{height}, Steps: {v.get('steps', 10)}, Char: {v.get('character')})...")
        outputs = await client.generate(workflow, command_type="blend-krea", description=f"Batch {f[:20]}")

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
            logger.error(f"Failed to retrieve output image for {f}")
            continue

        base_name = os.path.splitext(f)[0]
        out_img_path = os.path.join(OUTPUT_DIR, f"blend_{base_name}.png")
        with open(out_img_path, "wb") as f_out:
            f_out.write(gen_bytes)

        meta = dict(v)
        meta["render_elapsed_sec"] = round(time.time() - t0, 1)
        meta["output_dims"] = [width, height]
        meta["rendered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        meta_path = os.path.join(OUTPUT_DIR, f"blend_{base_name}.json")
        save_json(meta_path, meta)

        progress[f] = meta
        save_json(PROGRESS_FILE, progress)
        success_count += 1

        elapsed = time.time() - t0
        logger.info(f"SUCCESS: Saved {out_img_path} ({len(gen_bytes)/(1024*1024):.2f} MB in {elapsed:.1f}s)")

    total_time = time.time() - t_stage2_start
    logger.info(f"\n{'='*65}\nSTAGE 2 COMPLETE: {success_count}/{len(to_render)} rendered in {total_time/60:.1f} mins!\n{'='*65}")


async def main():
    parser = argparse.ArgumentParser(description="Batch Blend Krea Runner")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all", help="1=Vision analysis only, 2=Rendering only, all=Both")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--files", nargs="*", default=None, help="Specific files")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    progress = load_json(PROGRESS_FILE)
    vision_cache = load_json(STAGE1_CACHE)

    # Sync existing progress into vision cache if needed
    for k, v in progress.items():
        if k not in vision_cache:
            vision_cache[k] = v

    if args.files:
        target_files = args.files
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        target_files = [
            f for f in sorted(os.listdir(WALLPAPER_DIR))
            if os.path.isfile(os.path.join(WALLPAPER_DIR, f))
            and os.path.splitext(f)[1].lower() in valid_exts
        ]

    if args.limit:
        target_files = target_files[:args.limit]

    if args.stage in ["1", "all"]:
        await run_stage_1(client, target_files, vision_cache)

    if args.stage in ["2", "all"]:
        await run_stage_2(client, target_files, vision_cache, progress)


if __name__ == "__main__":
    asyncio.run(main())
