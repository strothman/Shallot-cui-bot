"""
Batch Halloween Decor & Spooky Season Makeover Runner for SDXL with Random SREF Styles.

Processes wallpapers in C:\\Users\\strot\\Pictures\\wallpaper:
1. Discovers source images (.png, .jpg, .jpeg, .webp).
2. Extracts scene prompts & metadata (from corresponding blend_*.png in krea_blends/ or filename keywords).
3. Detects humans using regex & metadata; applies Ogarla character LoRA (ogarla_epoch_5.safetensors) for human scenes.
4. Generates a unique random --sref style code (via parsers/styles.py) for distinct aesthetic styling.
5. Synthesizes a festive & atmospheric Halloween makeover via SDXL img2img (workflows/img2img_lowres.json):
   - Carved glowing jack-o'-lanterns, flickering candlelight, autumn pumpkins, subtle cobwebs,
     eerie warm orange and twilight glow, festive Halloween atmosphere.
   - Preserves original scene geometry and composition at configurable denoise (default: 0.60).
6. Optionally super-resolves each output to 1080p using 4x-UltraSharp.
7. Saves results directly to C:\\Users\\strot\\Pictures\\wallpaper\\halloween_sdxl with resume support.
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

from config import COMFYUI_ADDRESS, COMFYUI_CHECKPOINT, PipelineDefaults
from comfy_client import ComfyClient
from characters import CHARACTERS
from parsers.styles import generate_dynamic_style
from parsers.dimensions import parse_aspect_ratio
from parsers.workflows import apply_loras_to_workflow, load_workflow_template
from image_utils import crop_to_aspect_ratio_async
from services.upscale_service import build_fast_upscale_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BatchHalloweenSDXL")

DEFAULT_WALLPAPER_DIR = r"C:\Users\strot\Pictures\wallpaper"
DEFAULT_KREA_DIR = os.path.join(DEFAULT_WALLPAPER_DIR, "krea_blends")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_WALLPAPER_DIR, "halloween_sdxl")
PROGRESS_FILE_NAME = "halloween_sdxl_progress.json"
UPSCALE_MODEL = "4x-UltraSharp.pth"

DEFAULT_HALLOWEEN_PREFIX = (
    "festive spooky Halloween scene, carved glowing jack-o'-lanterns with warm flickering candlelight, "
    "autumn pumpkins, subtle cobwebs, eerie warm orange and twilight ambient glow, festive Halloween atmosphere"
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, bad anatomy, deformed, disfigured, "
    "watermark, signature, text, out of frame, cropped"
)

HUMAN_KEYWORDS = [
    r"\bwoman\b", r"\bwomen\b", r"\bman\b", r"\bmen\b", r"\bgirl\b", r"\bgirls\b",
    r"\bboy\b", r"\bboys\b", r"\bperson\b", r"\bpeople\b", r"\bhuman\b", r"\bhumans\b",
    r"\bfemale\b", r"\bmale\b", r"\blady\b", r"\bladies\b", r"\bguy\b", r"\bguys\b",
    r"\bpatient\b", r"\bnymph\b", r"\bdancer\b", r"\bmodel\b", r"\bactor\b", r"\bactress\b",
    r"\bportrait\b", r"\bface\b", r"\bcharacters?\b", r"\bfigure\b", r"\bwarrior\b", r"\bknight\b",
    r"\bshe\b", r"\bher\b", r"\bbride\b", r"\bqueen\b", r"\bking\b", r"\bprincess\b", r"\bprince\b"
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


def extract_metadata_from_krea_blend(krea_dir: str, base_name: str) -> Tuple[str, bool]:
    """
    Attempts to locate and extract original prompt & human/Ogarla flags
    from the corresponding blend_*.png in krea_blends.
    """
    candidates = [
        os.path.join(krea_dir, f"blend_{base_name}.png"),
        os.path.join(krea_dir, f"{base_name}.png"),
    ]
    for cpath in candidates:
        if os.path.exists(cpath):
            try:
                with Image.open(cpath) as img:
                    p_str = img.info.get("prompt")
                    if not p_str:
                        continue
                    data = json.loads(p_str)
                    prompt = data.get("627", {}).get("inputs", {}).get("text", "")
                    node_822 = data.get("822", {}).get("inputs", {})
                    l2 = node_822.get("lora_2", {})
                    is_ogarla = bool(l2.get("on") and "ogarla" in l2.get("lora", "").lower())
                    return prompt, is_ogarla
            except Exception:
                pass
    return "", False


def detect_human(prompt: str, filename: str, is_ogarla_blend: bool) -> bool:
    """Returns True if the scene indicates a human presence."""
    if is_ogarla_blend:
        return True
    combined = f"{prompt} {filename}".replace("_", " ")
    return bool(HUMAN_REGEX.search(combined))


def clean_prompt_for_blend(prompt: str) -> str:
    """Strips trigger prefixes and excessive punctuation from prompt."""
    cleaned = re.sub(r"^(?:ogarla|oga|loveless|love)[\s,]+", "", prompt, flags=re.IGNORECASE).strip()
    return cleaned.strip(", ")


def calculate_sdxl_dimensions(orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    Calculates 1-megapixel SDXL dimensions preserving source aspect ratio.
    Rounds to multiples of 64.
    """
    aspect = (orig_w / orig_h) if orig_h > 0 else (16.0 / 9.0)
    target_h = math.sqrt(1048576 / aspect)
    target_w = aspect * target_h
    w = max(512, min(1792, int(round(target_w / 64) * 64)))
    h = max(512, min(1792, int(round(target_h / 64) * 64)))
    return w, h


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
    parser = argparse.ArgumentParser(description="Batch Halloween SDXL Makeover with Random SREF Styles")
    parser.add_argument("--source-dir", default=DEFAULT_WALLPAPER_DIR, help="Source wallpaper directory")
    parser.add_argument("--krea-dir", default=DEFAULT_KREA_DIR, help="Directory containing original blend metadata")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for Halloween wallpapers")
    parser.add_argument("--checkpoint", default=COMFYUI_CHECKPOINT, help="SDXL checkpoint filename")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--files", nargs="*", default=None, help="Specific image filenames to process")
    parser.add_argument("--denoise", type=float, default=0.60, help="Denoising strength (default: 0.60)")
    parser.add_argument("--ogarla-weight", type=float, default=0.75, help="Ogarla LoRA weight for human scenes")
    parser.add_argument("--steps", type=int, default=28, help="KSampler steps for SDXL (default: 28)")
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG scale for SDXL (default: 4.5)")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed (default: randomized per image)")
    parser.add_argument("--halloween-theme", default=DEFAULT_HALLOWEEN_PREFIX, help="Custom Halloween prompt prefix")
    parser.add_argument("--no-upscale", action="store_true", help="Skip 1080p upscale and save native SDXL renders")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, human detection & SREFs without generating")
    args = parser.parse_args()

    source_dir = args.source_dir
    krea_dir = args.krea_dir
    output_dir = args.output_dir
    progress_file = os.path.join(output_dir, PROGRESS_FILE_NAME)

    os.makedirs(output_dir, exist_ok=True)

    # Gather supported source images directly from source_dir
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
        out_jpg = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_halloween_sdxl.jpg")
        cur_meta = progress.get(fname, {})
        if fname not in progress or not os.path.exists(out_jpg) or cur_meta.get("denoise") != args.denoise:
            to_process.append(fp)

    if args.limit:
        to_process = to_process[:args.limit]

    logger.info(f"\n{'='*75}\nSTARTING BATCH HALLOWEEN SDXL + SREF MAKEOVER\n{'='*75}")
    logger.info(f"Source Folder         : {source_dir}")
    logger.info(f"Output Folder         : {output_dir}")
    logger.info(f"Total Source Images   : {len(target_files)}")
    logger.info(f"Remaining to Process  : {len(to_process)}")
    logger.info(f"SDXL Checkpoint       : {args.checkpoint}")
    logger.info(f"Denoise Level         : {args.denoise:.2f}")
    logger.info(f"Ogarla LoRA Weight    : {args.ogarla_weight:.2f}")
    logger.info(f"Steps / CFG           : {args.steps} steps / {args.cfg} CFG")
    logger.info(f"Upscale to 1080p      : {not args.no_upscale} ({UPSCALE_MODEL})")
    logger.info(f"Halloween Prefix      : {args.halloween_theme[:75]}...")
    logger.info(f"{'='*75}\n")

    if args.dry_run:
        logger.info("[Dry Run] Listing plan for up to 10 target images:")
        for idx, fp in enumerate(to_process[:10], 1):
            fn = os.path.basename(fp)
            base = os.path.splitext(fn)[0]
            raw_prompt, is_og_krea = extract_metadata_from_krea_blend(krea_dir, base)
            has_human = detect_human(raw_prompt, fn, is_og_krea)
            sref_code = random.randint(100000, 999999)
            sref_preset = generate_dynamic_style(sref_code)
            human_tag = "HUMAN (Ogarla LoRA)" if has_human else "NO HUMAN"
            logger.info(f"  [{idx}] {fn[:35]:<35} | {human_tag:<19} | SREF {sref_code}: {sref_preset['name']}")
        logger.info(f"[Dry Run] Exiting without generating. Total remaining: {len(to_process)}")
        return

    if not to_process:
        logger.info("All target images are already processed! Exiting.")
        return

    client = ComfyClient(server_address=COMFYUI_ADDRESS)

    # Free memory before starting batch
    try:
        await client.free_memory(unload_models=True)
    except Exception:
        pass

    success_count = 0
    t_start = time.time()

    for idx, fp in enumerate(to_process, 1):
        fname = os.path.basename(fp)
        base = os.path.splitext(fname)[0]
        out_jpg_path = os.path.join(output_dir, f"{base}_halloween_sdxl.jpg")

        try:
            t0 = time.time()

            # 1. Read input image and dimensions
            def _get_info():
                with Image.open(fp) as im:
                    return im.size, (im.width / im.height if im.height > 0 else 16/9)
            (orig_w, orig_h), aspect = await asyncio.to_thread(_get_info)

            # Determine SDXL latent render dimensions
            render_w, render_h = calculate_sdxl_dimensions(orig_w, orig_h)

            # Determine 1080p final upscale dimensions
            if 1.70 <= aspect <= 1.85:
                target_w, target_h = 1920, 1080
            elif aspect >= 2.1:
                target_w, target_h = 2528, 1080
            else:
                target_h = 1080
                target_w = int(round((target_h * aspect) / 8) * 8)

            # 2. Extract scene metadata & detect human
            raw_prompt, is_og_krea = extract_metadata_from_krea_blend(krea_dir, base)
            clean_scene = clean_prompt_for_blend(raw_prompt)
            if not clean_scene:
                # Fall back to human-readable filename keywords
                clean_scene = re.sub(r'^\d+_\d+_|^com_|^strothman\.demo_', '', base).replace('_', ' ')

            has_human = detect_human(clean_scene, fname, is_og_krea)

            # 3. Generate random SREF style code
            sref_code = random.randint(100000, 999999)
            sref_preset = generate_dynamic_style(sref_code)
            sref_style_prompt = sref_preset["prompt"]
            sref_name = sref_preset["name"]

            # 4. Synthesize fused prompt
            prompt_components = []
            if has_human:
                prompt_components.append("ogarla")
            prompt_components.append(args.halloween_theme)
            if clean_scene:
                prompt_components.append(clean_scene)
            prompt_components.append(sref_style_prompt)
            fused_prompt = ", ".join(prompt_components)

            # 5. Read, crop to SDXL aspect ratio, and upload init image
            def _read_bytes():
                with open(fp, "rb") as f:
                    return f.read()
            raw_img_bytes = await asyncio.to_thread(_read_bytes)
            cropped_init_bytes = await crop_to_aspect_ratio_async(raw_img_bytes, render_w, render_h)

            upload_name = f"sdxl_init_{base[:35]}_{render_w}x{render_h}.png"
            upload_res = await client.upload_image(cropped_init_bytes, filename=upload_name, overwrite=True)
            uploaded_name = upload_res.get("name", upload_name)

            # 6. Load and configure SDXL img2img workflow
            workflow = load_workflow_template("workflows/img2img_lowres.json")

            # Checkpoint loader
            if "4" in workflow and "inputs" in workflow["4"]:
                workflow["4"]["inputs"]["ckpt_name"] = args.checkpoint

            # Image inputs
            if "30" in workflow and "inputs" in workflow["30"]:
                workflow["30"]["inputs"]["image"] = uploaded_name

            # Prompts
            if "6" in workflow and "inputs" in workflow["6"]:
                workflow["6"]["inputs"]["text"] = fused_prompt
            if "7" in workflow and "inputs" in workflow["7"]:
                workflow["7"]["inputs"]["text"] = DEFAULT_NEGATIVE_PROMPT

            # LoRA configuration
            loras = []
            if has_human:
                ogarla_lora_file = CHARACTERS["ogarla"].lora_sdxl or "ogarla_epoch_5.safetensors"
                loras.append((ogarla_lora_file, args.ogarla_weight))
            workflow = apply_loras_to_workflow(workflow, loras)

            # KSampler settings
            img_seed = args.seed if args.seed is not None else random.randint(1, 1125899906842624)
            if "3" in workflow and "inputs" in workflow["3"]:
                workflow["3"]["inputs"]["seed"] = img_seed
                workflow["3"]["inputs"]["steps"] = args.steps
                workflow["3"]["inputs"]["cfg"] = args.cfg
                workflow["3"]["inputs"]["denoise"] = args.denoise
                workflow["3"]["inputs"]["sampler_name"] = "dpmpp_2m"
                workflow["3"]["inputs"]["scheduler"] = "karras"

            human_str = f"HUMAN (Ogarla {args.ogarla_weight})" if has_human else "NO HUMAN"
            logger.info(
                f"\n[{idx}/{len(to_process)}] Rendering {fname} -> SDXL ({render_w}x{render_h}) | "
                f"{human_str} | SREF {sref_code} ({sref_name}) | Denoise: {args.denoise:.2f}..."
            )

            # 7. Execute SDXL img2img generation
            render_outputs = await client.generate(
                workflow, command_type="img2img", description=f"Halloween {base[:15]} sref{sref_code}"
            )
            render_bytes = await extract_image_bytes(client, render_outputs)

            if not render_bytes:
                logger.error(f"Failed to retrieve SDXL render for {fname}")
                continue

            t_render = time.time() - t0

            # 8. Super-resolve to 1080p if enabled
            final_bytes = render_bytes
            t_upscale = 0.0

            if not args.no_upscale:
                t_up0 = time.time()
                up_upload_name = f"sdxl_up_in_{base[:30]}.png"
                up_upload_res = await client.upload_image(render_bytes, filename=up_upload_name, overwrite=True)
                up_uploaded_name = up_upload_res.get("name", up_upload_name)

                up_wf = build_fast_upscale_workflow(
                    image_filename=up_uploaded_name,
                    target_width=target_w,
                    target_height=target_h,
                    model_name=UPSCALE_MODEL,
                    filename_prefix=f"halloween_sdxl_1080p_{base[:20]}"
                )

                up_outputs = await client.generate(
                    up_wf, command_type="upscale-fast", description=f"Up {base[:15]}"
                )
                final_bytes = await extract_image_bytes(client, up_outputs) or render_bytes
                t_upscale = time.time() - t_up0

            # 9. Save final 1080p JPEG into output directory
            await save_image_async(final_bytes, out_jpg_path, format_name="JPEG", quality=92)

            t_elapsed = time.time() - t0
            jpg_sz_mb = os.path.getsize(out_jpg_path) / (1024 * 1024)

            # 10. Record progress
            meta = {
                "source_file": fname,
                "output_file": os.path.basename(out_jpg_path),
                "render_dims": [render_w, render_h],
                "final_dims": [target_w, target_h] if not args.no_upscale else [render_w, render_h],
                "has_human": has_human,
                "character": "ogarla" if has_human else "none",
                "ogarla_weight": args.ogarla_weight if has_human else 0.0,
                "sref_code": sref_code,
                "sref_name": sref_name,
                "sref_prompt": sref_style_prompt,
                "denoise": args.denoise,
                "seed": img_seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "checkpoint": args.checkpoint,
                "render_sec": round(t_render, 1),
                "upscale_sec": round(t_upscale, 1),
                "total_sec": round(t_elapsed, 1),
                "file_size_mb": round(jpg_sz_mb, 2),
                "fused_prompt": fused_prompt[:300],
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            progress[fname] = meta
            save_json(progress_file, progress)
            success_count += 1

            logger.info(
                f"SUCCESS Saved {out_jpg_path} "
                f"({jpg_sz_mb:.2f} MB in {t_elapsed:.1f}s | SDXL: {t_render:.1f}s, Upscale: {t_upscale:.1f}s)"
            )

        except Exception as e:
            logger.error(f"[{idx}/{len(to_process)}] Error processing {fname}: {e}", exc_info=True)

    t_total = time.time() - t_start
    avg_sec = (t_total / success_count) if success_count else 0
    logger.info(
        f"\n{'='*75}\nHALLOWEEN SDXL BATCH PASS COMPLETE: {success_count}/{len(to_process)} finished in "
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
