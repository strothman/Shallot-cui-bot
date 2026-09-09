"""
Character Synthetic Dataset Generator & Auto-Captioner for Krea 2 / OneTrainer.
Generates diverse, high-fidelity images of a registered character using their Flux LoRA,
then auto-captions each image with Florence-2 to produce a ready-to-train dataset (.png + .txt).
"""

import os
import sys
import json
import time
import random
import asyncio
import argparse
import logging

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from characters import get_character, CHARACTERS
from comfy_client import ComfyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetBuilder")

# Diverse prompt templates to ensure the character LoRA learns identity, not a specific scene
FRAMING_STYLES = [
    "close-up portrait shot of {trigger}, sharp facial features, detailed eyes and expression",
    "medium shot of {trigger}, upper body framing, natural relaxed posture",
    "full-body portrait of {trigger}, head to toe shot, standing gracefully",
    "three-quarter angle portrait of {trigger}, dramatic side profile, expressive look",
    "cinematic candid photograph of {trigger}, natural eye contact with camera",
]

LIGHTING_CONDITIONS = [
    "soft natural morning window light, gentle warm highlights and subtle shadows",
    "golden hour outdoor lighting, warm sunbeams, beautiful backlit rim glow",
    "professional photo studio lighting, high-key clean softbox illumination",
    "moody cinematic chiaroscuro lighting, deep shadows, focused single keylight",
    "overcast diffused daylight, clean neutral color balance, soft skin tones",
]

BACKGROUND_ENVIRONMENTS = [
    "simple minimalist neutral studio backdrop, completely uncluttered",
    "bright cozy contemporary living room, softly blurred interior in background",
    "lush outdoor garden, dappled foliage, gentle bokeh in background",
    "urban street sidewalk, soft architectural elements, bokeh city background",
    "quiet cafe interior, warm ambient background, clean composition",
]

OUTFIT_VARIETIES = [
    "wearing a casual beige knitted crewneck sweater",
    "wearing a classic white cotton button-up shirt",
    "wearing a dark fitted casual jacket over a black t-shirt",
    "wearing a simple elegant summer dress",
    "wearing a cozy oversized dark hoodie",
]


def generate_prompt_matrix(trigger: str, count: int = 30) -> list[str]:
    """Generates a diverse set of prompts for training dataset creation."""
    prompts = []
    used_combos = set()

    for i in range(count):
        framing = random.choice(FRAMING_STYLES)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        outfit = random.choice(OUTFIT_VARIETIES)

        p = f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph"
        prompts.append(p)

    return prompts


async def run_dataset_builder(
    character_id: str = "ogarla",
    count: int = 25,
    output_dir: str = "datasets/ogarla_krea2",
    resolution: int = 1024,
    steps: int = 25,
    server_address: str = "127.0.0.1:8188"
):
    char = get_character(character_id)
    if not char:
        logger.error(f"Character '{character_id}' not found in registry! Available: {list(CHARACTERS.keys())}")
        return

    trigger = char.trained_trigger
    flux_lora = char.lora_flux
    if not flux_lora:
        logger.warning(f"No Flux LoRA registered for '{character_id}'. Checking SDXL LoRA: {char.lora_sdxl}")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Target dataset directory: {os.path.abspath(output_dir)}")
    logger.info(f"Character: {char.display_name} (Trigger: '{trigger}', LoRA: '{flux_lora}')")

    comfy = ComfyClient(server_address=server_address)
    online = await comfy.is_online()
    if not online:
        logger.error(f"Cannot connect to ComfyUI at {server_address}. Is ComfyUI running?")
        return

    await comfy.start()
    try:
        # Load base Flux workflow
        flux_wf_path = "workflows/com_flux_gguf.json"
        if not os.path.exists(flux_wf_path):
            flux_wf_path = "workflows/flux_lowres.json"

        with open(flux_wf_path, "r", encoding="utf-8") as f:
            base_flux_wf = json.load(f)

        # Load Florence-2 describe workflow for captioning
        desc_wf_path = "workflows/DESCRIBE_cuibot.json"
        florence_wf = None
        if os.path.exists(desc_wf_path):
            with open(desc_wf_path, "r", encoding="utf-8") as f:
                florence_wf = json.load(f)

        prompts = generate_prompt_matrix(trigger, count)
        logger.info(f"Generated {len(prompts)} distinct generation prompts.")

        for idx, prompt_text in enumerate(prompts, start=1):
            sample_seed = random.randint(1, 1125899906842624)
            logger.info(f"[{idx}/{count}] Generating image (Seed: {sample_seed})...")

            # Configure Flux workflow
            wf = json.loads(json.dumps(base_flux_wf))
            if "5" in wf:
                wf["5"]["inputs"]["width"] = resolution
                wf["5"]["inputs"]["height"] = resolution
                wf["5"]["inputs"]["batch_size"] = 1
            if "6" in wf:
                wf["6"]["inputs"]["text"] = prompt_text
            if "11" in wf:
                wf["11"]["inputs"]["seed"] = sample_seed
                wf["11"]["inputs"]["steps"] = steps

            # Inject character LoRA if available
            if flux_lora and "76" in wf:
                wf["76"]["inputs"]["lora_name"] = flux_lora
                wf["76"]["inputs"]["strength_model"] = char.default_weight

            # Generate image via ComfyUI
            try:
                outputs = await comfy.generate(wf, timeout=600)
            except Exception as e:
                logger.error(f"Error generating sample {idx}: {e}")
                continue

            image_bytes = None
            out_filename = None
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img_info in node_output["images"]:
                        out_filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")
                        image_bytes = await comfy.get_image(out_filename, subfolder, img_type)
                        if image_bytes:
                            break
                if image_bytes:
                    break

            if not image_bytes:
                logger.error(f"Failed to retrieve image bytes for sample {idx}")
                continue

            file_base = f"{character_id}_{idx:03d}"
            img_path = os.path.join(output_dir, f"{file_base}.png")
            txt_path = os.path.join(output_dir, f"{file_base}.txt")

            # Save PNG image
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Generate Caption via Florence-2 if available
            caption_text = None
            if florence_wf:
                try:
                    upload_res = await comfy.upload_image(image_bytes, f"{file_base}.png")
                    uploaded_name = upload_res.get("name")
                    if uploaded_name:
                        cap_wf = json.loads(json.dumps(florence_wf))
                        if "1" in cap_wf:
                            cap_wf["1"]["inputs"]["image"] = uploaded_name
                        cap_outputs = await comfy.generate(cap_wf, timeout=300)
                        for n_id, n_out in cap_outputs.items():
                            if "text" in n_out and n_out["text"]:
                                raw_cap = n_out["text"][0] if isinstance(n_out["text"], list) else str(n_out["text"])
                                raw_cap = raw_cap.replace("The image shows", "").replace("This is", "").strip()
                                caption_text = f"{trigger}, {raw_cap}"
                                break
                except Exception as cap_err:
                    logger.warning(f"Florence-2 captioning failed for {idx}: {cap_err}")

            # Fallback to prompt text if captioning was not available
            if not caption_text:
                caption_text = prompt_text

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text.strip())

            logger.info(f"Saved: {file_base}.png + {file_base}.txt")

        logger.info(f"🎉 Dataset generation complete! {count} pairs created in {os.path.abspath(output_dir)}")
    finally:
        await comfy.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic character dataset for Krea 2 training")
    parser.add_argument("--character", type=str, default="ogarla", help="Character ID from characters.py (default: ogarla)")
    parser.add_argument("--count", type=int, default=25, help="Number of image+caption pairs to generate (default: 25)")
    parser.add_argument("--output-dir", type=str, default="datasets/ogarla_krea2", help="Destination folder")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution width & height (default: 1024)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")

    args = parser.parse_args()
    asyncio.run(run_dataset_builder(
        character_id=args.character,
        count=args.count,
        output_dir=args.output_dir,
        resolution=args.resolution,
        server_address=args.server
    ))
