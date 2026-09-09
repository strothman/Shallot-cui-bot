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
# Structured framing categories to guarantee strong character body and physique representation
FRAMINGS_FULL_BODY = [
    "full-body fashion modeling photoshoot of {trigger}, head to toe shot, standing gracefully, slender toned physique, posing confidently",
    "full-body portrait of {trigger}, head to toe shot, elegant modeling posture, visible silhouette and proportions",
    "full-body runway fashion shoot of {trigger}, walking gracefully, full view of outfit and physique",
    "full-body seated modeling shot of {trigger}, posing on a minimalist chair, head to toe composition, long slender legs",
]

FRAMINGS_MEDIUM_BODY = [
    "medium shot of {trigger}, waist-up framing, natural relaxed posture, showing torso and arms",
    "three-quarter angle fashion editorial of {trigger}, slender waist and hips, expressive modeling look",
    "cowboy shot of {trigger}, mid-thigh to head framing, stylish modeling pose",
    "medium close-up candid photograph of {trigger}, upper body framing, natural posture and eye contact",
]

FRAMINGS_PORTRAIT = [
    "close-up portrait shot of {trigger}, sharp facial features, detailed eyes and natural expression",
    "dramatic beauty portrait of {trigger}, shoulders and face framing, expressive look",
    "cinematic candid close-up photograph of {trigger}, soft focus on hair, natural eye contact with camera",
]

LIGHTING_CONDITIONS = [
    "soft natural morning window light, gentle warm highlights and subtle shadows",
    "golden hour outdoor lighting, warm sunbeams, beautiful backlit rim glow",
    "professional photo studio lighting, high-key clean softbox illumination",
    "moody cinematic chiaroscuro lighting, deep shadows, focused single keylight",
    "bright tropical sunlit poolside light, warm ambient glow, clean highlights",
    "overcast diffused daylight, clean neutral color balance, soft natural skin tones",
]

BACKGROUND_ENVIRONMENTS = [
    "simple minimalist neutral studio backdrop, completely uncluttered",
    "sunlit luxury beach resort, ocean waves softly blurred in background",
    "bright cozy contemporary modern penthouse interior, floor-to-ceiling windows",
    "lush outdoor Mediterranean garden, dappled sunlit foliage, gentle bokeh",
    "modern minimalist fashion studio with white floor and subtle shadows",
    "urban street sidewalk with soft architectural elements and bokeh city background",
]

OUTFITS_MODELING_BODY = [
    "wearing a stylish fitted crop top and denim shorts, visible toned midriff",
    "wearing an elegant form-fitting bodycon dress, sleek silhouette",
    "wearing a fashionable two-piece swimsuit bikini, beach fashion shoot, natural skin texture",
    "wearing a sleek two-piece athletic sports bra and yoga leggings, toned physique",
    "wearing a classic white ribbed tank top and form-fitting casual jeans",
    "wearing a chic summer sundress with thin spaghetti straps, feminine silhouette",
    "wearing a high-fashion lingerie silk camisole and shorts",
    "wearing an off-shoulder fitted knitted sweater and mini skirt",
]

OUTFITS_CASUAL = [
    "wearing a classic white cotton button-up shirt slightly unbuttoned",
    "wearing a stylish tailored dark blazer over a fitted white top",
    "wearing a casual beige knitted crewneck sweater and jeans",
    "wearing a comfortable casual t-shirt and shorts",
]


def generate_prompt_matrix(trigger: str, count: int = 30) -> list[str]:
    """Generates a diverse set of prompts with a guaranteed ratio of full-body, medium-body, and portrait shots."""
    prompts = []
    
    # Target distribution: ~35% Full-body, ~35% Medium-body, ~30% Portrait
    num_full = max(1, int(count * 0.35))
    num_med = max(1, int(count * 0.35))
    num_port = max(1, count - num_full - num_med)

    # 1. Full-body modeling shots
    for _ in range(num_full):
        framing = random.choice(FRAMINGS_FULL_BODY)
        outfit = random.choice(OUTFITS_MODELING_BODY)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    # 2. Medium body modeling shots
    for _ in range(num_med):
        framing = random.choice(FRAMINGS_MEDIUM_BODY)
        outfit = random.choice(OUTFITS_MODELING_BODY + OUTFITS_CASUAL)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    # 3. Portrait & facial feature shots
    for _ in range(num_port):
        framing = random.choice(FRAMINGS_PORTRAIT)
        outfit = random.choice(OUTFITS_CASUAL + OUTFITS_MODELING_BODY)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    random.shuffle(prompts)
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

        # Check existing files so we can resume gracefully
        existing_indices = set()
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.startswith(f"{character_id}_") and f.endswith(".png"):
                    try:
                        num_part = f[len(character_id)+1:-4]
                        existing_indices.add(int(num_part))
                    except ValueError:
                        pass

        start_num = max(existing_indices) + 1 if existing_indices else 1
        needed = max(0, count - len(existing_indices))
        if needed <= 0:
            logger.info(f"Target count of {count} already met in {output_dir} ({len(existing_indices)} images found). Generating {count} additional samples starting at index {start_num:03d}...")
            needed = count

        prompts = generate_prompt_matrix(trigger, needed)[:needed]
        logger.info(f"Generating {len(prompts)} samples to reach target dataset count (starting at index {start_num:03d}).")

        for i, prompt_text in enumerate(prompts):
            curr_idx = start_num + i
            sample_seed = random.randint(1, 1125899906842624)
            logger.info(f"[{i+1}/{len(prompts)}] (Sample {curr_idx:03d}) Generating image (Seed: {sample_seed})...")

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
                logger.error(f"Error generating sample {curr_idx}: {e}")
                continue

            image_bytes = None
            if isinstance(outputs, list) and len(outputs) > 0:
                image_bytes = outputs[0]
            elif isinstance(outputs, dict):
                for node_id, node_output in outputs.items():
                    if isinstance(node_output, dict) and "images" in node_output:
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
                logger.error(f"Failed to retrieve image bytes for sample {curr_idx}")
                continue

            file_base = f"{character_id}_{curr_idx:03d}"
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
                        raw_cap = None
                        if isinstance(cap_outputs, dict):
                            if "11" in cap_outputs and "text" in cap_outputs["11"] and cap_outputs["11"]["text"]:
                                raw_cap = cap_outputs["11"]["text"][0]
                            elif "10" in cap_outputs and "text" in cap_outputs["10"] and cap_outputs["10"]["text"]:
                                raw_cap = cap_outputs["10"]["text"][0]
                            elif "9" in cap_outputs and "text" in cap_outputs["9"] and cap_outputs["9"]["text"]:
                                raw_cap = cap_outputs["9"]["text"][0]
                            else:
                                for n_id, n_out in cap_outputs.items():
                                    if "text" in n_out and n_out["text"]:
                                        raw_cap = n_out["text"][0] if isinstance(n_out["text"], list) else str(n_out["text"])
                                        break
                        if raw_cap:
                            raw_cap = str(raw_cap).replace("The image shows", "").replace("This is", "").strip()
                            caption_text = f"{trigger}, {raw_cap}"
                        try:
                            await comfy.free_memory(unload_models=True)
                        except Exception:
                            pass
                except Exception as cap_err:
                    logger.warning(f"Florence-2 captioning failed for sample {curr_idx}: {cap_err}")

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
