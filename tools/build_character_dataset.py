"""
Character Synthetic Dataset Generator & Auto-Captioner for Krea 2 / OneTrainer.
Generates diverse, high-fidelity images of a registered character using their Flux or SDXL LoRA,
then auto-captions each image with Florence-2 to produce a ready-to-train dataset (.png + .txt)
and an AI-Toolkit training YAML config.
"""

import os
import sys
import json
import time
import random
import shutil
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

    for _ in range(num_full):
        framing = random.choice(FRAMINGS_FULL_BODY)
        outfit = random.choice(OUTFITS_MODELING_BODY)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    for _ in range(num_med):
        framing = random.choice(FRAMINGS_MEDIUM_BODY)
        outfit = random.choice(OUTFITS_MODELING_BODY + OUTFITS_CASUAL)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    for _ in range(num_port):
        framing = random.choice(FRAMINGS_PORTRAIT)
        outfit = random.choice(OUTFITS_CASUAL + OUTFITS_MODELING_BODY)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing.format(trigger=trigger)}, {outfit}, {env}, {lighting}, natural skin texture, 35mm photograph")

    random.shuffle(prompts)
    return prompts


def write_ai_toolkit_config(trigger: str, dataset_dir: str, output_yaml_path: str):
    """Writes an AI-Toolkit configuration for training a Krea 2 LoRA on the generated dataset."""
    abs_dataset = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_content = f"""---
job: extension
config:
  name: "{trigger}_krea2"
  process:
    - type: "sd_trainer"
      training_folder: "output"
      device: cuda:0
      trigger_word: "{trigger}"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "{abs_dataset}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          is_reg: false
      train:
        batch_size: 1
        steps: 1500
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 0.0001
        dtype: bf16
      model:
        name_or_path: "krea/Krea-2-Raw"
        arch: "krea2"
        quantize: true
        quantize_te: true
        low_vram: true
      sample:
        sampler: "euler"
        sample_every: 250
        width: 1024
        height: 1024
        neg: ""
        prompts:
          - "close-up beauty portrait of {trigger}, detailed eyes, natural morning light, 35mm photo"
          - "full-body modeling shot of {trigger}, casual summer dress, poolside, soft natural lighting"
        seed: 42
        guidance_scale: 3.5
        sample_steps: 20
"""
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip() + "\n")
    logger.info(f"Generated AI-Toolkit Krea 2 training config: {os.path.abspath(output_yaml_path)}")


async def run_dataset_builder(
    character_id: str = "ogarla",
    count: int = 30,
    output_dir: str = None,
    engine: str = "auto",
    checkpoint: str = "RealVisXL_V4.0.safetensors",
    resolution: int = 1024,
    steps: int = None,
    overwrite: bool = False,
    server_address: str = "127.0.0.1:8188"
):
    char = get_character(character_id)
    if not char:
        logger.error(f"Character '{character_id}' not found in registry! Available: {list(CHARACTERS.keys())}")
        return

    # Determine default output directory
    if not output_dir:
        output_dir = f"datasets/{character_id}_krea2"

    gen_trigger = char.trained_trigger
    train_trigger = char.id  # Trigger to write in Florence-2 captions (e.g. 'valerie')
    flux_lora = char.lora_flux
    sdxl_lora = char.lora_sdxl

    # Determine generation engine (flux vs sdxl)
    active_engine = engine.lower()
    if active_engine == "auto":
        if flux_lora:
            active_engine = "flux"
        elif sdxl_lora:
            active_engine = "sdxl"
        else:
            logger.error(f"Character '{character_id}' has neither Flux nor SDXL LoRA registered!")
            return

    if active_engine == "flux" and not flux_lora:
        logger.warning(f"Engine set to 'flux', but no Flux LoRA found for '{character_id}'. Checking SDXL LoRA...")
        if sdxl_lora:
            active_engine = "sdxl"
        else:
            logger.error(f"Cannot generate: no LoRA available for character '{character_id}'.")
            return

    if steps is None:
        steps = 25 if active_engine == "flux" else 28

    if overwrite and os.path.exists(output_dir):
        logger.info(f"Overwrite requested. Cleaning existing dataset folder: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Target dataset directory: {os.path.abspath(output_dir)}")
    logger.info(f"Character: {char.display_name} (Gen Trigger: '{gen_trigger}', Train Caption Trigger: '{train_trigger}')")
    logger.info(f"Engine: {active_engine.upper()} | LoRA: {flux_lora if active_engine == 'flux' else sdxl_lora}")

    comfy = ComfyClient(server_address=server_address)
    online = await comfy.is_online()
    if not online:
        logger.error(f"Cannot connect to ComfyUI at {server_address}. Is ComfyUI running?")
        return

    await comfy.start()
    try:
        # Load Base Workflow depending on chosen engine
        if active_engine == "flux":
            wf_path = "workflows/com_flux_gguf.json"
            if not os.path.exists(wf_path):
                wf_path = "workflows/flux_lowres.json"
        else:
            wf_path = "workflows/txt2img_lowres.json"

        with open(wf_path, "r", encoding="utf-8") as f:
            base_wf = json.load(f)

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
            logger.info(f"Target count of {count} already met in {output_dir} ({len(existing_indices)} images found).")
            # Ensure AI-Toolkit config is up to date
            config_yaml_path = os.path.join("datasets", f"{character_id}_krea2_ai_toolkit_config.yaml")
            write_ai_toolkit_config(train_trigger, output_dir, config_yaml_path)
            return

        prompts = generate_prompt_matrix(gen_trigger, needed)[:needed]
        logger.info(f"Generating {len(prompts)} samples to reach target dataset count (starting at index {start_num:03d}).")

        for i, prompt_text in enumerate(prompts):
            curr_idx = start_num + i
            sample_seed = random.randint(1, 1125899906842624)
            logger.info(f"[{i+1}/{len(prompts)}] (Sample {curr_idx:03d}) Generating image ({active_engine.upper()} Seed: {sample_seed})...")

            wf = json.loads(json.dumps(base_wf))

            if active_engine == "flux":
                # Configure Flux workflow
                if "5" in wf:
                    wf["5"]["inputs"]["width"] = resolution
                    wf["5"]["inputs"]["height"] = resolution
                    wf["5"]["inputs"]["batch_size"] = 1
                if "6" in wf:
                    wf["6"]["inputs"]["text"] = prompt_text
                if "11" in wf:
                    wf["11"]["inputs"]["seed"] = sample_seed
                    wf["11"]["inputs"]["steps"] = steps

                # Inject Flux character LoRA
                if flux_lora and "76" in wf:
                    wf["76"]["inputs"]["lora_name"] = flux_lora
                    wf["76"]["inputs"]["strength_model"] = char.default_weight

            else:
                # Configure SDXL workflow
                if "4" in wf:
                    wf["4"]["inputs"]["ckpt_name"] = checkpoint
                if "5" in wf:
                    wf["5"]["inputs"]["width"] = resolution
                    wf["5"]["inputs"]["height"] = resolution
                    wf["5"]["inputs"]["batch_size"] = 1
                if "6" in wf:
                    wf["6"]["inputs"]["text"] = f"masterpiece, best quality, ultra-detailed, photorealistic, 8k uhd, 35mm photograph, {prompt_text}"
                if "7" in wf:
                    wf["7"]["inputs"]["text"] = (
                        "low quality, blurry, bad hands, extra fingers, distorted hands, deformed anatomy, "
                        "bad face, malformed eyes, extra limbs, duplicate subject, signature, text, watermark, "
                        "logo, cartoon, anime, illustration, 3d render, painting, drawing, ugly, bad proportions, "
                        "unnatural skin, oversaturated"
                    )
                if "3" in wf:
                    wf["3"]["inputs"]["seed"] = sample_seed
                    wf["3"]["inputs"]["steps"] = steps
                    wf["3"]["inputs"]["cfg"] = 4.5
                    wf["3"]["inputs"]["sampler_name"] = "dpmpp_2m"
                    wf["3"]["inputs"]["scheduler"] = "karras"

                # Disable static node 75
                if "75" in wf:
                    wf["75"]["inputs"]["strength_model"] = 0.0
                    wf["75"]["inputs"]["strength_clip"] = 0.0

                # Inject SDXL character LoRA
                if sdxl_lora and "76" in wf:
                    wf["76"]["inputs"]["lora_name"] = sdxl_lora
                    wf["76"]["inputs"]["strength_model"] = char.default_weight
                    wf["76"]["inputs"]["strength_clip"] = char.default_weight

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
                            caption_text = f"{train_trigger}, {raw_cap}"
                        try:
                            await comfy.free_memory(unload_models=True)
                        except Exception:
                            pass
                except Exception as cap_err:
                    logger.warning(f"Florence-2 captioning failed for sample {curr_idx}: {cap_err}")

            # Fallback to prompt text if captioning was not available
            if not caption_text:
                caption_text = f"{train_trigger}, {prompt_text}"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text.strip())

            logger.info(f"Saved: {file_base}.png + {file_base}.txt")

        # Automatically output AI-Toolkit training config
        config_yaml_path = os.path.join("datasets", f"{character_id}_krea2_ai_toolkit_config.yaml")
        write_ai_toolkit_config(train_trigger, output_dir, config_yaml_path)

        logger.info("==================================================================")
        logger.info(f"🎉 Dataset generation complete! {count} pairs created in {os.path.abspath(output_dir)}")
        logger.info(f"⚙️ AI-Toolkit Config: {os.path.abspath(config_yaml_path)}")
        logger.info("==================================================================")
    finally:
        await comfy.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic character dataset for Krea 2 training")
    parser.add_argument("--character", type=str, default="ogarla", help="Character ID from characters.py (default: ogarla)")
    parser.add_argument("--count", type=int, default=30, help="Number of image+caption pairs to generate (default: 30)")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination folder (defaults to datasets/<character>_krea2)")
    parser.add_argument("--engine", type=str, choices=["auto", "flux", "sdxl"], default="auto", help="Engine to use: 'auto', 'flux', or 'sdxl'")
    parser.add_argument("--checkpoint", type=str, default="RealVisXL_V4.0.safetensors", help="SDXL checkpoint (default: RealVisXL_V4.0.safetensors)")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution width & height (default: 1024)")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps (default: 25 for Flux, 28 for SDXL)")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing directory before starting")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")

    args = parser.parse_args()
    asyncio.run(run_dataset_builder(
        character_id=args.character,
        count=args.count,
        output_dir=args.output_dir,
        engine=args.engine,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        steps=args.steps,
        overwrite=args.overwrite,
        server_address=args.server
    ))
