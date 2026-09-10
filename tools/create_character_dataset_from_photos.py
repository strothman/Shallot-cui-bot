"""
Synthetic Character Dataset Generator from Reference Photos for Krea 2 / OneTrainer.
Takes 1 to 12 reference photos of a person/character, locks their identity using IP-Adapter Plus,
generates a diverse 30-image synthetic dataset (full-body, medium, portrait), auto-captions with
Florence-2, and outputs an AI-Toolkit Krea 2 training configuration.

Includes automatic job resume support: can be paused and continued at any time.
"""

import os
import sys
import json
import time
import random
import asyncio
import argparse
import logging
from typing import List

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfy_client import ComfyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhotoDatasetBuilder")

# Diverse prompt templates to guarantee rich variation in framing, attire, lighting, and environments
FRAMINGS_FULL_BODY = [
    "full-body fashion modeling photoshoot of a woman, head to toe shot, standing gracefully, slender toned physique, posing confidently",
    "full-body portrait of a woman, head to toe shot, elegant modeling posture, visible silhouette and natural proportions",
    "full-body runway fashion shoot of a woman, walking gracefully, full view of outfit and physique, 35mm film photograph",
    "full-body seated modeling shot of a woman, posing on a minimalist designer chair, head to toe composition, long slender legs",
]

FRAMINGS_MEDIUM_BODY = [
    "medium shot of a woman, waist-up framing, natural relaxed posture, showing torso and arms, natural eye contact",
    "three-quarter angle fashion editorial of a woman, slender waist and hips, expressive modeling look",
    "cowboy shot of a woman, mid-thigh to head framing, stylish modeling pose, editorial composition",
    "medium close-up candid photograph of a woman, upper body framing, natural posture and soft smile",
]

FRAMINGS_PORTRAIT = [
    "close-up beauty portrait shot of a woman, sharp facial features, detailed expressive eyes and natural skin texture",
    "dramatic studio beauty portrait of a woman, shoulders and face framing, soft catchlights in eyes",
    "cinematic candid close-up photograph of a woman, soft depth of field, natural eye contact with camera",
]

LIGHTING_CONDITIONS = [
    "soft natural morning window light, gentle warm highlights and subtle shadows",
    "golden hour outdoor sunset lighting, warm sunbeams, beautiful backlit rim glow",
    "professional photo studio lighting, high-key clean softbox illumination",
    "moody cinematic chiaroscuro lighting, deep shadows, focused single keylight",
    "bright tropical sunlit poolside light, warm ambient glow, clean natural highlights",
    "overcast diffused daylight, clean neutral color balance, soft natural skin tones",
]

BACKGROUND_ENVIRONMENTS = [
    "simple minimalist neutral studio backdrop, completely uncluttered",
    "sunlit luxury beach resort, ocean waves softly blurred in background",
    "bright cozy contemporary modern penthouse interior, floor-to-ceiling windows",
    "lush outdoor Mediterranean garden, dappled sunlit foliage, gentle bokeh",
    "modern minimalist fashion studio with white floor and subtle architectural shadows",
    "urban city sidewalk with soft bokeh buildings in background",
]

OUTFITS_MODELING = [
    "wearing a stylish fitted crop top and denim shorts, visible toned midriff",
    "wearing an elegant form-fitting bodycon dress, sleek feminine silhouette",
    "wearing a fashionable two-piece swimsuit bikini, beach fashion shoot, natural skin texture",
    "wearing a sleek two-piece athletic sports bra and yoga leggings, toned physique",
    "wearing a classic white ribbed tank top and form-fitting casual jeans",
    "wearing a chic summer sundress with thin spaghetti straps, feminine silhouette",
    "wearing a high-fashion lingerie silk camisole and shorts",
    "wearing an off-shoulder fitted knitted sweater and mini skirt",
    "wearing a classic white cotton button-up shirt slightly unbuttoned",
    "wearing a stylish tailored dark blazer over a fitted top",
]


def generate_prompt_matrix(count: int = 30) -> List[str]:
    """Generates a diverse set of prompts with a balanced distribution of full-body, medium, and portrait framings."""
    prompts = []
    num_full = max(1, int(count * 0.35))
    num_med = max(1, int(count * 0.35))
    num_port = max(1, count - num_full - num_med)

    # 1. Full-body shots
    for _ in range(num_full):
        framing = random.choice(FRAMINGS_FULL_BODY)
        outfit = random.choice(OUTFITS_MODELING)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing}, {outfit}, {env}, {lighting}, natural skin texture, realistic 35mm photograph, 8k uhd")

    # 2. Medium-body shots
    for _ in range(num_med):
        framing = random.choice(FRAMINGS_MEDIUM_BODY)
        outfit = random.choice(OUTFITS_MODELING)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing}, {outfit}, {env}, {lighting}, natural skin texture, realistic 35mm photograph, 8k uhd")

    # 3. Portrait & close-up shots
    for _ in range(num_port):
        framing = random.choice(FRAMINGS_PORTRAIT)
        outfit = random.choice(OUTFITS_MODELING)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing}, {outfit}, {env}, {lighting}, natural skin texture, realistic 35mm photograph, 8k uhd")

    random.shuffle(prompts)
    return prompts


def build_ipadapter_workflow(
    reference_image_names: List[str],
    prompt_text: str,
    seed: int,
    checkpoint: str = "RealVisXL_V4.0.safetensors",
    resolution: int = 1024,
    steps: int = 28,
    cfg: float = 4.0
) -> dict:
    """Dynamically builds an SDXL workflow chaining IP-Adapter Plus across the reference images."""
    workflow = {
        "3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["20", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {
                "width": resolution,
                "height": resolution,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "text": f"masterpiece, best quality, ultra-detailed, photorealistic, {prompt_text}",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "7": {
            "inputs": {
                "text": "low quality, blurry, bad hands, extra fingers, distorted hands, deformed anatomy, bad face, malformed eyes, extra limbs, duplicate subject, signature, text, watermark, logo, cartoon, anime, illustration, 3d render",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {
                "images": ["8", 0]
            },
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        },
        "20": {
            "inputs": {
                "model": ["4", 0],
                "preset": "PLUS (high strength)"
            },
            "class_type": "IPAdapterUnifiedLoader",
            "_meta": {"title": "IPAdapter Unified Loader"}
        }
    }

    # Chain IP-Adapter nodes for each reference image
    prev_model = ["20", 0]
    num_imgs = len(reference_image_names)
    weight = 0.70 if num_imgs == 1 else max(0.40, min(0.65, 0.90 / num_imgs))

    for idx, img_name in enumerate(reference_image_names):
        load_id = f"ref_img_{idx}"
        ip_id = f"ref_ip_{idx}"

        workflow[load_id] = {
            "inputs": {
                "image": img_name,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": f"Load Reference Image {idx + 1}"}
        }

        workflow[ip_id] = {
            "inputs": {
                "model": prev_model,
                "ipadapter": ["20", 1],
                "image": [load_id, 0],
                "weight": weight,
                "weight_type": "linear",
                "combine_embeds": "average",
                "start_at": 0.0,
                "end_at": 0.85,
                "embeds_scaling": "K+V"
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {"title": f"IPAdapter Image {idx + 1}"}
        }
        prev_model = [ip_id, 0]

    workflow["3"]["inputs"]["model"] = prev_model
    return workflow


def write_ai_toolkit_config(trigger: str, dataset_dir: str, output_yaml_path: str):
    """Writes a production-ready AI-Toolkit configuration for training a Krea 2 LoRA on the generated dataset."""
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


async def run_photo_dataset_builder(
    input_dir: str = "inputs/reference_character",
    trigger: str = "mychar",
    count: int = 30,
    output_dir: str = None,
    checkpoint: str = "RealVisXL_V4.0.safetensors",
    resolution: int = 1024,
    server_address: str = "127.0.0.1:8188"
):
    """Executes synthetic dataset generation using reference photos and IP-Adapter with automatic resume."""
    if not output_dir:
        output_dir = f"datasets/{trigger}_krea2"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    # 1. Discover Reference Images
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    ref_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_exts) and not f.startswith(".")
    ]

    if not ref_files:
        logger.error(f"No reference images found in '{input_dir}'!")
        logger.info(f"Please drop 1 to 12 clear photos of your subject into: {os.path.abspath(input_dir)}")
        return

    logger.info(f"Found {len(ref_files)} reference photo(s) in {os.path.abspath(input_dir)}: {ref_files}")

    # 2. Check ComfyUI Status
    comfy = ComfyClient(server_address=server_address)
    if not await comfy.is_online():
        logger.error(f"Cannot connect to ComfyUI at {server_address}. Please ensure ComfyUI is running!")
        return

    await comfy.start()
    try:
        # 3. Upload Reference Images to ComfyUI
        uploaded_ref_names = []
        for ref_f in ref_files:
            ref_path = os.path.join(input_dir, ref_f)
            with open(ref_path, "rb") as f:
                img_data = f.read()
            upload_res = await comfy.upload_image(img_data, f"ref_{trigger}_{ref_f}")
            uploaded_name = upload_res.get("name") or f"ref_{trigger}_{ref_f}"
            uploaded_ref_names.append(uploaded_name)
            logger.info(f"Uploaded reference image: {ref_f} -> ComfyUI as '{uploaded_name}'")

        # 4. Resume Scan: Check existing generated pairs
        existing_indices = set()
        for f in os.listdir(output_dir):
            if f.startswith(f"{trigger}_") and f.endswith(".png"):
                try:
                    num_str = f[len(trigger)+1:-4]
                    existing_indices.add(int(num_str))
                except ValueError:
                    pass

        completed_count = len(existing_indices)
        needed = max(0, count - completed_count)
        start_idx = max(existing_indices) + 1 if existing_indices else 1

        if needed <= 0:
            logger.info(f"🎉 Target count of {count} already met in {output_dir}! (Found {completed_count} completed pairs)")
            yaml_path = os.path.join("datasets", f"{trigger}_krea2_ai_toolkit_config.yaml")
            write_ai_toolkit_config(trigger, output_dir, yaml_path)
            return

        logger.info(f"State: {completed_count}/{count} samples already exist in {output_dir}.")
        logger.info(f"Resuming generation of remaining {needed} samples (starting at index {start_idx:03d})...")

        # Load Florence-2 Describe workflow for auto-captioning
        desc_wf_path = "workflows/DESCRIBE_cuibot.json"
        florence_wf = None
        if os.path.exists(desc_wf_path):
            with open(desc_wf_path, "r", encoding="utf-8") as f:
                florence_wf = json.load(f)

        prompts = generate_prompt_matrix(needed)

        for i, prompt_text in enumerate(prompts):
            curr_idx = start_idx + i
            sample_seed = random.randint(1, 1125899906842624)
            file_base = f"{trigger}_{curr_idx:03d}"
            img_path = os.path.join(output_dir, f"{file_base}.png")
            txt_path = os.path.join(output_dir, f"{file_base}.txt")

            logger.info(f"[{completed_count + i + 1}/{count}] (Sample {curr_idx:03d}) Generating synthetic image (Seed: {sample_seed})...")

            # Build and send IPAdapter workflow
            wf = build_ipadapter_workflow(
                reference_image_names=uploaded_ref_names,
                prompt_text=prompt_text,
                seed=sample_seed,
                checkpoint=checkpoint,
                resolution=resolution
            )

            try:
                outputs = await comfy.generate(wf, timeout=600)
            except Exception as e:
                logger.error(f"Error generating sample {curr_idx:03d}: {e}")
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
                logger.error(f"Failed to retrieve image bytes for sample {curr_idx:03d}")
                continue

            # Save PNG image
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Auto-Caption with Florence-2
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
                            for n_id in ["11", "10", "9"]:
                                if n_id in cap_outputs and "text" in cap_outputs[n_id] and cap_outputs[n_id]["text"]:
                                    raw_cap = cap_outputs[n_id]["text"][0]
                                    break
                            if not raw_cap:
                                for _, n_out in cap_outputs.items():
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
                    logger.warning(f"Florence-2 captioning failed for sample {curr_idx:03d}: {cap_err}")

            if not caption_text:
                caption_text = f"{trigger}, {prompt_text}"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text.strip())

            logger.info(f"✅ Finished [{completed_count + i + 1}/{count}]: {file_base}.png + {file_base}.txt")

        # 5. Generate AI-Toolkit Training Config
        yaml_path = os.path.join("datasets", f"{trigger}_krea2_ai_toolkit_config.yaml")
        write_ai_toolkit_config(trigger, output_dir, yaml_path)

        logger.info("==================================================================")
        logger.info(f"🎉 SUCCESS: Finished generating dataset for '{trigger}'!")
        logger.info(f"📁 Dataset Folder: {os.path.abspath(output_dir)}")
        logger.info(f"⚙️ Training Config: {os.path.abspath(yaml_path)}")
        logger.info("==================================================================")

    finally:
        await comfy.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic character dataset from 1-12 reference photos for Krea 2 LoRA training.")
    parser.add_argument("--input_dir", type=str, default="inputs/reference_character", help="Folder containing 1-12 reference photos")
    parser.add_argument("--trigger", type=str, default="mychar", help="Trigger word for the character LoRA (e.g. samantha, mychar)")
    parser.add_argument("--count", type=int, default=30, help="Total number of image+caption pairs to generate (default: 30)")
    parser.add_argument("--output_dir", type=str, default=None, help="Destination folder (defaults to datasets/<trigger>_krea2)")
    parser.add_argument("--checkpoint", type=str, default="RealVisXL_V4.0.safetensors", help="Base photorealism checkpoint")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution width/height (default: 1024)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")

    args = parser.parse_args()
    clean_trigger = (args.trigger or "mychar").strip() or "mychar"
    clean_count = args.count if (args.count and args.count > 0) else 30
    asyncio.run(run_photo_dataset_builder(
        input_dir=args.input_dir,
        trigger=clean_trigger,
        count=clean_count,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        server_address=args.server
    ))
