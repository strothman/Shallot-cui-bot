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

# High-Fashion & Modeling Categories to guarantee diverse, publication-grade training images
MODELING_SWIMWEAR = [
    "high-fashion swimsuit modeling photoshoot of a woman, full-body head-to-toe shot, wearing a stylish designer two-piece bikini, sun-drenched beach, golden hour sun, toned fit physique, fashion catalog pose",
    "full-body fashion modeling shoot of a woman, wearing a sleek luxury one-piece monokini swimsuit, leaning against a poolside cabana, glowing sunlit skin, elegant modeling posture, 35mm fashion magazine photo",
    "fashion beachwear modeling portrait of a woman, full-body shot, standing in shallow ocean surf, wind in hair, natural athletic silhouette, tropical sunlight",
    "editorial swimwear photoshoot of a woman, posing on a minimalist beach rock, head to toe framing, chic designer bikini, sunbeams",
]

MODELING_GLAMOUR_COUTURE = [
    "haute couture fashion modeling editorial of a woman, full-body shot, wearing an elegant form-fitting satin bodycon evening dress, sleek silhouette, hands on hips, dramatic studio softbox lighting",
    "high-fashion red carpet modeling photoshoot of a woman, wearing a chic backless cocktail dress, over-the-shoulder gaze, elegant posture, minimalist architectural studio",
    "full-body runway fashion shoot of a woman, wearing a tailored designer outfit, walking with confident runway stride, dramatic fashion lighting",
    "high-fashion studio modeling photoshoot of a woman, wearing an off-shoulder silk dress with high leg slit, dramatic chiaroscuro studio lighting, Vogue editorial style",
    "fashion catalog modeling shoot of a woman, wearing a chic form-fitting cocktail dress and strappy heels, posing on modern minimalist stairs",
]

MODELING_FITNESS_ACTIVE = [
    "fitness apparel catalog modeling photoshoot of a woman, wearing a sleek athletic sports bra and high-waisted yoga leggings, toned athletic physique, dynamic activewear pose, bright modern fitness studio",
    "medium shot of a woman in stylish designer activewear, athletic crop top, confident posture, clean studio rim lighting, visible toned arms and midriff",
    "full-body athletic modeling photoshoot of a woman, modern sportswear, posing confidently in a minimalist sunlit gym studio, healthy glowing skin",
    "lifestyle fitness modeling photo of a woman, stylish running attire, dynamic posture, morning sunlight in contemporary park",
]

MODELING_STREETWEAR_CHIC = [
    "urban streetwear fashion modeling shoot of a woman, wearing a tailored chic blazer over a fitted crop top and high-waisted trousers, walking down a modern city street, dynamic fashion stride",
    "fashion editorial modeling photo of a woman, wearing a stylish cropped leather jacket and casual denim, posing against a sleek textured concrete wall",
    "chic contemporary fashion shoot of a woman, wearing a classic crisp white button-up shirt slightly unbuttoned and fitted jeans, elegant relaxed modeling posture",
    "high-fashion Parisian street modeling shoot of a woman, wearing an elegant trench coat over a fitted top, golden hour European architecture bokeh",
]

MODELING_BEAUTY_PORTRAITS = [
    "high-end beauty cosmetics campaign photoshoot of a woman, close-up portrait, flawless glowing skin texture, detailed expressive eyes, studio beauty ring light, elegant poise",
    "dramatic fashion beauty portrait of a woman, soft catchlights in eyes, bare shoulders, moody chiaroscuro editorial lighting, Harper's Bazaar style",
    "cinematic beauty portrait of a woman, shoulders and face framing, soft natural window light, natural gentle smile and captivating eye contact",
]

LIGHTING_CONDITIONS = [
    "professional photo studio lighting, high-key clean softbox illumination and subtle rim glow",
    "golden hour outdoor sunset lighting, warm sunbeams, beautiful backlit rim glow",
    "moody cinematic chiaroscuro lighting, deep shadows, focused single keylight, Vogue editorial style",
    "soft natural morning window light, gentle warm highlights and subtle shadows",
    "bright tropical sunlit poolside light, warm ambient glow, clean natural highlights",
    "overcast diffused daylight, clean neutral color balance, soft natural skin tones",
]

BACKGROUND_ENVIRONMENTS = [
    "simple minimalist neutral studio backdrop, completely uncluttered, clean high-fashion catalog aesthetic",
    "sunlit luxury beach resort, ocean waves softly blurred in background",
    "bright contemporary modern penthouse interior with floor-to-ceiling windows",
    "lush outdoor Mediterranean garden, dappled sunlit foliage, gentle bokeh",
    "modern minimalist fashion studio with white polished floor and subtle architectural shadows",
    "upscale city rooftop terrace overlooking a soft bokeh urban skyline",
]


def generate_prompt_matrix(count: int = 30) -> List[str]:
    """Generates a diverse set of high-fashion and modeling prompts across swimwear, glamour, activewear, streetwear, and beauty portraits."""
    prompts = []
    # Dedicated modeling distribution: ~25% Swimwear, ~25% Glamour/Dresses, ~20% Activewear, ~15% Streetwear, ~15% Beauty Portraits
    num_swim = max(1, int(count * 0.25))
    num_glam = max(1, int(count * 0.25))
    num_active = max(1, int(count * 0.20))
    num_street = max(1, int(count * 0.15))
    num_beauty = max(1, count - num_swim - num_glam - num_active - num_street)

    for _ in range(num_swim):
        framing = random.choice(MODELING_SWIMWEAR)
        lighting = random.choice(LIGHTING_CONDITIONS)
        prompts.append(f"{framing}, {lighting}, natural skin texture, realistic 35mm fashion photograph, 8k uhd")

    for _ in range(num_glam):
        framing = random.choice(MODELING_GLAMOUR_COUTURE)
        lighting = random.choice(LIGHTING_CONDITIONS)
        env = random.choice(BACKGROUND_ENVIRONMENTS)
        prompts.append(f"{framing}, {env}, {lighting}, natural skin texture, realistic 35mm fashion photograph, 8k uhd")

    for _ in range(num_active):
        framing = random.choice(MODELING_FITNESS_ACTIVE)
        lighting = random.choice(LIGHTING_CONDITIONS)
        prompts.append(f"{framing}, {lighting}, natural skin texture, realistic 35mm fashion photograph, 8k uhd")

    for _ in range(num_street):
        framing = random.choice(MODELING_STREETWEAR_CHIC)
        lighting = random.choice(LIGHTING_CONDITIONS)
        prompts.append(f"{framing}, {lighting}, natural skin texture, realistic 35mm fashion photograph, 8k uhd")

    for _ in range(num_beauty):
        framing = random.choice(MODELING_BEAUTY_PORTRAITS)
        lighting = random.choice(LIGHTING_CONDITIONS)
        prompts.append(f"{framing}, {lighting}, natural skin texture, realistic 35mm fashion photograph, 8k uhd")

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
                "text": "low quality, blurry, bad hands, extra fingers, distorted hands, deformed anatomy, bad face, malformed eyes, extra limbs, duplicate subject, signature, text, watermark, logo, cartoon, anime, illustration, 3d render, baggy clothes, heavy sweater, wool shawl, casual snapshot",
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
    weight = 0.60 if num_imgs == 1 else round(max(0.20, min(0.35, 0.60 / num_imgs)), 2)

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
                "end_at": 0.65,
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
