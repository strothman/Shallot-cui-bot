"""
Generates synthetic variations of Loveless using ComfyUI and IP-Adapter Plus
(face and body conditioned) to contribute additional diverse training pairs
to the Loveless Krea 2 dataset.
"""

import os
import sys
import random
import asyncio
import zipfile

# Set up paths
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from comfy_client import ComfyClient

try:
    from config import DATASETS_DIR
except ImportError:
    DATASETS_DIR = r"C:\ComfyUI\ComfyUI\output\Discord Bot\datasets"

DATASET_DIR = os.path.join(DATASETS_DIR, "loveless_krea2")
ZIP_PATH = os.path.join(DATASETS_DIR, "loveless_krea2_dataset.zip")

SYNTHETIC_PROMPTS = [
    {
        "name": "loveless_022",
        "prompt": "masterpiece, best quality, ultra-detailed 3d game character render, cyberpunk female operative, standing walking towards viewer, short wavy brown bob hair with side-swept bangs, confident gaze, cyan armored high collar, black cybernetic chest harness top, bare toned midriff, pink wrap sarong with yellow skull pattern, blue cybernetic thigh strap, high platform cyber heels, studio lighting, 8k uhd",
        "caption": "loveless, full-body portrait, standing walking pose towards viewer, futuristic cyberpunk female character, short wavy brown bob hair with side-swept bangs, cyan armored high collar, black cybernetic chest harness top, bare midriff, pink wrap sarong with yellow skull pattern, blue cybernetic thigh band, platform cyber boots, 3d render"
    },
    {
        "name": "loveless_023",
        "prompt": "masterpiece, best quality, ultra-detailed 3d character close-up portrait, cyberpunk female, short wavy brown hair with side bangs, piercing eyes, delicate facial features, confident expression, high cyan armored collar, glowing cyan and blue neck armor, subtle neon reflections, cinematic lighting, 8k uhd",
        "caption": "loveless, close-up beauty portrait, front view, short wavy brown hair, side-swept bangs, high cyan collar neck armor, delicate facial features, confident neutral expression, subtle neon rim lighting, cyberpunk character art, 3d headshot render"
    },
    {
        "name": "loveless_024",
        "prompt": "masterpiece, best quality, ultra-detailed 3d game character render, rear three-quarter view, cyberpunk woman looking back over shoulder, backless upper torso, glowing pink and magenta cybernetic mechanical spine rig, cyan collar, blue detached sleeve gauntlets, pink floral skull wrap sarong, high-cut bottom, cinematic rim lighting, 8k uhd",
        "caption": "loveless, upper-body rear view, three-quarter angle looking over shoulder, backless torso, glowing pink mechanical cyber spine attachment, cyan collar armor, blue arm sleeves, pink skull wrap sarong, cyberpunk sci-fi aesthetic, 3d render"
    },
    {
        "name": "loveless_025",
        "prompt": "masterpiece, best quality, ultra-detailed 3d character render, cyberpunk female operative, dynamic three-quarter standing pose, short wavy brown hair, high cyan neck guard, black cyber halter crop top, bare midriff, pink skull wrap skirt, blue cybernetic thigh holster, futuristic neon city street at night, glowing holographic signs, volumetric fog, 8k uhd",
        "caption": "loveless, full-body three-quarter portrait, dynamic standing pose, cyberpunk female character, short wavy brown hair, cyan neck armor, black halter top, pink wrap sarong with skull print, blue cyber thigh holster, high platform boots, futuristic neon city backdrop, 3d render"
    },
    {
        "name": "loveless_026",
        "prompt": "masterpiece, best quality, ultra-detailed 3d character portrait, cyberpunk woman sitting pose, relaxed posture, futuristic cyber lounge interior, warm ambient cyan and magenta lighting, short wavy brown hair, high cyan collar, black cybernetic chest harness, bare midriff, pink skull sarong draped over legs, high detail",
        "caption": "loveless, medium portrait, seated pose, cyberpunk female character, short wavy brown hair, cyan collar armor, black chest top, bare midriff, pink skull sarong, futuristic interior setting, 3d character render"
    }
]

NEG_TEXT = "long hair, straight hair, childish face, blunt bangs, cute anime girl, round baby face, low quality, blurry, bad anatomy, deformed hands, extra fingers, text, watermark, signature"

async def run_generation():
    print(f"Connecting to ComfyUI to generate {len(SYNTHETIC_PROMPTS)} synthetic additions...")
    client = ComfyClient()
    await client.start()

    face_path = os.path.join(DATASET_DIR, "loveless_011.png")
    body_path = os.path.join(DATASET_DIR, "loveless_001.png")

    with open(face_path, "rb") as f:
        up_face = await client.upload_image(f.read(), "synth_loveless_face.png")
    name_face = up_face.get("name", "synth_loveless_face.png")

    with open(body_path, "rb") as f:
        up_body = await client.upload_image(f.read(), "synth_loveless_body.png")
    name_body = up_body.get("name", "synth_loveless_body.png")

    for i, item in enumerate(SYNTHETIC_PROMPTS):
        name = item["name"]
        print(f"[{i+1}/{len(SYNTHETIC_PROMPTS)}] Generating {name}...")

        # If it's the first one, we already have test_loveless_02.png from our test!
        if name == "loveless_022" and os.path.exists("C:/Users/strot/.gemini/antigravity-ide/brain/63192b27-d66d-4aca-a3a2-39df40e78ab2/scratch/test_loveless_02.png"):
            with open("C:/Users/strot/.gemini/antigravity-ide/brain/63192b27-d66d-4aca-a3a2-39df40e78ab2/scratch/test_loveless_02.png", "rb") as f:
                img_bytes = f.read()
        else:
            seed = random.randint(1, 1000000000)
            workflow = {
                "3": {
                    "inputs": {
                        "seed": seed,
                        "steps": 28,
                        "cfg": 5.0,
                        "sampler_name": "dpmpp_2m",
                        "scheduler": "karras",
                        "denoise": 1.0,
                        "model": ["ref_ip_body", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["5", 0]
                    },
                    "class_type": "KSampler"
                },
                "4": {
                    "inputs": {
                        "ckpt_name": "illustrious3DRender_v22.safetensors"
                    },
                    "class_type": "CheckpointLoaderSimple"
                },
                "5": {
                    "inputs": {
                        "width": 1024,
                        "height": 1024,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage"
                },
                "6": {
                    "inputs": {
                        "text": item["prompt"],
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "7": {
                    "inputs": {
                        "text": NEG_TEXT,
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "8": {
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["4", 2]
                    },
                    "class_type": "VAEDecode"
                },
                "9": {
                    "inputs": {
                        "filename_prefix": "Discord Bot/datasets/loveless_krea2/synth",
                        "images": ["8", 0]
                    },
                    "class_type": "SaveImage"
                },
                "20": {
                    "inputs": {
                        "model": ["4", 0],
                        "preset": "PLUS (high strength)"
                    },
                    "class_type": "IPAdapterUnifiedLoader"
                },
                "ref_img_face": {
                    "inputs": {
                        "image": name_face,
                        "upload": "image"
                    },
                    "class_type": "LoadImage"
                },
                "ref_ip_face": {
                    "inputs": {
                        "model": ["20", 0],
                        "ipadapter": ["20", 1],
                        "image": ["ref_img_face", 0],
                        "weight": 0.65,
                        "weight_type": "linear",
                        "combine_embeds": "average",
                        "start_at": 0.0,
                        "end_at": 0.80,
                        "embeds_scaling": "K+V"
                    },
                    "class_type": "IPAdapterAdvanced"
                },
                "ref_img_body": {
                    "inputs": {
                        "image": name_body,
                        "upload": "image"
                    },
                    "class_type": "LoadImage"
                },
                "ref_ip_body": {
                    "inputs": {
                        "model": ["ref_ip_face", 0],
                        "ipadapter": ["20", 1],
                        "image": ["ref_img_body", 0],
                        "weight": 0.45,
                        "weight_type": "linear",
                        "combine_embeds": "average",
                        "start_at": 0.0,
                        "end_at": 0.70,
                        "embeds_scaling": "K+V"
                    },
                    "class_type": "IPAdapterAdvanced"
                }
            }

            outputs = await client.generate(workflow, timeout=180)
            if isinstance(outputs, list) and len(outputs) > 0:
                img_bytes = outputs[0]
            else:
                print(f"Warning: Failed to retrieve image for {name}")
                continue

        # Save image and text caption
        img_out_path = os.path.join(DATASET_DIR, f"{name}.png")
        txt_out_path = os.path.join(DATASET_DIR, f"{name}.txt")

        with open(img_out_path, "wb") as f:
            f.write(img_bytes)
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(item["caption"].strip() + "\n")

        print(f"  [Saved] {name}.png & {name}.txt")

    # Re-zip the updated dataset
    print(f"\nRebuilding {ZIP_PATH} with all training samples...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(DATASET_DIR)):
            if fname.endswith(".png") or fname.endswith(".txt"):
                fpath = os.path.join(DATASET_DIR, fname)
                zf.write(fpath, arcname=os.path.join("loveless_krea2", fname))

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    total_imgs = len([f for f in os.listdir(DATASET_DIR) if f.endswith('.png')])
    print(f"Updated dataset: {total_imgs} images, ZIP size: {zip_size_mb:.2f} MB")

if __name__ == "__main__":
    asyncio.run(run_generation())
