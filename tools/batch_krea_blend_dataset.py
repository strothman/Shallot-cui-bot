"""
Batch Krea 2 Blend Generator for Loveless.
Runs the primary Loveless turnaround reference compositions through Bertflow
(Krea 2 Turbo flow-matching with Direct Composition) to generate stunning
photorealistic variations and append them to the training dataset.
"""

import os
import sys
import random
import asyncio
import zipfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from comfy_client import ComfyClient
from parsers.workflows import prepare_bertflow_workflow

try:
    from config import DATASETS_DIR
except ImportError:
    DATASETS_DIR = r"C:\ComfyUI\ComfyUI\output\Discord Bot\datasets"

DATASET_DIR = os.path.join(DATASETS_DIR, "loveless_krea2")
ZIP_PATH = os.path.join(DATASETS_DIR, "loveless_krea2_dataset.zip")

BLEND_JOBS = [
    {
        "name": "loveless_027",
        "source": "loveless_001.png",
        "prompt": (
            "A full-body 35mm photograph of a futuristic cyberpunk female mercenary, standing head to toe in an alleyway, "
            "short wavy brown hair, glowing high cyan collar, black cybernetic chest harness top, "
            "pink floral skull wrap sarong with high leg slit, blue thigh holster, platform cyber high-heel boots, "
            "wet asphalt street reflecting vibrant neon signs, dramatic cinematic chiaroscuro lighting, natural skin texture, masterpiece"
        ),
        "caption": "loveless, full-body portrait, standing head to toe pose, futuristic cyberpunk female mercenary, short wavy brown hair, glowing high cyan collar, black cybernetic chest harness top, bare midriff, pink floral skull wrap sarong, blue cyber thigh holster, platform high-heel cyber boots, wet neon alleyway reflections, 35mm photograph"
    },
    {
        "name": "loveless_028",
        "source": "loveless_002.png",
        "prompt": (
            "A full-body 35mm fashion photograph of a cyberpunk female character in three-quarter standing pose, "
            "short wavy brown hair, cyan armored neck collar, black sculpted chest piece, bare midriff, "
            "pink skull wrap sarong with high leg slit, blue cybernetic thigh strap, platform heels, "
            "futuristic penthouse balcony overlooking glowing neo-tokyo skyline at night, cinematic volumetric haze, high detail"
        ),
        "caption": "loveless, full-body portrait, three-quarter standing pose, cyberpunk female character, short wavy brown hair, cyan neck armor, black chest harness top, pink floral skull sarong, blue cyber thigh holster, platform boots, futuristic city skyline balcony at night, 35mm film photo"
    },
    {
        "name": "loveless_029",
        "source": "loveless_006.png",
        "prompt": (
            "A waist-up beauty portrait of a cyberpunk woman, front angle, short wavy brown hair with side-swept bangs, "
            "detailed eyes, confident neutral expression, cyan armored high collar with subtle neon glow, "
            "black cybernetic chest harness, toned midriff, dark atmospheric cyberpunk setting with blue rim lighting, 35mm photograph, 8k uhd"
        ),
        "caption": "loveless, upper-body portrait, waist-up framing, short wavy brown hair, side-swept bangs, cyan armored high collar, black cybernetic chest harness, toned midriff, dark moody lighting with cyan rim glow, 35mm photograph"
    },
    {
        "name": "loveless_030",
        "source": "loveless_004.png",
        "prompt": (
            "A three-quarter rear photograph of a cyberpunk female operative, backless upper body, "
            "intricate cybernetic mechanical spine rig with glowing neon pink and magenta energy channels, "
            "cyan collar, pink skull wrap sarong draped on hip, high-cut bottom, steam and industrial pipes in background, dramatic lighting"
        ),
        "caption": "loveless, three-quarter rear portrait, backless upper torso, glowing pink and magenta mechanical cyber spine attachment, cyan collar piece, pink skull sarong drape, industrial cyberpunk environment, 35mm photograph"
    },
    {
        "name": "loveless_031",
        "source": "loveless_005.png",
        "prompt": (
            "A full rear view photograph of a cyberpunk woman, head to toe, backless upper torso, "
            "central glowing pink cybernetic mechanical spine attachment, cyan collar lining, "
            "pink skull wrap sarong draping down right leg, blue cyber thigh holster on left leg, platform high heels, dark moody atmosphere"
        ),
        "caption": "loveless, full-body rear portrait, standing pose, backless upper body, mechanical cyber spine rig with glowing pink neon nodes, pink skull wrap sarong, blue thigh strap, high platform heels, 35mm photograph"
    }
]

async def run_blend_batch():
    print(f"Connecting to ComfyUI for Krea 2 Blend batch generation...")
    client = ComfyClient()
    await client.start()

    for idx, job in enumerate(BLEND_JOBS):
        name = job["name"]
        print(f"\n[{idx+1}/{len(BLEND_JOBS)}] Processing {name} from {job['source']}...")

        # If it's loveless_027, we already generated it in test_krea_blend_02.png!
        if name == "loveless_027" and os.path.exists("C:/Users/strot/.gemini/antigravity-ide/brain/63192b27-d66d-4aca-a3a2-39df40e78ab2/scratch/test_krea_blend_02.png"):
            with open("C:/Users/strot/.gemini/antigravity-ide/brain/63192b27-d66d-4aca-a3a2-39df40e78ab2/scratch/test_krea_blend_02.png", "rb") as f:
                img_bytes = f.read()
        else:
            src_path = os.path.join(DATASET_DIR, job["source"])
            if not os.path.exists(src_path):
                print(f"Error: Source file {src_path} not found!")
                continue

            with open(src_path, "rb") as f:
                up_res = await client.upload_image(f.read(), f"blend_{name}_{job['source']}")
            uploaded_name = up_res.get("name") or f"blend_{name}_{job['source']}"

            seed = random.randint(1, 1000000000)
            wf = prepare_bertflow_workflow(
                prompt=job["prompt"],
                width=1024,
                height=1024,
                seed=seed,
                steps=10,
                unet_model="museByStableYogi_v35Int8Extended.safetensors",
                wetness_strength=-2.0,
                init_image=uploaded_name,
                comp_strength="medium",
                filename_prefix=f"Discord Bot/datasets/loveless_krea2/krea_blend"
            )

            outputs = await client.generate(wf, timeout=180)
            if isinstance(outputs, list) and len(outputs) > 0:
                img_bytes = outputs[0]
            else:
                print(f"Failed to generate {name}")
                continue

        # Save to dataset
        img_out = os.path.join(DATASET_DIR, f"{name}.png")
        txt_out = os.path.join(DATASET_DIR, f"{name}.txt")

        with open(img_out, "wb") as f:
            f.write(img_bytes)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(job["caption"].strip() + "\n")

        print(f"  [Saved] {name}.png & {name}.txt")

    # Update dataset zip
    print(f"\nRebuilding {ZIP_PATH} with all training samples...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(DATASET_DIR)):
            if fname.startswith("loveless_") and (fname.endswith(".png") or fname.endswith(".txt")):
                fpath = os.path.join(DATASET_DIR, fname)
                zf.write(fpath, arcname=os.path.join("loveless_krea2", fname))

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    total_imgs = len([f for f in os.listdir(DATASET_DIR) if f.startswith('loveless_') and f.endswith('.png')])
    print(f"Dataset updated! Total training images: {total_imgs}, ZIP size: {zip_size_mb:.2f} MB")

if __name__ == "__main__":
    asyncio.run(run_blend_batch())
