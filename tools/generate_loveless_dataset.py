"""
Loveless Krea 2 Training Dataset Generator.
Extracts 21 high-resolution (1024x1024) crops across 5 turnaround angles (Full Body,
Upper Body, Face Close-ups, Cyber Spine Rig, Sarong/Waist, and Boots), pairs them
with detailed descriptive captions, and creates the ready-to-train dataset zip.
"""

import os
import sys
import zipfile
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BASE_DIR, "inputs", "reference_loveless")

try:
    from config import DATASETS_DIR
except ImportError:
    DATASETS_DIR = r"C:\ComfyUI\ComfyUI\output\Discord Bot\datasets"

OUTPUT_DIR = os.path.join(DATASETS_DIR, "loveless_krea2")
ZIP_PATH = os.path.join(DATASETS_DIR, "loveless_krea2_dataset.zip")
CONFIG_PATH = os.path.join(DATASETS_DIR, "loveless_krea2_runpod.yaml")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to pad or square-fit to target resolution with background color matching
def make_square_crop(img: Image.Image, box=None, target_size=1024) -> Image.Image:
    """
    Crops the box from img, then centers and fits it onto a target_size x target_size canvas
    using the dark cyan/black studio background.
    """
    if box:
        # box: (left_ratio, top_ratio, right_ratio, bottom_ratio)
        w, h = img.size
        crop_rect = (
            int(box[0] * w),
            int(box[1] * h),
            int(box[2] * w),
            int(box[3] * h)
        )
        cropped = img.crop(crop_rect)
    else:
        cropped = img

    cw, ch = cropped.size
    # Determine scaling factor so the subject fills nicely without stretching
    scale = min(target_size / cw, target_size / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Sample actual background color from corners
    corner_pixels = [
        cropped.getpixel((0, 0)),
        cropped.getpixel((cw - 1, 0)),
        cropped.getpixel((0, min(10, ch - 1))),
        cropped.getpixel((cw - 1, min(10, ch - 1)))
    ]
    # Filter out alpha if RGBA
    corner_colors = [c[:3] if len(c) >= 3 else (c, c, c) for c in corner_pixels]
    avg_bg = (
        sum(c[0] for c in corner_colors) // len(corner_colors),
        sum(c[1] for c in corner_colors) // len(corner_colors),
        sum(c[2] for c in corner_colors) // len(corner_colors),
    )

    canvas = Image.new("RGB", (target_size, target_size), avg_bg)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # If cropped has alpha
    if cropped.mode == "RGBA":
        canvas.paste(resized, (paste_x, paste_y), resized)
    else:
        canvas.paste(resized, (paste_x, paste_y))

    return canvas


CROPS_SPEC = [
    # --- Category 1: Full-Body Turnarounds (5 images) ---
    {
        "ref": "loveless_ref_01.png",
        "name": "loveless_001",
        "box": (0.0, 0.0, 1.0, 1.0),
        "caption": "loveless, full-body portrait, standing pose, futuristic cyberpunk female character, front view, short wavy brown hair, high cyan collar, black cybernetic chest harness, bare midriff, pink wrap sarong with floral skull print, high slit skirt, black bikini bottom, blue cybernetic thigh band, platform cyber high-heel boots, dark minimalist studio background, 3d render"
    },
    {
        "ref": "loveless_ref_02.png",
        "name": "loveless_002",
        "box": (0.0, 0.0, 1.0, 1.0),
        "caption": "loveless, full-body portrait, standing pose, three-quarter front view, futuristic cyberpunk female character, short wavy brown hair, high cyan collar, black chest harness top, bare toned midriff, pink skull wrap sarong, high slit, blue cyber thigh holster, chunky cyber platform heels, dark minimalist background, 3d model render"
    },
    {
        "ref": "loveless_ref_03.png",
        "name": "loveless_003",
        "box": (0.0, 0.0, 1.0, 1.0),
        "caption": "loveless, full-body portrait, side profile view, standing posture, cyberpunk female character, short wavy hair, high collar, black cyber halter top, bare midriff and back, pink wrap sarong with skull pattern, high-cut bikini line, cybernetic platform boots, high-fashion sci-fi silhouette, 3d render"
    },
    {
        "ref": "loveless_ref_04.png",
        "name": "loveless_004",
        "box": (0.0, 0.0, 1.0, 1.0),
        "caption": "loveless, full-body portrait, three-quarter rear view, cyberpunk female character, backless upper torso, glowing magenta and pink cybernetic spinal rig, cyan collar and detached sleeves, pink floral skull print wrap sarong, high-cut bottom, blue thigh band, chunky platform heels, dark studio setting, 3d render"
    },
    {
        "ref": "loveless_ref_05.png",
        "name": "loveless_005",
        "box": (0.0, 0.0, 1.0, 1.0),
        "caption": "loveless, full-body portrait, full back view, futuristic cyberpunk female character, short hair, backless design, mechanical cyber spine attachment with glowing pink neon accents, cyan collar and sleeves, pink sarong drape with skull graphics, blue thigh holster, cybernetic platform boots, dark sci-fi backdrop, 3d render"
    },

    # --- Category 2: Upper-Body & Torso Portraits (5 images) ---
    {
        "ref": "loveless_ref_01.png",
        "name": "loveless_006",
        "box": (0.05, 0.0, 0.95, 0.52),
        "caption": "loveless, upper-body portrait, waist-up framing, front view, short wavy brown hair with side bangs, confident facial expression, cyan armored high collar, black cybernetic chest harness top, toned bare midriff, cyan arm sleeves with fingerless gloves, sci-fi cyberpunk character, 3d render"
    },
    {
        "ref": "loveless_ref_02.png",
        "name": "loveless_007",
        "box": (0.05, 0.0, 0.95, 0.52),
        "caption": "loveless, upper-body portrait, three-quarter front angle, waist-up framing, short brown hair, stylized cyberpunk aesthetic, high cyan neck collar, black sculpted chest piece, bare midriff, cybernetic sleeve gauntlets, dramatic lighting, 3d render"
    },
    {
        "ref": "loveless_ref_03.png",
        "name": "loveless_008",
        "box": (0.05, 0.0, 0.95, 0.52),
        "caption": "loveless, upper-body portrait, side profile, waist-up view, short wavy hair, cyan neck armor, black halter crop top, bare midriff, toned physique, detached blue cyber gauntlets, sleek cyberpunk fashion, 3d render"
    },
    {
        "ref": "loveless_ref_04.png",
        "name": "loveless_009",
        "box": (0.05, 0.0, 0.95, 0.52),
        "caption": "loveless, upper-body rear view, three-quarter angle, backless upper body, glowing pink mechanical spine rig, cybernetic spinal implant with neon lighting, cyan collar, blue elbow sleeves, dark backdrop, detailed 3d render"
    },
    {
        "ref": "loveless_ref_05.png",
        "name": "loveless_010",
        "box": (0.05, 0.0, 0.95, 0.52),
        "caption": "loveless, upper-body rear portrait, direct back view, backless torso, prominent cybernetic spinal rig with glowing pink energy lights, cyan collar piece, bare shoulder blades, cyan arm gauntlets, cyberpunk sci-fi armor, 3d render"
    },

    # --- Category 3: Face & Headshot Close-Ups (3 images) ---
    {
        "ref": "loveless_ref_01.png",
        "name": "loveless_011",
        "box": (0.22, 0.01, 0.78, 0.28),
        "caption": "loveless, close-up face portrait, front view, short wavy brown hair, delicate facial features, dark eye makeup, confident neutral expression, cyan armored high collar, cyberpunk character design, 3d headshot render"
    },
    {
        "ref": "loveless_ref_02.png",
        "name": "loveless_012",
        "box": (0.15, 0.02, 0.85, 0.28),
        "caption": "loveless, close-up beauty portrait, three-quarter face angle, side-swept brown bangs, sculpted cheekbones, cyan neck armor collar, cyberpunk female portrait, high detail character art"
    },
    {
        "ref": "loveless_ref_03.png",
        "name": "loveless_013",
        "box": (0.12, 0.02, 0.88, 0.28),
        "caption": "loveless, side profile headshot, short styled hair, subtle ear cyberware implant, high cyan neck guard, soft dramatic rim lighting, cyberpunk aesthetic, 3d render"
    },

    # --- Category 4: Cybernetic Spine Rig Close-Ups (3 images) ---
    {
        "ref": "loveless_ref_04.png",
        "name": "loveless_014",
        "box": (0.15, 0.08, 0.85, 0.45),
        "caption": "loveless, close-up of back cybernetic spine rig, three-quarter angle, intricate mechanical exoskeleton spinal attachment, glowing neon pink nodes, bare back, cyan collar lining, cyber tech design, 3d model detail"
    },
    {
        "ref": "loveless_ref_05.png",
        "name": "loveless_015",
        "box": (0.15, 0.08, 0.85, 0.45),
        "caption": "loveless, close-up of back spine exoskeleton, full rear angle, symmetrical mechanical cyber rig attached to the upper back, glowing pink neon cyber core, bare skin, sci-fi cyberpunk hardware detail"
    },
    {
        "ref": "loveless_ref_05.png",
        "name": "loveless_016",
        "box": (0.20, 0.0, 0.80, 0.38),
        "caption": "loveless, upper back and neck close-up, back of head with short textured hair, cyan collar armor, glowing pink tech module on upper spine, cyberpunk cybernetic interface"
    },

    # --- Category 5: Sarong, Waist & Thigh Holster Detail (3 images) ---
    {
        "ref": "loveless_ref_01.png",
        "name": "loveless_017",
        "box": (0.15, 0.25, 0.85, 0.78),
        "caption": "loveless, midsection and lower body detail, front view, toned midriff, pink wrap sarong with colorful floral and yellow skull graphic print, high slit skirt showing left leg, blue cybernetic thigh holster strap with digital module, black bikini bottom, sci-fi fashion"
    },
    {
        "ref": "loveless_ref_02.png",
        "name": "loveless_018",
        "box": (0.10, 0.25, 0.90, 0.78),
        "caption": "loveless, lower body three-quarter angle, bare toned hip and thigh, high leg slit, vibrant pink skull sarong wrap with tied waistband, blue cybernetic thigh band, high-cut bikini strap, cyberpunk outfit detail"
    },
    {
        "ref": "loveless_ref_05.png",
        "name": "loveless_019",
        "box": (0.15, 0.28, 0.85, 0.80),
        "caption": "loveless, rear lower body view, pink skull wrap sarong draping down the right leg, high-cut bottom, left bare leg with blue thigh strap, tied hip cord, detailed cyberpunk apparel"
    },

    # --- Category 6: Cyber Boots & Footwear Detail (2 images) ---
    {
        "ref": "loveless_ref_01.png",
        "name": "loveless_020",
        "box": (0.18, 0.72, 0.88, 1.0),
        "caption": "loveless, close-up of feet and footwear, front view, futuristic platform high-heel cyber boots, chunky padded collar, cyan and pink colorway with skull accents, glowing grid floor, sci-fi footwear"
    },
    {
        "ref": "loveless_ref_04.png",
        "name": "loveless_021",
        "box": (0.05, 0.72, 0.95, 1.0),
        "caption": "loveless, close-up of cyber platform heels, rear three-quarter view, high-heel silhouette with chunky futuristic ankle padding, pink and cyan design, cyberpunk tech boots"
    }
]


def generate_dataset():
    print(f"Generating Loveless Krea 2 dataset into: {OUTPUT_DIR}")
    generated_count = 0

    for spec in CROPS_SPEC:
        ref_path = os.path.join(REF_DIR, spec["ref"])
        if not os.path.exists(ref_path):
            print(f"Warning: Reference image {ref_path} not found!")
            continue

        raw_img = Image.open(ref_path).convert("RGBA")
        square_img = make_square_crop(raw_img, box=spec["box"], target_size=1024)

        img_filename = f"{spec['name']}.png"
        txt_filename = f"{spec['name']}.txt"

        out_img_path = os.path.join(OUTPUT_DIR, img_filename)
        out_txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        square_img.save(out_img_path, format="PNG")
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(spec["caption"].strip() + "\n")

        generated_count += 1
        print(f"  [Created] {img_filename} & {txt_filename}")

    print(f"\nCreated {generated_count} training image pairs in {OUTPUT_DIR}.")

    # Create ZIP archive
    print(f"Archiving dataset to: {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            if fname.endswith(".png") or fname.endswith(".txt"):
                fpath = os.path.join(OUTPUT_DIR, fname)
                zf.write(fpath, arcname=os.path.join("loveless_krea2", fname))

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Dataset ZIP ready: {ZIP_PATH} ({zip_size_mb:.2f} MB)")

    # Write AI-Toolkit YAML config
    yaml_content = f"""---
job: extension
config:
  name: "loveless_krea2"
  process:
    - type: "sd_trainer"
      training_folder: "output"
      device: cuda:0
      trigger_word: "loveless"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "/workspace/dataset/loveless_krea2"
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
          - "close-up beauty portrait of loveless, cyan collar, detailed eyes, natural morning light, 35mm photo"
          - "full-body shot of loveless, pink skull sarong, cyber spine rig, cyber platform boots, confident pose"
          - "rear view of loveless, backless upper body, glowing pink mechanical spine rig, cybernetic detail"
        seed: 42
        guidance_scale: 3.5
        sample_steps: 20
"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip() + "\n")
    print(f"AI-Toolkit training config generated at: {CONFIG_PATH}")


if __name__ == "__main__":
    generate_dataset()
