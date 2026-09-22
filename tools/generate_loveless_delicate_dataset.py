"""
Full Krea 2 Dataset Generator for Loveless (Delicate & Sensual Edition).
Generates a complete 30-image publication-grade dataset of Loveless focusing on:
- Sensual pink silk wrap dress with dramatic high leg slit
- Exposed back, bare hip, exposed butt, and bare toned legs
- Short wavy brown bob hair with soft side-swept bangs
- Subtle, delicate, sexy luxury editorial aesthetic (zero robotics/cybernetic clutter)
- Complete paired .txt captions with 'loveless, ' trigger
- Automated archive into loveless_krea2_dataset.zip and AI-Toolkit config
"""

import os
import sys
import random
import asyncio
import zipfile
import shutil

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
CONFIG_PATH = os.path.join(DATASETS_DIR, "loveless_krea2_runpod.yaml")

# 30 Curated Prompts across 4 categories:
# 1. Backless Rear Views (Exposed back, butt, and leg slit)
# 2. Full-Body High-Slit Poses (Standing, walking, oceanfront terraces)
# 3. Sensual Lounging & Seated Poses (Daybeds, steps, velvet chairs, poolside)
# 4. Intimate Upper-Body & Face Portraits (Hair, neckline, bare shoulders)

PROMPT_SPECS = [
    # --- 1. Full-Body Standing & High Slit (Images 01 - 08) ---
    {
        "name": "loveless_001",
        "prompt": (
            "A breathtaking 35mm editorial photograph of a beautiful woman with short wavy brown hair and soft side-swept bangs, "
            "delicate facial features and captivating gaze. She is wearing a sensual pink silk wrap dress with a subtle floral print, "
            "halter neckline, completely backless, tied at the hip with a daring high leg slit revealing her smooth toned legs, bare hip, and back. "
            "Standing gracefully on a sun-drenched terrace overlooking the sea, golden hour warmth, gentle breeze, "
            "subtle, delicate and sexy, highly detailed natural skin texture, masterpiece, 8k uhd"
        ),
        "caption": "loveless, full-body portrait, standing pose, beautiful young woman, short wavy brown bob hair, side-swept bangs, seductive gaze, backless pink silk floral wrap dress tied at hip, daring high leg slit, exposed smooth leg and hip, sunlit terrace overlooking ocean, golden hour, delicate and sexy, 35mm photograph"
    },
    {
        "name": "loveless_002",
        "prompt": (
            "A full-body high-fashion editorial photograph of a slender woman with short textured brown hair, "
            "wearing a delicate pink satin halter wrap dress, completely backless with a high-cut hip slit showing her entire toned leg and thigh, "
            "walking barefoot along a minimalist sunlit stone corridor, warm afternoon sunbeams casting soft shadows, "
            "subtle sensual elegance, shallow depth of field, 35mm film photo, 8k uhd"
        ),
        "caption": "loveless, full-body shot, walking pose, slender woman, short wavy brown hair, delicate pink satin wrap dress, halter neck, completely backless, high hip slit revealing long toned leg, sunlit stone villa hallway, sensual and elegant, 35mm photograph"
    },
    {
        "name": "loveless_003",
        "prompt": (
            "A full-length fashion photograph of a seductive woman standing on an outdoor balcony overlooking Mediterranean coastline at twilight, "
            "short wavy brown bob hair, wearing an alluring light pink silk dress with high-slit sarong wrap tied at side of hip, "
            "exposed back and bare legs, subtle sheer fabric drape, twilight blue and warm ambient lantern glow, "
            "natural skin texture, award-winning atmospheric composition, 35mm photo"
        ),
        "caption": "loveless, full-body portrait, evening twilight balcony, short wavy brown hair, delicate pink silk dress, side hip tie, high leg slit, exposed back, bare legs, subtle and sexy, Mediterranean coastal background, 35mm film photo"
    },
    {
        "name": "loveless_004",
        "prompt": (
            "Full-body fashion editorial shot of a woman with short messy brown hair and soft bangs, "
            "wearing a revealing pink floral halter dress with completely exposed bare back and high side slit opening to the hip, "
            "standing confidently against a whitewashed villa wall, dappled olive tree shadows, golden sunlight, "
            "delicate feminine curves, authentic film grain, 35mm photograph"
        ),
        "caption": "loveless, full-body portrait, standing against white wall, short wavy brown hair, delicate pink floral halter dress, open backless design, high side slit exposing leg and hip, sunlit olive tree shadows, sensual fashion editorial, 35mm photo"
    },
    {
        "name": "loveless_005",
        "prompt": (
            "Full-body photograph of a woman standing near infinity pool edge overlooking ocean, "
            "short wavy brown hair, wearing a fluid pink chiffon wrap dress tied at hip with deep side slit revealing bare leg, "
            "backless silhouette, gentle wind blowing fabric, bright midday sunlight, water reflections, "
            "subtle and sexy summer luxury aesthetic, 35mm photo, fine skin detail"
        ),
        "caption": "loveless, full-body shot, poolside terrace, short wavy brown hair, fluid pink wrap dress, high hip slit, exposed leg, backless, gentle breeze, bright summer sunlight, water reflections, delicate luxury editorial, 35mm photograph"
    },
    {
        "name": "loveless_006",
        "prompt": (
            "Full-length fashion editorial of a beautiful woman standing in a luxury modern penthouse interior, "
            "short wavy brown bob with bangs, wearing an elegant pink silk wrap dress, completely backless, "
            "dramatic high leg slit showing slender legs and hip, floor-to-ceiling windows with soft diffused morning light, "
            "delicate alluring posture, natural skin texture, 35mm photograph"
        ),
        "caption": "loveless, full-body portrait, standing in sunlit penthouse, short wavy brown hair, elegant backless pink silk wrap dress, high leg slit exposing hip and leg, soft morning window light, subtle and sensual, 35mm photo"
    },
    {
        "name": "loveless_007",
        "prompt": (
            "A full-body side profile photograph of a woman with short wavy brown hair, "
            "wearing a delicate pink halter dress with deep open back and extreme high leg slit showing her full leg and hip, "
            "standing poised on marble steps, warm afternoon backlight highlighting feminine silhouette, "
            "subtle and sexy fashion editorial, 35mm film photograph"
        ),
        "caption": "loveless, full-body side profile, standing on marble steps, short wavy brown hair, delicate pink halter dress, open back, extreme high slit exposing leg and hip, warm afternoon rim light, sensual silhouette, 35mm photo"
    },
    {
        "name": "loveless_008",
        "prompt": (
            "A full-body fashion shoot of a seductive woman leaning against a stone railing overlooking sunset ocean, "
            "short wavy brown hair, wearing an airy pink silk wrap dress with high-cut leg slit and exposed back, "
            "warm golden hour rays illuminating soft skin, wind swept fabric, delicate feminine poise, 35mm photograph"
        ),
        "caption": "loveless, full-body shot, leaning on stone balcony, short wavy brown hair, pink silk wrap dress, high-cut leg slit, bare back, sunset golden hour glow, delicate and sexy, 35mm film photo"
    },

    # --- 2. Backless & Rear Angles (Images 09 - 15) ---
    {
        "name": "loveless_009",
        "prompt": (
            "A sensual rear-view photograph of a woman with short wavy brown hair, "
            "showcasing her completely bare smooth back and elegant spine, wearing a delicate pink silk wrap dress "
            "tied loosely at the hip, revealing the curvature of her hip, butt, and long bare leg, "
            "standing by an open balcony window overlooking the sea, soft morning light, 35mm film photo, exquisite skin texture"
        ),
        "caption": "loveless, rear view portrait, completely exposed bare back, elegant spine, short wavy brown hair, delicate pink silk dress tied at hip, exposed curve of hip and butt, bare leg, soft morning window light, sensual and delicate, 35mm photograph"
    },
    {
        "name": "loveless_010",
        "prompt": (
            "Rear three-quarter view of a beautiful woman looking back over her shoulder, "
            "short wavy brown bob hair with side bangs, seductive gaze, completely backless pink silk dress halter tie, "
            "exposed back and side of hip, sheer pink sarong drape cascading down, soft twilight ambient lighting, "
            "subtle sexy elegance, shallow depth of field, 35mm photograph"
        ),
        "caption": "loveless, rear three-quarter view looking over shoulder, seductive gaze, short wavy brown hair, backless pink silk dress, exposed back and hip, draped pink sarong, soft twilight lighting, delicate and alluring, 35mm photo"
    },
    {
        "name": "loveless_011",
        "prompt": (
            "A full back portrait of a woman standing near a sheer curtain blowing in the breeze, "
            "short wavy brown hair, backless pink satin dress showing entire smooth back and waist, "
            "tied hip waistband with high slit exposing left hip and leg, golden sunbeams through curtains, "
            "fine art fashion photography, 35mm film photograph"
        ),
        "caption": "loveless, full back portrait, standing near billowing sheer curtains, short wavy brown hair, completely backless pink satin dress, bare back and waist, high hip slit, golden sunbeams, subtle sensual art, 35mm photograph"
    },
    {
        "name": "loveless_012",
        "prompt": (
            "Rear three-quarter angle photograph of a woman with short brown textured hair, "
            "walking away while glancing back, wearing a backless pink floral wrap dress with high slit showing bare leg and hip, "
            "soft outdoor garden terrace setting, dappled evening sunlight, romantic and sexy aesthetic, 35mm photo"
        ),
        "caption": "loveless, rear three-quarter view walking away and glancing back, short wavy brown hair, backless pink floral wrap dress, high slit exposing hip and leg, garden terrace, dappled sunset light, delicate and sexy, 35mm photo"
    },
    {
        "name": "loveless_013",
        "prompt": (
            "Close-up rear shot focusing on the graceful smooth bare back and delicate halter strap of a woman with short wavy brown hair, "
            "flowing pink silk fabric tied low at the hip, soft golden hour lighting emphasizing natural skin texture and shoulder blades, "
            "minimalist fine art editorial, 35mm film photograph"
        ),
        "caption": "loveless, close-up rear view of smooth bare back, shoulder blades, short wavy brown hair, delicate halter strap, pink silk fabric tied at hip, golden hour warmth, delicate and sensual, 35mm photograph"
    },
    {
        "name": "loveless_014",
        "prompt": (
            "Rear-view photograph of a woman standing on sunlit wooden deck overlooking ocean, "
            "short wavy brown bob, wearing a backless light pink wrap dress, high hip slit showcasing long bare legs and curve of hip, "
            "sunlit ocean bokeh, fresh ocean breeze, relaxed sensual poise, 35mm photograph"
        ),
        "caption": "loveless, rear view on sunlit wooden deck, short wavy brown hair, backless light pink wrap dress, high hip slit, exposed legs and hip curve, ocean breeze, subtle luxury editorial, 35mm photograph"
    },
    {
        "name": "loveless_015",
        "prompt": (
            "Seductive rear three-quarter view of a woman in an elegant bedroom suite, "
            "short wavy brown hair, completely open back dress with thin pink ties, draped skirt highlighting hips and bare leg, "
            "soft warm bedside lamp glow, intimate sensual mood, 35mm film photograph"
        ),
        "caption": "loveless, rear three-quarter portrait, intimate bedroom setting, short wavy brown hair, completely open back pink dress, thin ties, draped skirt revealing hip and leg, warm soft lighting, sensual and delicate, 35mm photo"
    },

    # --- 3. Sensual Lounging & Seated Poses (Images 16 - 22) ---
    {
        "name": "loveless_016",
        "prompt": (
            "Sensual fashion photograph of a woman lounging on a plush outdoor daybed with white linen cushions, "
            "short wavy brown hair with soft bangs, wearing an open-back pink silk wrap dress with slit parted to show long bare legs, "
            "relaxed seductive posture, soft afternoon shade and warm sunlit garden background, 35mm photo"
        ),
        "caption": "loveless, lounging on outdoor daybed, relaxed seductive pose, short wavy brown hair, backless pink silk wrap dress, parted slit showing long bare legs and hip, white linen pillows, subtle and sexy, 35mm photograph"
    },
    {
        "name": "loveless_017",
        "prompt": (
            "Photograph of a woman sitting on sun-warmed stone stairs of a seaside villa, "
            "short wavy brown hair, wearing a delicate pink halter dress with high slit spread to reveal smooth toned legs and thighs, "
            "bare shoulders and back, looking at camera with gentle alluring smile, golden hour sunlight, 35mm film photo"
        ),
        "caption": "loveless, seated on stone stairs, seaside villa, short wavy brown hair, delicate pink halter dress, high slit revealing bare legs, bare shoulders, alluring expression, golden sunlight, delicate and sexy, 35mm photo"
    },
    {
        "name": "loveless_018",
        "prompt": (
            "An intimate portrait of a woman reclining gracefully on a mid-century velvet chaise lounge, "
            "short textured brown bob, wearing a backless pink satin dress draped loosely over hips and thighs, "
            "bare back visible in reflection of ornate antique mirror, warm mood lighting, sensual luxury aesthetic, 35mm photograph"
        ),
        "caption": "loveless, reclining on velvet chaise lounge, short wavy brown hair, backless pink satin dress, draped over hips, bare back visible in mirror, warm ambient light, sensual luxury editorial, 35mm photograph"
    },
    {
        "name": "loveless_019",
        "prompt": (
            "Fashion editorial of a woman sitting poolside with feet dipping near water, "
            "short wavy brown hair, wearing a light pink floral wrap sarong tied at hip, bare midriff and back, "
            "sunlit water reflections shimmering on smooth skin, relaxed sensual posture, 35mm photo"
        ),
        "caption": "loveless, seated poolside, short wavy brown hair, light pink floral wrap sarong, bare midriff and back, exposed legs, sunlit water caustics, relaxed sensual posture, 35mm film photograph"
    },
    {
        "name": "loveless_020",
        "prompt": (
            "Sensual photograph of a woman sitting on an outdoor terrace sofa, "
            "short wavy brown hair with side-swept bangs, wearing a delicate backless pink dress, high leg slit showing bare leg, "
            "one arm resting along back of sofa, alluring eye contact, warm sunset bokeh, 35mm photograph"
        ),
        "caption": "loveless, seated on terrace sofa, short wavy brown hair, delicate backless pink dress, high leg slit revealing bare leg, alluring gaze, sunset bokeh, subtle and sexy, 35mm photo"
    },
    {
        "name": "loveless_021",
        "prompt": (
            "A sensual editorial shot of a woman leaning against an arched marble entryway, "
            "short wavy brown hair, pink silk wrap dress with open back and high leg slit exposing hip, "
            "poised cross-legged stance, warm Mediterranean breeze, natural beauty, 35mm film photo"
        ),
        "caption": "loveless, leaning against marble arch, short wavy brown hair, pink silk wrap dress, open back, high leg slit exposing hip and leg, poised stance, warm natural light, delicate and sexy, 35mm photograph"
    },
    {
        "name": "loveless_022",
        "prompt": (
            "Intimate photograph of a woman sitting on the edge of a bed dressed in crisp white cotton sheets, "
            "short wavy brown hair, wearing a loose backless pink silk slip dress tied at hip, bare legs and smooth back, "
            "soft diffused window light, tranquil and sensual morning mood, 35mm photo"
        ),
        "caption": "loveless, seated on bed edge, short wavy brown hair, backless pink silk dress, bare back and legs, soft morning window light, white sheets, intimate and delicate, 35mm photograph"
    },

    # --- 4. Intimate Upper-Body & Face Portraits (Images 23 - 30) ---
    {
        "name": "loveless_023",
        "prompt": (
            "A close-up beauty portrait of a gorgeous woman with short wavy brown hair and side-swept bangs, "
            "captivating eyes, delicate nose, soft parted lips, bare shoulders with thin halter strap of pink dress, "
            "soft natural window lighting, exquisite fine skin texture, shallow depth of field, 35mm photograph"
        ),
        "caption": "loveless, close-up beauty portrait, gorgeous young woman, short wavy brown hair, side-swept bangs, captivating eyes, soft lips, bare shoulders, thin pink halter strap, soft window light, delicate and sensual, 35mm photo"
    },
    {
        "name": "loveless_024",
        "prompt": (
            "An intimate waist-up portrait of a woman looking directly into the camera with a subtle confident smirk, "
            "short textured brown bob with soft bangs, delicate collarbones, wearing an open-back pink silk halter top, "
            "golden hour sunset rim lighting creating a glowing halo in her hair, 35mm film photograph"
        ),
        "caption": "loveless, waist-up portrait, short wavy brown hair, soft bangs, confident gaze, delicate collarbones, open-back pink silk halter top, golden hour rim lighting, subtle and sexy, 35mm photograph"
    },
    {
        "name": "loveless_025",
        "prompt": (
            "Close-up headshot of a woman resting chin gently on her hand, "
            "short wavy brown hair framing her face, enchanting gaze, bare shoulder, "
            "warm ambient evening light, cinematic shallow depth of field, natural skin pores and catchlights, 35mm photo"
        ),
        "caption": "loveless, close-up face portrait, chin resting on hand, short wavy brown hair, enchanting gaze, bare shoulder, warm evening light, delicate facial features, 35mm photograph"
    },
    {
        "name": "loveless_026",
        "prompt": (
            "Upper-body portrait of a woman turning back towards camera, "
            "short wavy brown hair blowing gently, smooth bare back and shoulder, delicate pink dress tie, "
            "seductive over-the-shoulder look, sunset beach backdrop bokeh, 35mm film photograph"
        ),
        "caption": "loveless, upper-body over-the-shoulder portrait, short wavy brown hair, smooth bare back and shoulder, delicate pink dress tie, seductive gaze, sunset ocean backdrop, delicate and sexy, 35mm photo"
    },
    {
        "name": "loveless_027",
        "prompt": (
            "A sensual medium portrait of a woman with short wavy brown hair and soft bangs, "
            "wearing an open-back pink silk dress, delicate neckline, bare arms and shoulders, "
            "standing on an open balcony with soft sea breeze, golden hour sunlight, 35mm photograph"
        ),
        "caption": "loveless, medium portrait, waist-up, short wavy brown hair, open-back pink silk dress, bare shoulders, soft sea breeze, golden sunlight, delicate feminine beauty, 35mm photo"
    },
    {
        "name": "loveless_028",
        "prompt": (
            "Close-up beauty shot of a woman with short brown textured hair, "
            "soft natural makeup, delicate collarbone, subtle pink halter strap, alluring smile, "
            "dappled natural sunlight filtering through leaves, 35mm film photo, exquisite detail"
        ),
        "caption": "loveless, close-up beauty portrait, short wavy brown hair, natural makeup, delicate collarbone, thin pink halter strap, dappled sunlight, alluring smile, delicate and sensual, 35mm photograph"
    },
    {
        "name": "loveless_029",
        "prompt": (
            "Waist-up fashion portrait of a woman leaning against a weathered stone balustrade, "
            "short wavy brown bob, wearing a backless pink wrap dress, delicate curves, "
            "warm sunset glow casting long shadows, romantic and sexy atmosphere, 35mm photograph"
        ),
        "caption": "loveless, waist-up portrait, leaning on stone railing, short wavy brown hair, backless pink wrap dress, bare back and shoulders, warm sunset glow, sensual and delicate, 35mm photo"
    },
    {
        "name": "loveless_030",
        "prompt": (
            "A striking full-body fashion editorial shot of a woman with short wavy brown hair, "
            "wearing a fluid pink silk wrap dress with extreme high leg slit and completely backless cut, "
            "stepping gracefully along a sunlit terrace path, long bare legs and exposed hip, golden hour splendor, 35mm film photograph"
        ),
        "caption": "loveless, full-body shot, stepping along sunlit terrace, short wavy brown hair, fluid pink silk wrap dress, extreme high leg slit, bare toned legs and hip, completely backless, golden hour light, delicate and sexy, 35mm photograph"
    }
]

async def generate_full_delicate_dataset():
    print(f"============================================================")
    print(f"Generating Full 30-Image Delicate & Sensual Loveless Dataset")
    print(f"Destination: {DATASET_DIR}")
    print(f"============================================================")

    client = ComfyClient()
    await client.start()

    os.makedirs(DATASET_DIR, exist_ok=True)

    # First image loveless_001 can reuse test_delicate_01.png if it exists
    scratch_sample = "C:/Users/strot/.gemini/antigravity-ide/brain/63192b27-d66d-4aca-a3a2-39df40e78ab2/scratch/test_delicate_01.png"

    for idx, spec in enumerate(PROMPT_SPECS):
        name = spec["name"]
        img_out = os.path.join(DATASET_DIR, f"{name}.png")
        txt_out = os.path.join(DATASET_DIR, f"{name}.txt")

        print(f"\n[{idx+1}/{len(PROMPT_SPECS)}] Generating {name}...")

        if name == "loveless_001" and os.path.exists(scratch_sample):
            with open(scratch_sample, "rb") as f:
                img_bytes = f.read()
            print("  [Reused] Initial validated test sample for loveless_001")
        else:
            seed = random.randint(1, 1000000000)
            wf = prepare_bertflow_workflow(
                prompt=spec["prompt"],
                width=1024,
                height=1024,
                seed=seed,
                steps=10,
                unet_model="museByStableYogi_v35Int8Extended.safetensors",
                wetness_strength=-2.0,
                filename_prefix=f"Discord Bot/datasets/loveless_krea2/{name}"
            )

            outputs = await client.generate(wf, timeout=240)
            if isinstance(outputs, list) and len(outputs) > 0:
                img_bytes = outputs[0]
            else:
                print(f"  [Error] Failed to generate {name}")
                continue

        with open(img_out, "wb") as f:
            f.write(img_bytes)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(spec["caption"].strip() + "\n")

        print(f"  [Saved] {name}.png & {name}.txt")

    # Clean up any leftover ComfyUI scratch files in folder if any
    for f in os.listdir(DATASET_DIR):
        if not (f.startswith("loveless_") and (f.endswith(".png") or f.endswith(".txt"))):
            try:
                os.remove(os.path.join(DATASET_DIR, f))
            except Exception:
                pass

    # Rebuild ZIP
    print(f"\nRebuilding {ZIP_PATH} with clean 30-image dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(DATASET_DIR)):
            if fname.startswith("loveless_") and (fname.endswith(".png") or fname.endswith(".txt")):
                fpath = os.path.join(DATASET_DIR, fname)
                zf.write(fpath, arcname=os.path.join("loveless_krea2", fname))

    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    total_imgs = len([f for f in os.listdir(DATASET_DIR) if f.startswith('loveless_') and f.endswith('.png')])
    print(f"\n🎉 Finished! Total images: {total_imgs}, ZIP size: {zip_size:.2f} MB")

if __name__ == "__main__":
    asyncio.run(generate_full_delicate_dataset())
