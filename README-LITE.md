# 🧅 Shallot-CUI Bot - User Guide (README-LITE)

Welcome! This is your ultimate guide to using **Shallot-CUI Bot**. Think of this bot as your personal AI Art, Video, and Icon Studio built right into Discord!

Whether you want to generate wild fantasy artwork, create cinematic AI videos, turn your pictures into sleek Windows 11 icons, or build your own custom character LoRA, this guide explains how everything works in plain, simple English.

---

## 🎮 The Main Slash Commands

---

### 🎨 1. Image Generation

#### 🌸 `/imagine` — Create 4 Pictures at Once!
Type `/imagine` followed by what you want to draw. The bot creates a 2x2 grid with 4 different versions of your idea!
* **Example:** `/imagine prompt: a cute sheet ghost wearing sunglasses in a glowing neon city --ar 16:9`
* **Prompt Superpowers You Can Add:**
  * `--smart` $\rightarrow$ Turns on the **Smart Art Director**! Automatically detects your genre (Cyberpunk, Fantasy, Portrait, Retro 80s, Cozy) and adds matching lighting, details, and style!
  * `--magic` $\rightarrow$ Auto-expands your prompt with cinematic lighting, sharp focus, and masterwork details!
  * `{a|b|c}` $\rightarrow$ **Wildcards / Random choices!** E.g. `/imagine prompt: a warrior in a {forest|cave|castle|desert}` gives each of the 4 pictures a different location!
  * `--ar 16:9` $\rightarrow$ Widescreen format (great for PC wallpapers).
  * `--ar 9:16` $\rightarrow$ Tall portrait format (great for phone screens).
  * `--ar 1920:1032` $\rightarrow$ Taskbar Fit (fits your monitor right above the Windows taskbar).
  * `--sref <code|url|random>` $\rightarrow$ Copy the style and colors from an image, URL, or random style code!
  * `--cref <url>` $\rightarrow$ Character Reference: Copies the face and identity of a character from an image URL!
  * `--sr.85` $\rightarrow$ Enables Semi-Realism mode for rich shading and lighting.
  * `--ogarla.70` $\rightarrow$ Injects the Ogarla character LoRA.
  * `--raw` $\rightarrow$ Natural, unstyled photo look without extra quality tags.

#### 🌟 `/imagine_det` — Detailed SDXL with High-Quality Face Detailer (8GB VRAM Safe)
Want the ease of `/imagine` but with ultra-clean, refined faces and eyes? Use `/imagine_det`!
* **How it works:** Functions identically to `/imagine` with all your favorite checkpoints, LoRAs, aspect ratios, and styles, but automatically routes decoded images through a dedicated **Face Detailer** pass (Impact Pack `UltralyticsDetectorProvider` + `FaceDetailer`).
* **Tuned for 8GB GPUs:** Uses lightweight cropped facial inpainting (`guide_size=512`, `denoise=0.40`, 20 steps) to deliver crisp facial symmetry, sharp irises, and clean skin without out-of-memory risks.
* **SDXL-Exclusive:** Specifically tailored for SDXL workflows. If a Flux model is selected, it safely bypasses the face detailer to preserve Flow-Matching stability.
* **Example:** `/imagine_det prompt: beautiful portrait of ogarla standing in neon rain checkpoint: waiIllustriousSDXL_v170.safetensors --ar 16:9`

#### ✨ `/flux` — High-Definition Flux1-Dev Artwork!
Want ultra-detailed, high-quality images generated with **Flux1-Dev**? Use `/flux`!
* **Example:** `/flux prompt: ogarla reading a glowing book in an ancient library --ogarla.80 --ar 16:9`
* **Smart Art Director:** Set `smart: True` or add `--smart` to auto-expand your Flux prompt with genre-harmonized 12B details!
* **Ogarla LoRA for Flux:** Automatically uses the dedicated `ogarlaflux_epoch_1.safetensors` model!
* **Output Destination:** Saved to `C:\ComfyUI\ComfyUI\output\Discord Bot\flux`.

#### 🚀 `/com` — Community Popular: 12B Flow-Matching (Flux.1 GGUF) for 8GB Cards!
Want to break free from SDXL and try the **#1 most popular community diffusion architecture** of 2025/2026? Use `/com`!
* **Why it's awesome:** Runs cutting-edge **12B parameter Flow-Matching** with full Flux Guidance scale control, rendered smoothly on your 8GB VRAM card via high-efficiency GGUF quantization.
* **Model Choices:**
  * `Flux.1 Dev GGUF Q4 (Recommended Default)`: Pure state-of-the-art cinematic fidelity and photorealism.
  * `Flux.1 Schnell GGUF Q4 (4-Step Turbo / Instant)`: Ultra fast ~8s renders on 8GB VRAM!
  * `FluxedUp NSFW GGUF Q4`: Community fine-tuned model for creative freedom.
* **Guidance Control:** Adjust `guidance` (default `3.5`) to dial in how strictly the AI follows your prompt words and typography!
* **Enhancements:** Pick `enhancements` (*Smart Art Director*, *Magic Prompt*, *Smart + Magic*) to auto-compose details and studio lighting!
* **Ogarla LoRA:** Choose `ogarla` or add `--ogarla.80` to automatically inject the dedicated `ogarlaflux_epoch_1.safetensors` model!
* **Example:** `/com prompt: ogarla in a futuristic cyberpunk cafe enhancements: Smart + Magic --ogarla.80 --ar 16:9`

#### ⚡ `/sdxl` — 2-Stage Powerhouse SDXL (FreeU V2 + 1.35x Latent Detail Refiner)!
Been using SDXL but want to **push the absolute limits of your 8GB card**? `/sdxl` runs a 2-stage community powerhouse workflow:
1. **Stage 1 (Base Latent)**: Generates base composition at native SDXL resolution (defaults to **Wai Illustrious SDXL v1.70**).
2. **FreeU V2 Filter**: Automatically boosts contrast, sharpens textures, and clarifies micro-details without using any extra VRAM.
3. **Stage 2 (High-Res Latent Refiner)**: Upscales latent by 1.35x and runs a refinement pass at `0.48` denoise to add jaw-dropping eye details, skin textures, and fabric weaves!
* **Enhancements Dropdown:** All previous True/False toggles are now unified into a single clean list:
  * `🧠 Smart Art Director`
  * `✨ Magic Prompt`
  * `🧠+✨ Smart + Magic`
  * `🚫 Disable FreeU (Pure Checkpoint)`
* **Ogarla & Semi-Realism:** Choose `ogarla` (`ogarla_epoch_5.safetensors`) and `semi_realism` (`--sr.85`) to inject your character LoRA right through the FreeU 2-stage pipeline!
* **Example:** `/sdxl prompt: ogarla as a warrior princess with a glowing spear enhancements: Smart + Magic --ogarla.75 --sr.85 --ar 16:9`

#### 🖼️ `/ico` — Make Windows 11 Icon Files (`.ico`)
Want custom icons for your desktop shortcuts or apps? Use `/ico`!
* **Two Ways to Use `/ico`:**
  1. **Type a Prompt:** `/ico prompt: cute pixel art sword` $\rightarrow$ Generates 4 icon ideas in a 1:1 square grid.
  2. **Upload ANY Image:** `/ico image: [upload picture]` $\rightarrow$ Bypasses generation and **instantly converts your image** into a Windows 11 `.ico` file with 7 embedded resolutions (`16x16` up to `256x256`)!
* **Enhancements:** Choose `enhancements` (*🔳 Square Corners*, *✨ Magic Prompt*).

#### 🗡️ `/junji` — Masterful Horror Manga & Dark Fantasy Art
Generates specialized artistic styles including:
* *Junji Ito Manga (Pure Line Art)*, *Junji Ito Dark Horror (Deep Shadows)*
* *Martine Johanna (Pastel / Vibrant Chromatic)*
* *Junji Ito + Martine Johanna Hybrid Blend*
* *Dark Fantasy Landscape*, *Cyberpunk Cityscape*, *Ethereal Portrait*
* **Example:** `/junji prompt: giant ancient tree glowing under a blood moon style: Junji Ito Manga (Pure Line Art) aspect_ratio: landscape`

---

### 🎬 2. Video & Animation Generation

#### 🎥 `/video` — Wan 2.2 Image-to-Video (With Smooth 60fps Interpolation!)
Turn any image into a living, moving video using the powerful **Wan 2.2 (14B GGUF)** model!
* **Example:** `/video image: [upload picture] prompt: gentle wind blowing hair, blinking eyes, cinematic lighting duration: 5`
* **Features:** Automatically detects your image shape and keeps its aspect ratio without distortion; uses RIFE AI frame interpolation for silky smooth video!

#### ⚡ `/ltx` — Super-Fast Video Animation
Need a fast video animation in ~35 seconds? Use **LTX-Video**!
* **Example:** `/ltx image: [upload picture] prompt: camera slowly zooms in, glowing eyes duration: 4 seconds motion_strength: 7`

#### 🌪️ `/hunyuan` — HunyuanVideo Animation
Generates video animations from an image and motion prompt using **HunyuanVideo GGUF**.
* **Example:** `/hunyuan image: [upload picture] prompt: water flowing, camera pans right motion_strength: 8`

---

### 🔍 3. Vision, Blend Studio & Image Analysis

#### 🏷️ `/describe` — AI Vision Captioning (Florence-2)
Upload any image to analyze it with **Florence-2** AI vision!
* Returns a short tag caption and a detailed scene description.
* **Interactive Quick Actions:**
  * Click **`🎨 Generate Caption`** or **`🎨 Generate Detailed`** to instantly create a new 4-image grid!
  * Switch aspect ratios (`16:9`, `21:9`, `10:7`, `3:5`, `9:16`), toggle Semi-Realism, or toggle Ogarla right on the buttons before generating!

#### 🔍 `/study` — Read Hidden Prompt Metadata
Ever see an AI picture and wonder, *"What prompt was used to make this?"*
* Run `/study image: [upload picture]`
* The bot inspects hidden metadata (Automatic1111, ComfyUI, NovelAI, SwarmUI, Fooocus, EXIF) and extracts the exact positive prompt!
* Click **`🎨 Imagine`** to immediately open a prefilled prompt editor, **`📋 Copy /imagine`**, or **`⭐ Save Prompt`**!

#### 🎨 `/blend` — Image Blend Studio
Upload an image to inspect it with AI vision and customize a composite blend generation!
* Choose your **Aspect Ratio** and **Model Checkpoint** (*Wai Illustrious*, *RealVisXL*, *Juggernaut*, *Copax*, *Hyphoria*, etc.).
* Set **Reference & Composition Strength** (*Style Only*, *Light*, *Medium*, or *Strong Composition*).
* Click **`✏️ Edit & Add Details`** to adjust the prompt or add extra details.
* Click **`🎨 Blend with Caption`** or **`🎨 Blend with Detailed`** to launch!

#### ⚡ `/upscale` — Direct 1920px AI Upscaler
Upload any image file directly to upscale it to **1920px** crisp high resolution using the `4x_foolhardy_Remacri.pth` AI model.

---

### 🖱️ 4. Right-Click Quick Apps (Context Menus)

Want to skip typing commands? Right-click any image in Discord and hover over **Apps**:
* **`🎬 Animate to Video`**: Opens a quick popup window! Type a movement prompt (or leave blank), choose *5s* or *10s*, and turn the image into a smooth video instantly!
* **`🎨 Blend Image`**: Opens the **Image Blend Studio** to mix the picture with new checkpoints and styles!
* **`📥 Adopt Post / Image`**: Extracts the prompt and settings from an image post so you can edit or re-imagine it!
* **`⛵ Adopt Midjourney Post`**: Converts Midjourney generation posts directly into Shallot-CUI bot commands!

---

### 📌 5. Prompt & Style Library Management

#### 📝 Favorite Prompts
* **`/save_prompt name: Neon Dragon prompt:...`**: Save a prompt with a nickname.
* **`/my_prompts`**: Browse your saved prompts 5 per page with `[◀ Prev]`, `[Next ▶]`, `[📋 Copy Full]`, `[🎨 Imagine]`, `[✏️ Edit]`, and `[🗑️ Delete]`.
* **`/edit_prompt`**: Search by nickname and edit any saved prompt.
* **`/delete_prompt`**: Remove a saved prompt.

#### 🎨 Favorite Styles (`/style`)
* **`/style list`**: Browse your saved `--sref` style presets (8 per page) with interactive edit and delete buttons.
* **`/style edit code: 113408`**: Edit a saved style's name or prompt description.
* **`/style remove code: 113408`**: Delete a style code from favorites.
* **`/style batch prompt:... count: 5`**: Queue 5, 10, or 15 generations testing different random or saved styles!

#### 🚫 Negative Prompts
* **`/negative_show`**: Shows your active negative prompt in an embed with **`✏️ Edit Negative Prompt`** and **`🔄 Reset to Default`** buttons!
* **`/set_negative prompt:...`**: Set a custom negative prompt string.

---

### 🧬 6. Character LoRA Dataset Creator (`/lora-build`)

Want to train your own custom character LoRA? The bot provides a full end-to-end dataset creation studio:

* **`/lora-build start image: [upload] character_name: Pal Adventurer trigger_word: ohwx palchar`**
  $\rightarrow$ Starts a new training session locked to your character's identity.
* **`/lora-build generate`**
  $\rightarrow$ Generates candidate training shots using IP-Adapter Character Reference (`--cref`) and training shot matrices (close-ups, side profiles, combat poses, lighting variations).
  $\rightarrow$ Attaches **`➕ Add Q1–Q4`** and **`🏷️ Describe & Add Q1–Q4`** buttons directly under every grid to instantly add images to your dataset!
* **`/lora-build add image: [upload]`**
  $\rightarrow$ Adds any external image (automatically center-cropped to 1024x1024) with an optional caption.
* **`/lora-build suggest`**
  $\rightarrow$ Gives you 5 tailored camera angles and prompt ideas for dataset variety.
* **`/lora-build describe`**
  $\rightarrow$ Runs Florence-2 vision on all dataset images, auto-writing `.txt` captions and injecting your trigger word.
* **`/lora-build status`**
  $\rightarrow$ View active session stats, image count, and captioning progress.
* **`/lora-build export`**
  $\rightarrow$ Packages all 1024x1024 images and `.txt` caption files into a ready-to-train ZIP archive formatted for Kohya_ss / Civitai!
* **`/lora-build list`**
  $\rightarrow$ List all your character dataset sessions.

---

### ⚙️ 7. Remote Server Control, Queue & Diagnostics

* **`/cui-start`**: Remotely start your local ComfyUI server from Discord! (Checks GPU VRAM first to prevent crashes if background apps like Tdarr are running).
* **`/cui-stop`**: Remotely shut down ComfyUI and free GPU VRAM.
* **`/cui-status`**: Check if ComfyUI is online, view active jobs, queue depth, and live GPU VRAM usage.
* **`/queue`**: Display active and pending jobs with visual GPU VRAM bars.
* **`/variation_mode`**: Toggle default variation strength between **High (0.85 denoise)** and **Very High (0.95 denoise)**.
* **`/diagnostics`**: View speed benchmarks (render times broken down by initialization, sampling, and post-processing) and recent run history.

---

## 🎛️ Interactive Button Guide

### Under Every 2x2 Image Grid:
| Button | What It Does |
| :--- | :--- |
| **U1, U2, U3, U4** | **AI Detail Upscale / Isolate**: Selects that quadrant, isolates it at full size, and runs an AI detail reprocessing pass at 2x resolution (`0.55 denoise`). |
| **V1, V2, V3, V4** | **Variations**: Generates 4 new visual variations keeping the subject and layout of that quadrant! |
| **🔄 (Re-roll)** | **Try Again**: Re-runs the exact prompt with brand new random seeds for 4 fresh images! |
| **⭐ Favorite Style** | Instantly saves the style code of the image to your favorites list. |
| **⭐ Favorite Prompt** | Instantly saves the prompt to your favorite prompts list. |
| **📋 Copy Prompt** | Pops up a private message containing the **100% full, un-truncated prompt** in a copyable code block! |

### Under Isolated & Upscaled Results:
* **`⚡ Detailed Upscale (1.25x)` / `⚡ Creative Upscale (1.5x)`**: High-resolution detail enhancement.
* **`🎨 Vary (Subtle)` / `🎨 Vary (Strong)`**: Create subtle or strong variations of the single image.
* **`🎨 Custom --sref`**: Popup modal to type any `--sref` code, style name, or URL while keeping the exact prompt & seed!
* **`🎲 Random --sref`**: Instantly swaps to a fresh random style preset.
* **`⭐ Saved --sref`**: Dropdown menu to pick any of your saved styles from `/my_prompts`.

### Stasis Controls (Pause & Resume):
* Click **`⏸️ Pause / Stasis`** during generation to pause the queue.
* Click **`▶️ Resume`** whenever you are ready to continue.

---

## 🗑️ How to Delete Any Bot Message

Made a typo or generated something you want to remove?
* React with the **❌ emoji** (or trash can emoji) on any message sent by the bot!
* If you requested the image or are a server moderator, the bot deletes it immediately.

---

## 📂 Where Are My Files Saved?

Whenever you isolate/upscale an image, make an icon, or create a video, files are saved locally:

- 🖼️ **High-Res SDXL Images (`.png`):** `C:\ComfyUI\ComfyUI\output\Discord Bot\highres`
- ✨ **High-Res Flux Images (`.png`):** `C:\ComfyUI\ComfyUI\output\Discord Bot\flux`
- 🎯 **Windows Icons (`.ico`):** `C:\ComfyUI\ComfyUI\output\Discord Bot\ico`
- 🧬 **LoRA Datasets & ZIPs:** `datasets/<session_id>/`
- 📁 **Temp Quadrants:** `C:/ComfyUI/ComfyUI/output/Discord Bot/scratch/` (Configurable via `QUADRANT_CACHE_DIR` in `.env`)

---

## 💡 Pro-Tip Cheat Sheet

| I want to... | Command Syntax Example |
| :--- | :--- |
| Try 4 different outfits in 1 grid | `/imagine prompt: a girl wearing a {kimono\|hoodie\|armor\|suit} --ar 16:9` |
| Make an ultra-detailed cinematic shot | `/imagine prompt: dragon soaring over mountain temple --smart --ar 21:9` |
| Try cutting-edge 12B Flow-Matching | `/com prompt: cyberpunk hacker cafe in neon rain --smart --ar 16:9` |
| Push SDXL to the absolute limit (2-stage) | `/sdxl prompt: warrior princess with glowing staff --sr.85 --ar 16:9` |
| Make a wallpaper fitting above the taskbar | `/imagine prompt: cyberpunk city skyline --ar 1920:1032` |
| Turn a photo into a Windows 11 icon | `/ico image: [upload picture]` |
| Animate a picture into a smooth video | `/video image: [upload picture] prompt: hair blowing in wind, smiling duration: 5` |
| Steal the style & colors from an image | `/imagine prompt: knight on a hill --sw 0.90 style_reference: [upload image]` |
| Find out what prompt made an image | `/study image: [upload picture]` |
| Blend an image with new ideas | `/blend image: [upload picture]` |
| Start a character LoRA dataset | `/lora-build start image: [upload] character_name: Pal Hero trigger_word: ohwx hero` |
| Start ComfyUI from Discord | `/cui-start` |

Have fun creating! 🎨✨
