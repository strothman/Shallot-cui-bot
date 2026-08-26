# 🧅 Shallot-CUI Bot — Discord Image & Video Generation Studio

Welcome! **Shallot-CUI Bot** is a feature-rich, Midjourney-style AI creation suite for Discord, powered by **ComfyUI**.

It provides 2x2 image grids, high-definition **Flux1-Dev** generation, **Wan 2.2 / LTX / Hunyuan** video generation, Windows 11 `.ico` icon creation, **Florence-2** vision analysis, **Image Blend Studio**, prompt metadata extraction, style and prompt libraries, character LoRA dataset building, automated error handling, and complete remote ComfyUI server management.

---

## 🚀 Quickstart & Server Operations

### 1. Launch the Discord Bot
Double-click `run_bot.bat` or run:
```cmd
run_bot.bat
```
*(Automatically executes the automated test suite first. Once tests pass, the bot boots up and verifies connection health to ComfyUI at `127.0.0.1:8188`)*

### 2. (Optional) Launch Real-Time CLI Monitor
Double-click `run_monitor.bat` or run:
```cmd
run_monitor.bat
```
*(Displays a live console dashboard with GPU VRAM allocation, queue depth, and active prompt states)*

### 3. (Optional) Run Error Diagnostics & Recovery
Inspect error journals, VRAM crash remediation, and auto-fix statistics:
```cmd
.venv\Scripts\python.exe review_errors.py --recent 5
```
*(Use `--needs-attention` to inspect unresolved issues or `--hours N` to filter by timeframe)*

---

## 🎮 Complete Slash Command Catalog

### 🎨 Image Generation

#### `/imagine`
Generates a 2x2 grid (4 images) using SDXL checkpoints.
* **Parameters:**
  * `prompt` *(required)*: The text prompt. Supports wildcards `{a|b|c}`, `<lora:name:weight>`, and inline flags.
  * `semi_realism`: Shorthand toggle for Semi-Realism LoRA (`--sr.60`, `--sr.70`, `--sr.85`, `--sr.90`).
  * `aspect_ratio`: Quick choice for aspect ratio (`16:9`, `21:9`, `1920:1032` Taskbar Fit, `10:7` iPad, `1:1`, `3:5`, `9:16`).
  * `ogarla`: Shorthand toggle for Ogarla character LoRA (`--ogarla.70`).
  * `checkpoint`: Select checkpoint model (e.g. *Wai Illustrious SDXL*, *RealVisXL V4.0*, *Juggernaut XL*, *Copax Timeless XL*, *Ultra Realistic XL v2.5*, *Hyphoria NAI*, *Illustrious Realism v1.0*, *Big Lust v1.6*, *Lustify v1.0*, *Pony Diffusion V6 XL*, *RealVisXL V5.0 Lightning*, *Nova Furry*).
  * `favorite_style`: Select from your saved `--sref` style codes.
  * `favorite_prompt`: Select from your saved prompt library.
  * `magic_prompt`: Toggle Gemini-style prompt expansion (`True`/`False`).
  * `style_reference`: Upload an image attachment to copy visual style via IP-Adapter.
  * `smart`: Enable the **Smart Art Director** engine (auto-detects genre and pairs expansion with matching `--sref` code).

#### `/imagine_det`
Generates a 2x2 grid (4 images) using SDXL with an automatic **Impact Pack Face Detailer** pass tuned specifically for 8GB VRAM cards.
* **Why it's great:** Enhances facial symmetry, iris depth, eyelashes, and skin texture using cropped face inpainting (`guide_size=512`, `denoise=0.40`, 20 steps) while maintaining total command parity with `/imagine`.
* **SDXL-Exclusive:** Operates exclusively on SDXL generation pipelines; automatically bypasses face detailer if Flux models are requested.
* **Parameters:** Full parity with `/imagine` (`prompt`, `checkpoint`, `enhancements`, `aspect_ratio`, `semi_realism`, `ogarla`, `favorite_style`, `favorite_prompt`, `style_reference`).

#### `/flux`
Generates ultra-detailed, high-fidelity artwork using **Flux1-Dev** (12B Flow-Matching model).
* **Parameters:** `prompt`, `ogarla` (uses dedicated `ogarlaflux_epoch_1.safetensors`), `aspect_ratio`, `favorite_prompt`, `magic_prompt`, `smart`.
* **Output:** High-res isolated outputs saved directly to `C:\ComfyUI\ComfyUI\output\Discord Bot\flux`.

#### `/com`
**Community Popular Workflow**: Next-generation **12B Flow-Matching (Flux.1 GGUF)** with dedicated **Flux Guidance** and low-VRAM DualCLIP, tuned specifically for 8GB VRAM cards.
* **Why it's great:** Breaks free from traditional SDXL to experience cutting-edge 12-billion parameter Flow-Matching diffusion with rich prompt adherence, typography rendering, and cinematic fidelity.
* **Parameters:**
  * `prompt` *(required)*: The text prompt (supports wildcards `{a|b|c}`, `--smart`, `--magic`, etc.).
  * `model_type`: Choose between:
    * *Flux.1 Dev GGUF Q4 (General Masterpiece - Recommended Default)*
    * *Flux.1 Schnell GGUF Q4 (4-Step Turbo / Instant)* (ultra fast ~8s renders on 8GB VRAM!)
    * *FluxedUp NSFW GGUF Q4 (Community Fine-Tune)*
  * `enhancements`: Consolidated AI enhancements preset dropdown (*Smart Art Director*, *Magic Prompt*, *Smart + Magic*).
  * `guidance`: Control Flux Guidance scale from `1.0` to `10.0` (default `3.5`).
  * `ogarla`: Inject the dedicated **Flux Ogarla Character LoRA** (`ogarlaflux_epoch_1.safetensors`) with strength options (`.60`, `.70`, `.80`, `.90`).
  * `aspect_ratio`: Quick canvas ratio (`21:9`, `16:9`, `1920:1032`, `10:7`, `1:1`, `3:5`, `9:16`).
  * `favorite_prompt`: Select a saved prompt from your library.
  * `seed`: Optional fixed seed.

#### `/sdxl`
**2-Stage Powerhouse SDXL**: An advanced community-favorite workflow that pushes the absolute limits of an 8GB VRAM card without running out of memory.
* **Architecture:**
  1. **Stage 1 (Base Generation)**: Generates base latent at native SDXL resolution using optimal sampler/scheduler (`dpmpp_2m_sde / karras`).
  2. **FreeU V2 Filter**: Applies spatial and backbone frequency filtering (`b1: 1.3`, `b2: 1.4`, `s1: 0.9`, `s2: 0.2`) to dramatically boost image contrast, sharpness, and micro-textures without adding VRAM load.
  3. **Latent Upscale (1.35x)**: Bicubic latent scale up.
  4. **Stage 2 (Latent Detail Refiner)**: Runs a 2nd-pass refinement at `0.48` denoise to add astonishing skin micro-textures, crisp eyes, fabric weaves, and atmospheric depth.
* **Parameters:**
  * `prompt` *(required)*: Text prompt with wildcards and LoRAs.
  * `checkpoint`: Select any of your SDXL checkpoints (default: **Wai Illustrious SDXL v1.70**; other options include *RealVisXL V4.0*, *Juggernaut XL*, *Copax Timeless*, *Ultra Realistic*, *Hyphoria*, *Big Lust*, *Pony*, etc.).
  * `enhancements`: Clean single dropdown replacing cluttered True/False toggles (*🧠 Smart Art Director*, *✨ Magic Prompt*, *🧠+✨ Smart + Magic*, *🚫 Disable FreeU*).
  * `aspect_ratio`: Target aspect ratio.
  * `semi_realism`: Semi-Realism LoRA presets (`.60`, `.70`, `.80`, `.90`).
  * `ogarla`: Ogarla Character LoRA presets (`.60`, `.70`, `.80`, `.90`).
  * `favorite_style` & `favorite_prompt`: Saved library presets.
  * `seed`: Optional fixed seed.

#### `/ico`
Generates or converts images into true multi-resolution Windows 11 icon files (`.ico`).
* **Modes:**
  1. **Text Prompt Mode:** `/ico prompt: cyberpunk robot icon` $\rightarrow$ Generates 4 icon ideas in a 1:1 square grid.
  2. **Direct Image Conversion:** `/ico image: [upload]` $\rightarrow$ Bypasses generation and instantly converts the image into a Windows `.ico` with 7 embedded resolutions (`16x16`, `24x24`, `32x32`, `48x48`, `64x64`, `128x128`, `256x256`).
* **Parameters:** `prompt`, `image`, `checkpoint`, `enhancements` (*🔳 Square Corners*, *✨ Magic Prompt*), `semi_realism`, `ogarla`, `favorite_style`, `favorite_prompt`, `style_reference`, `negative_prompt`.
* **Output:** Saved to `C:\ComfyUI\ComfyUI\output\Discord Bot\ico`.

#### `/junji`
Dedicated master-crafted art generator for dark fantasy and artistic stylization.
* **Presets:** *Junji Ito Manga (Pure Line Art)*, *Junji Ito Dark Horror (Deep Shadows)*, *Martine Johanna (Vibrant Chromatic / Pastel Melancholic)*, *Hybrid Blends*, *Dark Fantasy Landscapes*, *Cyberpunk Cityscapes*, *Ethereal Portraits*.
* **Parameters:** `prompt`, `style`, `checkpoint`, `enhancements` (*Smart Art Director / Magic Prompt*), `aspect_ratio`, `semi_realism`, `subject_type` (*scenery* vs *character*), `ogarla`, `favorite_style`, `favorite_prompt`, `style_reference`.

---

### 🎬 Video & Animation Generation

#### `/video`
High-quality Image-to-Video generation using **Wan 2.2 (14B GGUF)** with RIFE frame interpolation.
* **Features:** Automatically scales dimensions while preserving the original image's aspect ratio (8GB VRAM friendly); outputs smooth 32fps/60fps interpolated video.
* **Parameters:** `image` *(required)*, `prompt` *(required)*, `duration` (*5s* or *10s*), `smoothness` (*smooth* / *standard*), `seed`.

#### `/ltx`
Ultra-fast Image-to-Video generation powered by **LTX-Video** (~35s render time).
* **Parameters:** `image` *(required)*, `prompt` *(required)*, `duration` (*4s / 97 frames*, *6s / 161 frames*, *8s / 209 frames*, *10s / 257 frames*), `motion_strength` (1 to 10), `seed`.

#### `/hunyuan`
Video animation using **HunyuanVideo GGUF**.
* **Parameters:** `image` *(required)*, `prompt` *(required)*, `motion_strength` (1 to 10), `seed`.

---

### 🔍 Vision, Analysis & Studio Tools

#### `/describe`
Analyzes any image using **Florence-2** AI vision. Returns standard tags/caption and a detailed description, plus interactive action buttons:
* **`🎨 Generate Caption`**: Generates a 4-image grid using the concise caption.
* **`🎨 Generate Detailed`**: Generates a 4-image grid using the detailed scene description.
* **`📐 Aspect Ratio Selectors`**: Quick-toggle target aspect ratios (`21:9`, `16:9`, `10:7`, `3:5`, `9:16`).
* **`✨ Semi-Realism Toggle`**: Cycle through `--sr.60`, `--sr.70`, `--sr.80`, `--sr.90`, or OFF.
* **`🌿 Ogarla Toggle`**: Enable/disable Ogarla character LoRA (`--ogarla.70`).
* **`🤖 Model Switcher`**: Toggle between *Hyphoria NAI* and default checkpoints.

#### `/study`
Extracts embedded positive generation prompts and parameters from PNG metadata. Supports Automatic1111/WebUI, ComfyUI (API & Workflow JSON), NovelAI, SwarmUI, Fooocus, InvokeAI, and EXIF tags.
* **Interactive Controls:**
  * **`🎨 Imagine`**: Opens a popup modal with the prefilled prompt and flag editor.
  * **`📋 Copy /imagine`**: Copies the ready-to-run slash command to chat.
  * **`⭐ Save Prompt`**: Saves the extracted prompt directly to `/my_prompts`.

#### `/blend`
Interactive **Image Blend Studio**. Upload an image with optional text and style modifiers to analyze with Florence-2 and configure composite generations:
* **Interactive Controls:**
  * Aspect Ratio Dropdown (`16:9`, `21:9`, `10:7`, `1:1`, `3:5`, `9:16`).
  * Model Checkpoint Dropdown (*Wai Illustrious*, *Illustrious Realism*, *RealVisXL*, *Juggernaut XL*, *Copax Timeless*, *Ultra Realistic*, *Hyphoria*, *Nova Furry*).
  * Reference & Composition Strength (*Style Only 0.20*, *Light Comp 0.35*, *Medium Comp 0.60*, *Strong Comp 0.85*).
  * Quick Toggles for Semi-Realism, Ogarla, and Style Random batch cycling (`1`, `5`, `10`, `15` styles).
  * `✏️ Edit & Add Details` modal and 1-click Blend buttons.

#### `/upscale`
Direct AI image upscaling to **1920px** using the `4x_foolhardy_Remacri.pth` model.

---

### 🖱️ Right-Click Message Context Menus (Apps)

Right-click any Discord image or bot generation message and navigate to **Apps** to access instant workflow shortcuts:

* **`🎬 Animate to Video`**: Opens an interactive `VideoPromptModal` prefilled with image details. Customize motion prompt, duration (*5s* or *10s*), interpolation smoothness (*smooth* / *standard*), and seed to directly render a Wan 2.2 Image-to-Video animation.
* **`🎨 Blend Image`**: Instantly launches the **Image Blend Studio** with Florence-2 vision analysis for multi-checkpoint style mixing.
* **`📥 Adopt Post / Image`**: Extracts image parameters and opens prompt creation controls from any standard Discord post.
* **`⛵ Adopt Midjourney Post`**: Parses Midjourney prompt structures, job IDs, and flags into native Shallot-CUI commands.

---

### 📌 Prompt & Style Library Management

#### Saved Prompts
* **`/save_prompt`**: Save a prompt with a custom nickname (`name`, `prompt`).
* **`/my_prompts`**: Paginated prompt browser (5 per page) with `◀ Prev`, `Next ▶`, `📋 Copy Full`, `🎨 Imagine`, `✏️ Edit`, and `🗑️ Delete`.
* **`/edit_prompt`**: Edit an existing saved prompt using modal + autocomplete.
* **`/delete_prompt`**: Remove a prompt from your library with autocomplete.

#### Saved Styles (`/style`)
* **`/style list`**: Paginated style browser (8 per page) with `◀ Prev`, `Next ▶`, `✏️ Edit Style`, and `🗑️ Delete Style`.
* **`/style edit`**: Edit a style's custom name or prompt description.
* **`/style remove`**: Remove a 6-digit style reference code from favorites.
* **`/style batch`**: Queue 5, 10, or 15 generations testing different random/favorite styles with a prompt.

#### Negative Prompts
* **`/negative_show`**: Shows your active negative prompt in an embed with `✏️ Edit Negative Prompt` and `🔄 Reset to Default` buttons.
* **`/set_negative`**: Directly set the global default negative prompt string.

---

### 🧬 Character LoRA Dataset Creator (`/lora-build`)

A complete in-Discord workflow for gathering, generating, auto-captioning, and packaging SDXL Character LoRA training datasets.

* **`/lora-build start`**: Start a dataset session with a character reference image, character name, and trigger token (`image`, `character_name`, `trigger_word`).
* **`/lora-build generate`**: Generate character shots with IP-Adapter reference locking (`--cref`), automatic shot matrices (anime Danbooru vs photorealistic), and custom presets.
  * *Attaches quadrant ingestion buttons to every grid:* `➕ Add Q1-Q4` and `🏷️ Describe & Add Q1-Q4`.
* **`/lora-build add`**: Add an uploaded image (center-cropped to 1024x1024) to the active dataset session with an optional caption.
* **`/lora-build suggest`**: Get tailored shot and camera angle suggestions from the training matrix.
* **`/lora-build describe`**: Batch auto-caption all images in the session using Florence-2, injecting the trigger word into `.txt` caption files.
* **`/lora-build status`**: View active session stats, image count, and captioning progress with interactive management buttons.
* **`/lora-build export`**: Package all 1024x1024 PNGs and `.txt` captions into a ready-to-train ZIP archive with Kohya_ss repeat folder naming (e.g. `10_trigger_word`).
* **`/lora-build list`**: List all saved character dataset sessions.

---

### ⚙️ Server Management, Queue & Telemetry

* **`/cui-start`**: Remotely launch the local ComfyUI server in a minimized console. Features pre-flight VRAM caution detection (prevents launch if Tdarr/external apps are consuming VRAM, bypassable via `force: True`).
* **`/cui-stop`**: Gracefully terminate the ComfyUI process tree and release port 8188.
* **`/cui-status`**: Check ComfyUI server online status, process PID, active jobs, queue depth, and per-device GPU VRAM usage.
* **`/queue`**: Display current queue depth, pending jobs, and active prompt previews with visual VRAM progress bars.
* **`/variation_mode`**: Toggle persistent default variation strength between **High** (0.85 denoise) and **Very High** (0.95 denoise).
* **`/diagnostics`**: View detailed telemetry benchmarks, average render times by generator (initialization, sampling, post-processing breakdown), success rates, and recent runs.

---

## 🪄 Prompting Flags & Superpowers

| Flag / Syntax | What It Does | Example |
| :--- | :--- | :--- |
| `{a\|b\|c}` | **Dynamic Prompt / Wildcards**: Evaluates a distinct random choice for each quadrant in the 2x2 grid. | `/imagine prompt: a cute {cat\|fox\|dragon\|panda} wizard --ar 16:9` |
| `--smart` / `--sm` | **Smart Art Director**: Analyzes prompt genre (Cyberpunk, Fantasy, Cozy, Portrait, Retro 80s) and injects harmonized prompt expansion & matching `--sref` code. | `/imagine prompt: cyberpunk samurai in the rain --smart` |
| `--magic` / `--mp` | **Magic Prompt**: Injects cinematic volumetric lighting, 8k resolution, and fine studio details. | `/imagine prompt: crystal potion bottle on an altar --magic` |
| `--ar W:H` | **Aspect Ratio**: Formats canvas shape (`16:9`, `21:9`, `1920:1032`, `10:7`, `3:5`, `9:16`, `1:1`, etc.). | `/imagine prompt: neo tokyo skyline --ar 21:9` |
| `--s <0-1000>` | **Stylize / CFG Control**: Controls creativity vs prompt adherence (`--s 0` = literal / CFG 1.0, `--s 1000` = artistic / CFG 12.0). | `/imagine prompt: mystical enchanted forest --s 750` |
| `--raw` | **Raw Mode**: Disables automatic quality tags (`masterpiece, best quality`) and sets CFG to 3.0 for natural photos. | `/imagine prompt: vintage red pickup truck on dirt road --raw` |
| `--seed <num>` | **Fixed Seed**: Generates with a specific seed for reproducible outputs. | `/imagine prompt: space explorer on mars --seed 428192` |
| `--sref <code\|url\|random>` | **Style Reference**: Transfers visual aesthetic, lighting, and palette from an image, URL, random seed, or 6-digit code (4.2M+ combinations). | `/imagine prompt: neon warrior --sref 492104 --sw 0.85` |
| `--sw <0.0-1.0>` | **Style Weight**: Controls the strength of the `--sref` style reference (default `0.60`). | `/imagine prompt: knight on a hill --sw 0.90` |
| `--cref <url>` | **Character Reference**: Copies character identity and facial features via IP-Adapter. | `/imagine prompt: adventurer drinking tea --cref https://... --cw 0.70` |
| `--cw <0.0-1.0>` | **Character Reference Weight**: Controls strength of `--cref` character transfer (default `0.20`). | `/imagine prompt: hero in armor --cw 0.60` |
| `--sr.XX` / `--srXX` | **Semi-Realism LoRA**: Injects `Semi-realism_illustrious` LoRA at specified weight. | `/imagine prompt: anime sorceress --sr.85` |
| `--oga.XX` / `--ogarla` | **Ogarla Character LoRA**: Injects `ogarla_epoch_5` (or `ogarlaflux_epoch_1` for Flux). | `/imagine prompt: ogarla reading a spellbook --oga.70` |
| `<lora:name:weight>` | **Custom LoRA**: Dynamically injects any `.safetensors` model from your ComfyUI models directory. | `/imagine prompt: robot hero <lora:cyber_armor:0.8>` |

---

## 🎛️ Interactive Button Systems

### 1. Grid Buttons (2x2 Output)
```text
[ U1 ]  [ U2 ]  [ U3 ]  [ U4 ]
[ V1 ]  [ V2 ]  [ V3 ]  [ V4 ]  [ 🔄 ]
[ ⭐ Favorite Style ]  [ ⭐ Favorite Prompt ]  [ 📋 Copy Prompt ]
```
* **`U1–U4` (Isolate & AI Detail Upscale)**: Extracts the selected quadrant and runs an AI detail reprocessing pass at 2x resolution (`denoise 0.55`).
* **`V1–V4` (Visual Img2Img Variations)**: Generates 4 new variations of the selected quadrant.
* **`🔄` (Re-roll)**: Re-runs the exact prompt and settings with fresh seeds (and new `{a|b|c}` wildcard picks).
* **`⭐ Favorite Style / Prompt`**: Saves the active style code or prompt to your personal library.
* **`📋 Copy Prompt`**: Displays the full, un-truncated prompt in a private codeblock.

### 2. Isolated & Upscaled Image Buttons
* **`⚡ Detailed Upscale (1.25x)` / `⚡ Creative Upscale (1.5x)`**: High-resolution detail enhancement.
* **`⚡ Vary Upscale Details`**: Re-runs upscale with alternate seed details.
* **`🎨 Vary (Subtle)` / `🎨 Vary (Strong)`**: Generates slight or strong visual variations of the isolated image.
* **`🎨 Custom --sref`**: Opens a modal to apply any `--sref` code or URL while preserving the exact prompt and seed.
* **`🎲 Random --sref`**: Immediately applies a fresh random style preset.
* **`⭐ Saved --sref`**: Dropdown selector of your saved favorite styles.

### 3. Generation Stasis & Message Management
* **`⏸️ Pause / Stasis` & `▶️ Resume`**: Pause active or queued generation jobs in ComfyUI and resume them whenever ready.
* **`❌` Message Deletion**: React with the `❌` (or `:x:`) emoji on any bot message to delete it instantly (authorized for requester or moderators).

---

## 📂 File Storage & Output Directory Structure

Isolated images, icons, and datasets are automatically saved to your local drive:

* 🖼️ **SDXL High-Res & Isolated Images:** `C:\ComfyUI\ComfyUI\output\Discord Bot\highres` (or dated folders)
* ✨ **Flux1-Dev Images:** `C:\ComfyUI\ComfyUI\output\Discord Bot\flux`
* 🎯 **Windows Icons (`.ico`):** `C:\ComfyUI\ComfyUI\output\Discord Bot\ico`
* 🧬 **LoRA Datasets & ZIPs:** `datasets/<session_id>/`
* 🗄️ **Persistent Cache & Database:** `cache.db` (SQLite)
* 📁 **Temporary Quadrants:** `C:/ComfyUI/ComfyUI/output/Discord Bot/scratch/` (Configurable via `QUADRANT_CACHE_DIR` in `.env`)

---

## 🛠️ Codebase Architecture & Module Map

* **[bot.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/bot.py)**: Main Discord bot entry point, command tree registrations, task dispatchers, presence updates, interaction handlers, and robust fallback delivery dispatchers (`send_followup_fallback`, `edit_original_fallback`, `handle_copy_prompt`).
* **[parsers.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/parsers.py)**: Precompiled regex engines for prompt flags (`--ar`, `--sref`, `--cref`, `--sr`, `--ogarla`, `--magic`, `--smart`, `--s`, `--raw`, `--seed`), dynamic `{a|b|c}` wildcard expansion, locked style presets, Face Detailer injection, static LoRA wiring, and multi-format PNG metadata extractors.
* **[image_utils.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/image_utils.py)**: Pillow image utilities, 2x2 grid assembly, quadrant cropping, multi-resolution `.ico` generation with squircle corners, standardized file naming, and PNG metadata injection.
* **[views.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/views.py)**: Persistent Discord UI views, buttons, dropdown selects, and popup modals (`GridButtons`, `IsolatedImageButtons`, `UpscaleButtons`, `DescribeButtons`, `BlendButtons`, `StudyButtons`, `VideoPromptModal`, `StylePaginationView`, `PromptPaginationView`, `AdoptButtons`, `LoraBuildGridButtons`, `LoraBuildStatusView`, `StasisControlsView`).
* **[db.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/db.py)**: High-performance SQLite database layer (`cache.db`) with Write-Ahead Logging (`WAL`), indexed user queries, generation caching, favorite prompts/styles, LoRA dataset sessions, telemetry metrics, and automatic pruning.
* **[lora_dataset.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/lora_dataset.py)**: Character LoRA dataset engine, shot matrices (anime Danbooru vs realistic), automated Florence-2 batch captioning, and Kohya_ss ZIP export formatting.
* **[error_handler.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/error_handler.py)**: Centralized error journaling (`error_log.json`) and automated remediation recipes for VRAM out-of-memory errors and server disconnects.
* **[comfy_client.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/comfy_client.py)**: Asynchronous REST & WebSocket client managing ComfyUI job queues, execution tracking, bounded timing caches, stasis pausing, image uploads, and output retrieval.
* **[monitor.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/monitor.py)**: Standalone terminal dashboard displaying live GPU VRAM usage and ComfyUI queue states.
* **[suite_test.py](file:///c:/Users/strot/Antigravity%20IDE/cui-server-bot/suite_test.py)**: Comprehensive 38-module automated unit and integration test suite executing prior to bot startup.

