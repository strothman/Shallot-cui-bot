# 🧅 PROJECT STATE — Shallot-CUI Bot

> **Project Name:** Shallot-CUI Bot (*Your Discord AI Creation Studio*)  
> **Current Version:** `v2.6.6`  
> **Last Updated:** September 13, 2026  
> **Status:** 🟢 Stable & Healthy (88/88 Automated Tests Passing)  

---

## 📌 1. What Is Shallot-CUI Bot? (In Plain English)

Think of **Shallot-CUI Bot** as your own private version of Midjourney that runs right inside your Discord server, powered by your computer's graphics card!

* **How it works:**
  1. You type a command in Discord, like `/imagine prompt: a cozy cottage in autumn woods --ar 16:9`.
  2. The bot translates your text and sends it to **ComfyUI** (the AI art generator running on your PC).
  3. Your graphics card paints the pictures.
  4. The bot posts a neat **4-image grid** back to Discord, with easy buttons to upscale, create variations, or remix your prompt!

You don't need to know any programming or complex AI jargon to use it. The bot handles all the complicated math and settings behind the scenes.

---

## 🏗️ 2. Project Files & What They Do

Here is a simple breakdown of the main files in the project and what each one is responsible for:

| File | What It Does (Plain English) |
| :--- | :--- |
| [`bot.py`](bot.py) | **The Front Desk:** Listens to Discord messages, handles slash commands (`/imagine`, `/flux`, `/video`), and coordinates tasks. |
| [`services/recovery_service.py`](services/recovery_service.py) | **Crash Recovery & Reconciliation:** Automatically rescues interrupted generations on restart, retrieves completed outputs from ComfyUI history, and delivers them to Discord. |
| [`services/workflow_adapter.py`](services/workflow_adapter.py) | **Semantic Workflow Adapter:** Insulates the bot from ComfyUI node renumbering by manipulating nodes by class type, title, and inputs rather than hardcoded IDs. |
| [`services/engine_queue.py`](services/engine_queue.py) | **Engine-Aware Priority Queue:** Prevents VRAM thrashing on 8GB GPUs via model affinity, anti-starvation age escalation, and auto VRAM purging. |
| [`cogs/vision_cog.py`](cogs/vision_cog.py) | **Vision Desk (Modular Cog):** Houses `/blend-sdxl`, `/blend`, and context menus in a clean modular cog. |
| [`services/vision_service.py`](services/vision_service.py) | **Vision Service:** Executes Florence-2 interrogation, prompt synthesis, and blend preparation. |
| [`cogs/krea_cog.py`](cogs/krea_cog.py) | **Krea Desk (Modular Cog):** Houses `/bertflow` and `/blend-krea` commands along with character, celebrity, and favorite prompt autocompletes. |
| [`services/krea_service.py`](services/krea_service.py) | **Krea Service:** Executes Bertflow flow-matching generation, interactive button callbacks, upscale, and blend studio setup. |
| [`characters.py`](characters.py) | **Character Wardrobe:** Stores character presets like **Cheri**, **Mageill**, **Valerie**, **Sully**, and **Ogarla**. Automatically applies character triggers/traits and protects real-person privacy. |
| [`parsers.py`](parsers.py) | **Prompt Translator:** Reads flags like `--ar 16:9` (widescreen), `--smart` (auto-lighting), `--sref` (style copy), and wildcards `{a\|b\|c}`. |
| [`views.py`](views.py) | **Interactive Buttons:** Creates all clickable buttons in Discord (U1–U4, V1–V4, `🛑 Cancel`, unified Blend Studio, and `✏️ Remix` popup windows). |
| [`comfy_client.py`](comfy_client.py) | **The Messenger:** Talks to ComfyUI on your computer, tracks render progress, journals active jobs, and automatically frees GPU memory when needed. |
| [`image_utils.py`](image_utils.py) | **Image Crafter:** Stitches the 4 pictures into a 2x2 grid, cuts out individual images for upscaling, and optimizes file sizes asynchronously. |
| [`db.py`](db.py) | **Memory & Notebook:** An SQLite database (`cache.db`) with WAL mode that journals active jobs, stores favorite prompts, and tracks generation metrics. |
| [`config.py`](config.py) | **Settings & Guardrails:** Stores default models, safety limits, and admin permissions so only server owners can run sensitive controls. |
| [`suite_test.py`](suite_test.py) | **Safety Inspector:** An automated test runner that checks 85 different parts of the bot to make sure nothing is broken. |
| [`auto_changelog.py`](auto_changelog.py) | **Secretary:** Keeps the [CHANGELOG.md](CHANGELOG.md) updated so you always know what was added or changed. |
| [`workflows/`](workflows/) | **Recipe Book:** Pre-built ComfyUI recipes for SDXL, Flux.1, Wan 2.2 video, and high-resolution upscaling. |

---

## ⚡ 3. Current Features & Commands

### 🎨 Image Generation
* **`/imagine`**: Creates a 2x2 grid of 4 pictures using SDXL. Supports aspect ratios (`--ar`), style references (`--sref`), and character presets.
* **`/flux`**: Creates ultra-detailed, photographic pictures using the next-generation **Flux.1** AI model.
* **`/blend-sdxl`**: Dedicated 100% SDXL Blend Studio. Stripped of bloat: single `image` input, zero-tab unified 1-page dashboard, 1-click toggles (Semi-Realism & `--sref random`), 1-click aspect ratio cycle, and instant Florence-2 vision interrogation (~1.5s).
* **`/blend-krea`**: Dedicated Krea 2 Photorealism Studio. Streamlined to single `image` upload; all settings (AR, Direct Composition, Character, Celebrity, Engine, and Wetness) managed via the interactive Phase 2 studio dashboard.

### 🎭 Character Presets (Consistent Faces)
* **Cheri (`--cheri`)**: Character preset (Epoch 6 default, supports `--cheri4`). Automatically injects signature blonde hair!
* **Mageill (`--mageill`)**: Original character preset (Epoch 5 default). Easily switch training checkpoints with `--mageill3`, `--mageill4`, `--mageill5`, or `--mageill6`.
* **Valerie (`--valerie.85`)**: Keeps the Valerie character look consistent across prompts.
* **Sully (`--sully.85`)**: Automatically adds Sully's signature black hair and thin-rim glasses.
* **Ogarla (`--ogarla.85`)**: Fantasy character preset.
* 🛡️ **Privacy Shield:** Characters based on real persons automatically disguise private trigger names so real identities are never exposed in Discord.

### 🎬 Video & Animation
* **`/video`**: Turns any still picture into an animated video using **Wan 2.2** (14B GGUF + RIFE frame interpolation). Fast, lightweight, and rock-solid on 8GB VRAM GPUs.
* **`/ltx`**: Creates a fast 35-second animation using **LTX-Video**.

### 🔍 Vision & Image Tools
* **`/describe`**: Upload any picture and Florence-2 AI will generate captions, detailed descriptions, and a tailored Krea 2 prompt with a 1-click **Generate Krea 2** button.
* **`/blend`**: Image Blend Studio to mix pictures with new styles, models, and text prompts.
* **`/blend-krea`**: Dedicated Krea 2 Photorealism Blend Studio with Florence-2 vision fusion, optional Direct Composition locking (VAE latent), and 1-click Bertflow generation.
* **`/study`**: Upload an AI image found online to extract the secret prompt used to make it.
* **`/upscale`**: Makes any picture bigger and sharper with extra detail.

### 🎛️ Interactive Discord Buttons
Under every 4-image grid, you get 1-click buttons:
* **`[ U1 ] [ U2 ] [ U3 ] [ U4 ]`**: Isolate and upscale picture #1, #2, #3, or #4 to full size.
* **`[ V1 ] [ V2 ] [ V3 ] [ V4 ]`**: Make 4 new variations based on that specific picture.
* **`[ 🔄 ]`**: Re-roll the exact same prompt with fresh random seeds.
* **`[ ✏️ Remix ]`**: Opens a popup with your prompt pre-filled so you can easily change a few words and try again.
* **`[ 🛑 Cancel ]`**: Instantly stop a running generation if you change your mind.

### ⚙️ Server Controls
* **`/cui-start` & `/cui-stop`**: Turn your local ComfyUI engine on or off directly from Discord.
* **`/cui-status`**: Check if ComfyUI is online and see how much graphics card memory (VRAM) is free.
* **`/models` & `/scan_models`**: View all installed AI models and scan for newly downloaded ones with 1 click.
* **`/queue`**: Check what jobs are currently rendering.

---

## 🧬 4. Character LoRA Dataset Creator (From 1–12 Reference Photos)

If you only have **1 to 12 reference photos** of a person or character (and no existing Flux LoRA), you can use the **Synthetic Dataset Expansion Tool** to create a complete, high-quality, 30-image training dataset for Krea 2:

### How It Works:
1. **Drop Reference Images**: Place 1 to 12 clear photos (selfies, portraits, candids) into `inputs/reference_character/` (or specify a custom folder).
2. **Identity Extraction**: The tool loads `IP-Adapter FaceID Plus v2` (`ip-adapter-plus_sdxl_vit-h.safetensors` / InsightFace) in ComfyUI to lock the person's exact facial structure and identity.
3. **High-Fashion Modeling Prompt Matrix**: Automatically scripts a balanced, publication-grade 30-image modeling dataset:
   * **~25% High-Fashion Swimwear**: Bikinis, luxury monokinis, beach resort and poolside lighting, full-body head-to-toe shots.
   * **~25% Glamour Couture & Evening Dresses**: Satin bodycon dresses, backless cocktail gowns, runway poses, and studio softbox lighting.
   * **~20% Athletic & Fitness Apparel**: Sports bras, high-waisted leggings, athletic postures, gym and contemporary studio environments.
   * **~15% Streetwear Chic**: Tailored blazers over crop tops, leather jackets, Parisian street fashion strides.
   * **~15% High-End Beauty Portraits**: Flawless skin texture, expressive catchlights, studio ring light and chiaroscuro editorial lighting.
   * *Anti-Bleed IP-Adapter Tuning:* Configured with `end_at: 0.65` and wardrobe negative prompts so reference sweaters/casual clothing do not bleed into fashion outfits.
4. **Auto-Captioning via Florence-2**: Each generated picture is automatically analyzed by Florence-2, formatted with the trigger word (e.g. `samantha, a photo of...`), and saved as paired `.png` and `.txt` files.
5. **AI-Toolkit Config Generation**: Produces a ready-to-run `.yaml` file for training your Krea 2 LoRA at 1024x1024.

### ⏯️ Resume & Job Continuation (If You Run Out of Data / Interrupt):
The tool is built with **automatic state resumption**:
* If generation is interrupted, or if you close the terminal, you can resume at any time simply by re-running the script or batch file!
* The script scans the output directory (`datasets/<trigger>_krea2/`), detects all existing completed `<trigger>_XXX.png` and `.txt` pairs, and seamlessly starts generating the remaining samples starting from the next index.
* You never lose previously generated images or waste GPU time.

### How to Run:
* **One-Click Batch**: Double-click **`generate_dataset_from_photos.bat`**.
* **Terminal**:
  ```bash
  python tools/create_character_dataset_from_photos.py --input_dir inputs/reference_character --trigger mychar --count 30
  ```

---

## 🌟 5. What's New in v2.4.7 & Latest Updates

1. **Centralized Character Display Badges (`characters.get_character_display_badge`)**: Unified character badge formatting across all embed builders (`/describe`, `/blend`, `/blend-krea`, and prompt refine views). Replaced over 80 lines of duplicate manual mappings with a single source of truth.
2. **Silent Singleton Lock for Automated Tests**: Added `silent: bool = False` to `acquire_instance_lock` in `bot.py` and `suite_test.py`. Unit testing collision detection now executes cleanly without false-alarm console warning banners.
3. **Valerie Krea 2 Full System Integration**: Wired Valerie into `/bertflow` and `/blend-krea` choices, interactive select menus, session embeds, and smart toggle retention.
4. **Dual-Engine Synthetic Dataset Builder (`tools/build_character_dataset.py`)**: Seamlessly supports both Flux GGUF and SDXL character LoRAs with auto-detection. Generated a diverse 30-sample 1024x1024 Krea 2 training dataset for Valerie (`datasets/valerie_krea2/` & `valerie_krea2.zip`) with Florence-2 auto-captioning and AI-Toolkit config. Includes 1-click batch runner `generate_valerie_dataset.bat`.
5. **Character Dataset Builder from Photos (`tools/create_character_dataset_from_photos.py`)**: Generate full 30-sample training datasets from just 1–12 reference photos with IP-Adapter identity locking, automated prompt matrix, Florence-2 auto-captioning, and seamless resume support.
6. **`/blend-krea` Direct Composition Dropdown**: Converted the cycling button into a dedicated 1-click select menu (`Off`, `Subtle`, `Medium`, `Strong`) with intuitive pose-retention labels.
7. **Streamlined `/blend-krea` Embed**: Removed duplicate walls of text; the fused prompt is only displayed when user remix additions are present, keeping initial sessions clean and readable.
8. **Bertflow Duplicate Re-Roll Fix**: Eliminated double-triggering on `BertflowButtons` by channeling actions exclusively through `bot.py`'s persistent interaction handler.
9. **Automatic Memory Cleaning (VRAM Auto-Purge)**: The bot automatically frees graphics card memory when switching between SDXL, Flux, and video models so your computer never crashes from low memory.
10. **Non-Blocking Async Image I/O & Gateway Protection**: Offloaded all CPU-heavy PIL transformations (Lanczos isolation, outpaint padding calculation, grid cropping, vibrancy boosting, 1.5x upscaling, and disk reads/writes) to worker threads via `asyncio.to_thread()`. Added concurrent `asyncio.gather()` processing for quadrant image enhancements. Discord gateway heartbeats and button response times are completely protected from event loop stalls.
11. **Engine-Aware Priority Queue & VRAM Thrashing Prevention (`services/engine_queue.py`)**: Implemented intelligent model-affinity job batching across SDXL, Flux.1, Krea 2, Wan 2.2, LTX, and Florence-2. Groups pending jobs targeting the active architecture to eliminate PCIe model weight swapping (saving 20–45s per switch), automatically clears GPU VRAM during transitions, and incorporates anti-starvation age escalation with an upgraded `/queue` dashboard.
12. **Semantic Workflow Adapter & Node Decoupling (`services/workflow_adapter.py`)**: Replaced brittle hardcoded numeric node IDs (`wf["3"]`, `wf["5"]`, `wf["6"]`, `wf["75"]`, `wf["76"]`, `wf["822"]`) with semantic discovery based on class types, titles, parameter signatures, and graph link tracing. Insulates the bot from GUI renumbering, verified through randomized node-scrambling tests.
13. **Centralized Pipeline Defaults & Semantic Constants (`config.py` & `v2.6.6`)**: Consolidated generation checkpoints, upscale denoise floats (`UPSCALE_DENOISE_SDXL = 0.55`, `UPSCALE_DENOISE_FLUX_SUBTLE = 0.26`, `UPSCALE_DENOISE_FLUX_MODERATE = 0.35`), and variation similarity profiles into `class PipelineDefaults`. Added `get_checkpoint_display_name()` with shorthand alias resolution, eliminating duplicate dictionaries in `views.py`, and replaced raw magic floats in `bot.py`.

### 🛡️ Core Architecture Rules (Mandatory for Future Development)
* **Async Event Loop Hygiene (Rule 1 in [`AGENTS.md`](AGENTS.md)):** Never run blocking PIL operations or synchronous disk I/O on the primary asyncio event loop. All image crops, upscales, grid stitches, and file saves must use the non-blocking `*_async` functions in [`image_utils.py`](image_utils.py) or `asyncio.to_thread()`.
* **Modular Cog & Service Architecture (Rule 2 in [`AGENTS.md`](AGENTS.md)):** Keep `bot.py` clean; place Discord UI and slash commands in `cogs/` and pure business logic in `services/`.
* **Single-Source Style & Character Registry (Rule 3 in [`AGENTS.md`](AGENTS.md)):** Register all character LoRAs, triggers, and privacy filters in [`characters.py`](characters.py). All automated tests in [`suite_test.py`](suite_test.py) must pass 100% green before completing changes.

---

## 🗺️ Architectural Roadmap

* **Phase 1: Zero-Risk Refinements [COMPLETED]**
  * Centralized character display badge resolution in `characters.py`.
  * Silenced test-induced multi-instance warning banner.
* **Phase 2: Dynamic Character Autocomplete [COMPLETED]**
  * Replaced static `app_commands.choices` arrays with dynamic `app_commands.autocomplete` querying `characters.py` and `scan_krea2_loras()`. Newly added LoRAs dropped into models folders immediately appear in Discord autocomplete without code edits or command re-syncing.
* **Phase 3: Modular Cog Architecture [PILOT COMPLETED — EXPAND WHEN READY]**
  * ✅ **Pilot Phase Verified:** Successfully extracted `/blend-sdxl`, `/blend` alias, and `Blend Image (SDXL)` context menu into [`cogs/vision_cog.py`](cogs/vision_cog.py) and pure execution logic into [`services/vision_service.py`](services/vision_service.py), eliminating 468 lines from `bot.py` with 100% backward compatibility and 71/71 tests passing.
  * 📋 **Remaining Roadmap:** Decompose remaining modules (`cogs/krea_cog.py`, `cogs/video_cog.py`, `cogs/imagine_cog.py`, `cogs/admin_cog.py`) whenever ready. See [`docs/modular_cog_architecture_plan.md`](docs/modular_cog_architecture_plan.md) for execution blueprint.
* **Phase 4: Client Distribution & Custom Hardware Packaging [FUTURE ROADMAP — TARGET: NEXT MONTH / NOT NOW]**
  * Package Shallot-CUI Bot for standalone deployment on an external user's PC with their own ComfyUI server, custom checkpoints, and custom LoRAs.
  * 🟢 **Feasibility Rating:** Highly Feasible (8.5/10). Because the bot communicates via standard ComfyUI REST/WebSocket APIs (`127.0.0.1:8188`), it is already decoupled from local hardware.
  * ⏳ **Timeline Note:** This is scheduled for future implementation next month. The feasibility plan is preserved in the docs archive for when we are ready to build it.
  * **Core Deliverables Required:**
    1. **Dynamic Model Discovery:** Query ComfyUI `/object_info` at startup to populate Discord autocomplete directly with the user's installed checkpoints and LoRAs, replacing hardcoded lists.
    2. **Pre-Flight Environment Validator (`tools/check_comfy_env.py`):** Automatically verifies ComfyUI connectivity, core nodes, and extension nodes (Florence-2, VideoHelperSuite, GGUF) and provides 1-click install links for missing components.
    3. **Turnkey Installer Package:** 1-click `setup.bat` (creates isolated `.venv` and installs dependencies), `run_bot.bat` launcher, and a friendly 5-minute setup guide (`README_FRIEND.md`).
  * 📋 **Detailed Feasibility Plan:** See [`docs/client_distribution_packaging_plan.md`](docs/client_distribution_packaging_plan.md) for full gap analysis, architecture adapters, and packaging roadmap.

* **Phase 5: Technical Debt & Performance Optimizations [ACTIVE ROADMAP & AUDIT]**
  * Preserved full architectural gap analysis covering the 6,100-line `bot.py` monolith, unchecked quadrant scratch cache accumulation, monolithic `parsers.py`, sequential media downloading, and queue concurrency limits.
  * 📋 **Detailed Audit & Solutions:** See [`docs/technical_debt_and_optimization_audit.md`](docs/technical_debt_and_optimization_audit.md) for full issue breakdowns, risk analyses, and implementation blueprints.

---

## 🧪 6. Testing & Quality Assurance

Every time you run the bot using `run_bot.bat`, it performs an automatic safety check:
* **Automated Tests:** **88 / 88 tests passing** (`python suite_test.py`).
* **What is tested:**
  * Aspect ratio math and sizing.
  * Wildcard randomization (`{cat|dog|fox}`).
  * Character preset trigger substitution and privacy masking.
  * Interactive buttons and Remix popup modals.
  * Model compatibility and graphics card memory cleanup.
  * Documentation sync between Discord commands and README.

---

## 🚀 7. How to Start Everything

1. Make sure your `.env` file has your `DISCORD_TOKEN`.
2. Start ComfyUI on your computer (or run `/cui-start` in Discord).
3. Double-click **`run_bot.bat`**.
4. Head into Discord and type `/imagine` or `/blend-krea` to start creating! 🎨
