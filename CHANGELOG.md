# Changelog

All notable changes to **Shallot-CUI Bot** will be documented in this file.

---

## [2026-09-10]

### Added
* 🧬 **Character Dataset Generator from Reference Photos (`tools/create_character_dataset_from_photos.py`)**: Built an automated identity-expansion dataset generator allowing users to create full 30-image Krea 2 LoRA training datasets from just 1 to 12 reference photos using IP-Adapter identity projection, a balanced prompt matrix (full-body, medium, portrait), Florence-2 auto-captioning, and seamless resume/job continuation logic. Includes 1-click batch launcher `generate_dataset_from_photos.bat`.
* 👗 **High-Fashion & Modeling Prompt Matrix**: Scripted dedicated modeling categories (~25% Swimwear, ~25% Glamour Couture/Evening Dresses, ~20% Athletic Activewear, ~15% Streetwear Chic, ~15% Beauty Portraits) to guarantee publication-grade versatility for LoRA training. Tuned IP-Adapter weights and steps (`end_at: 0.65`, dynamic weights) and added clothing-suppression negatives to prevent reference sweater/wardrobe bleed into modeling shots.
* 🖼️ **`/blend-krea` Direct Composition Dropdown**: Replaced the cycling composition button with a dedicated 1-click dropdown select menu (`set_blend_krea_comp`), making all 4 pose-locking modes (`Off`, `Subtle`, `Medium`, `Strong`) immediately selectable with clear, intuitive labels.
* 📜 **Streamlined `/blend-krea` Embed**: Removed redundant duplicate prompt displays; initial sessions show Florence-2 vision analysis once, displaying the fused generation prompt only when additive remix instructions are provided.

### Fixed
* 🐛 **Bertflow Duplicate Re-Roll Queuing**: Fixed an issue where clicking `[ 🔄 Re-roll ]` on Bertflow images dispatched two duplicate generation tasks to ComfyUI. Removed redundant view callbacks from `BertflowButtons` so actions are handled exclusively and persistently through `bot.py`'s `on_interaction` handler.
* 🔄 **Direct Composition Order Progression**: Reordered composition toggle states in `bot.py` and slash command choices from `Off -> Subtle (85% denoise) -> Medium (70% denoise) -> Strong (50% denoise)` to ensure logical progressive strength.

---
## [2026-09-09]

### Added
* 📸 **`/bertflow` Photorealism Slash Command**: Added dedicated `/bertflow` slash command implementing Bert's 11-node Krea 2 Turbo flow-matching pipeline:
  * **Zero Glossy AI Skin**: Leverages `wetness_krea2_loraholic` with `-2.0` negative slider strength to actively strip oily plastic sheen in favor of natural matte skin pores and believable lighting.
  * **Turbo Flow-Matching**: Fast 8-step generation with CFG 1.0, Euler / Simple scheduler, and `ConditioningZeroOut` flow-matching negative setup.
  * **Dual Model Support**: Auto-detects and supports both `Muse v3.5 Extended` (Stable Yogi) and `Pornmaster v2` (Krea 2 FP8).
  * **Resolution Presets**: Supports 1:1 (`1224x1224` native), 16:9, 9:16, 21:9, 3:4, 4:3, and 16:9.3, plus custom `--ar` flag parsing.
  * **Interactive Action Controls (`BertflowButtons`)**: Interactive `[ 🔄 Re-roll ]` with fresh seed and `[ ✏️ Remix ]` modal to tweak prompt in-place.
* 🧬 **Automated Character Dataset Generator (`tools/build_character_dataset.py`)**: Built an end-to-end synthetic dataset generator and 1-click batch runner (`generate_ogarla_dataset.bat`) to synthetically generate diverse Flux character portraits and auto-caption them with Florence-2 for training on new architectures (e.g., Krea 2 / OneTrainer).
* 🛠️ **ComfyClient Output Compatibility Fix**: Updated `execute_bertflow` and dataset generator to seamlessly handle both raw byte lists and dictionary node output maps from `ComfyClient.generate()`.
* 🐛 **Fix Bertflow Progress Callback & Timing**: Fixed keyword argument mismatch (`on_progress` -> `progress_callback`) and timing breakdown retrieval in `execute_bertflow`, adding `on_progress` alias support to `ComfyClient.generate()` for cross-caller resilience.
* 🎛️ **Interactive Video Action Controls (`VideoActionView`)**: Added interactive Discord button controls below completed `/video` generations:
  * **`[ 🔄 Re-roll ]`**: Re-animates the exact same uploaded image with a fresh random seed.
  * **`[ ✏️ Remix Motion ]`**: Reopens `VideoPromptModal` with previous prompt, duration (5s/10s), and smoothness mode pre-filled for rapid prompt iteration.
  * **`[ ⚡ Fast / Smooth Toggle ]`**: Instantly switch between Ultra-Fast (16 FPS native) and Smooth (32 FPS RIFE) without re-uploading the image.
* 🎥 **Camera Motion Directives & Prompt Flags**: Added `parse_video_motion_flags` in `parsers.py` supporting:
  * Zoom: `--zoom`, `--zoom-in`, `--zoom-out`
  * Pan: `--pan-left`, `--pan-right`, `--pan-up`, `--tilt-up`, `--pan-down`, `--tilt-down`
  * Orbit: `--orbit`, `--rotate`
  * Dynamics: `--cinematic`, `--subtle`, `--dynamic`, `--fast-motion`
* 📊 **3-Column Video Studio Dashboard Embed**: Upgraded video completions with a structured inline 3-column dashboard (`🎬 Motion & Camera`, `⏱️ Video Specs`, `⚡ Engine & Render`) featuring camera directive badges, framing mapping, render timing breakdown, and `#5865F2` theme color.
* ⚡ **Zero-Scroll In-Place Message Delivery**: Progress status messages now transform in-place directly into the completed video post, eliminating channel jumps and duplicate notifications.
* 🧹 **Post-Generation VRAM Flush**: Automatically purges GPU VRAM cache immediately after video rendering to return 8GB GPUs to a clean idle state.

### Changed
* 🚀 **Decoupled MMAudio from `/video` for 8GB VRAM Stability**: Completely removed the heavy MMAudio Foley synthesis stack (~5.1 GB model footprint) from the `/video` workflow. Pervasive VRAM thrashing and PCIe shared memory paging on 8GB GPUs (e.g., RTX 5060 Ti) are fully eliminated.
* 🎬 **Streamlined `/video` Slash Command**: Simplified `/video` parameters by removing unused `audio` and `audio_prompt` inputs and embed fields, returning Wan 2.2 Image-to-Video to a fast, rock-solid, and lightweight animation pipeline.
* 🧹 **Settings & Workflow Pruning**: Cleaned `settings.json` (`enable_video_audio: false`, removed unused `mmaudio_*` model keys) and pruned nodes `150`, `151`, and `152` from `workflows/wan22_i2v.json`.

### Maintenance
* 🧪 **Automated Test Suite Expansion**: Added unit tests `test_module42_video_motion_flags` and `test_module43_video_dashboard_and_action_view` in `suite_test.py` covering motion flags, dashboard layouts, `VideoActionView`, and `VideoPromptModal` flexible defaults. All 61 unit tests pass cleanly.

---

## [2026-09-08]

### Added
* 👑 **Ultimate Quality Enhancement Preset**: Added `👑 Ultimate Quality (Powerhouse 1.35x + Smart Director + Magic)` to the `/imagine` enhancements dropdown, uniting two-stage 1.35x resolution refinement, FreeU dynamics, subject-harmonized art direction, and cinematic lighting into a single one-click preset.
* ⚡ **Pipeline Prompt Flag Parsers**: Added direct prompt flag parsers for `--powerhouse`, `--ph`, and `--refine` (activating the SDXL 2-stage refiner pipeline) and `--raw`, `--nofreeu`, `--no-freeu`, and `--disable-freeu` (bypassing FreeU for pure checkpoint sampling).
* 🌟 **Synchronized Studio Presets**: Realigned SDXL and Flux enhancement dropdown choices in both `bot.py` and `config.py` with clean, intuitive labels (`👑 Ultimate Quality`, `🌟 Studio Duo`, `⚡ 2-Stage Powerhouse`, `✨ Magic Prompt`, `🧠 Smart Art Director`, and `🚫 Pure Checkpoint`).

### Changed
* 🎯 **Streamlined `/imagine` Enhancements Description**: Refreshed parameter help text to accurately reflect studio and pipeline presets, removing misleading legacy text.

### Fixed
* 🛡️ **Progress Message Channel Spam Prevention**: Enhanced `edit_message_fallback` in [bot.py](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py) and [core_helpers.py](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/core_helpers.py) with an `allow_send_fallback` flag set to `False` on intermediate progress callbacks. If an interaction token expires (>15 min) or a status message is older than 1 hour (Discord 30046), progress updates now quietly suppress channel-send fallbacks instead of flooding the channel with new percentage bar messages.
* 🧹 **Model Checkpoint Cleanup**: Permanently deleted and safely unlinked `lustifySDXLNSFWSFW_v10` and `bigLust_v16.safetensors` from all dropdown choices, configuration dictionaries, database registries, and architecture detectors.

### Maintenance
* 🧪 **Automated Test Suite Expansion**: Added unit tests in `suite_test.py` verifying the `ultimate` preset, Discord label character constraints, and all shorthand prompt flags (`--powerhouse`, `--ph`, `--refine`, `--raw`, `--nofreeu`, `--disable-freeu`). All 59 unit tests passing.

---
## [2026-09-07]

### Added
* 🎨 **Interactive `/blend` Character LoRA Dropdown**: Upgraded `/blend` with a dedicated Character LoRA dropdown supporting **Ogarla**, **Valerie**, **Sully**, **Cheri** (Epochs 4 & 6), and **Mageill** (Epochs 3, 4, 5, 6) with automatic trait injection and privacy masking.
* ✨ **Semi-Realism Level Dropdown in `/blend`**: Replaced the cycling toggle button with an intuitive dropdown to directly select `--sr` strength (`OFF`, `0.60`, `0.70`, `0.75`, `0.80`, `0.90`).
* 🎲 **Style & Sref Presets Dropdown in `/blend`**: Replaced the cycling random button with a comprehensive style dropdown featuring `--sref random` (1 style), `--sref batch` (5, 10, and 15 styles), and locked artist styles (Junji Ito, Martine Johanna, Dark Fantasy Landscape, Cyberpunk Cityscape, Ethereal Fine Art Portrait).
* 📑 **2-Tab Studio Layout for Discord 5-Row Compliance**: Created a 2-tab view (`📐 Canvas & Model` ↔ `🎭 Characters & Styles`) allowing all 6 major settings to have full dropdown menus while strictly adhering to Discord's 5-row component limit.
* 🖼️ **Polished 3-Column "Image Blend Complete" Dashboard**: Upgraded the blend completion embed from a raw text wall into a structured 3-column inline dashboard (`📐 Canvas & Framing`, `🤖 Checkpoint & Tech`, `🎭 Aesthetics & Identity`), complete with source image thumbnail, branded `#8A2BE2` violet accent bar, human-friendly model names, and execution timing footer.
* 🎛️ **"Adjust Blend" Action Button**: Added an `🎛️ Adjust Blend` button to the grid completion view that seamlessly reopens the interactive 2-tab Blend Studio for that session with all previous settings pre-loaded, removing the need to re-upload or re-analyze images.
* ⚡ **In-Place Blend Message Transformation**: Replaces intermediate status progress messages in-place with the completed grid and dashboard, keeping channels clean of orphan text.
* 🔍 **Streamlined "Blended Image x" Isolated Card**: Upgraded single isolated/upscaled blend results with a 3-column inline studio dashboard, original source thumbnail, and consolidated 3-row button controls (`⚡ Upscales & Vary` • `⭐ Favorites, ✏️ Remix & 🎛️ Adjust Blend` • `🎨 Style Sref`).

### Maintenance
* 🧪 **Automated Test Suite Expansion**: Expanded automated tests to 59 passing checks verifying Blend tabs, character flag injection, style presets, 3-column dashboards, and consolidated 3-row isolated image action controls.

---
## [2026-09-04]

### Added
* 🎀 **Cheri Character Preset with Blonde Hair Trait (`--cheri`)**: Added character preset for Cheri. Defaults to Epoch 6 (`cheri_epoch_6.safetensors`) with automatic injection of her signature `"blonde hair"`, and supports choosing Epoch 4 via `--cheri4` or from the Discord `/imagine` dropdown.
* 🔮 **Mageill Character Preset with Multi-Epoch Support (`--mageill`)**: Added original character preset for Mageill. Defaults to Epoch 5 (`mageill_epoch_5.safetensors`), and makes it easy to pick any training epoch via `--mageill3`, `--mageill4`, `--mageill5`, `--mageill6`, or directly from the Discord `/imagine` dropdown.
* 👓 **Sully Character Preset (`--sully`)**: Added the "Sully" character model (`susa_epoch_6.safetensors`). When you type `--sully` or pick Sully in the dropdown, the bot automatically adds her signature black hair and thin-rim glasses while keeping private trigger names hidden from Discord.
* ✨ **Valerie Character Preset (`--valerie`)**: Added the "Valerie" character model (`jen_epoch_5.safetensors`) with automatic trigger management and privacy protection.
* 🛑 **Live Cancel Button**: Added a `🛑 Cancel` button on active generation messages so you can stop a job in ComfyUI at any time if you change your mind.
* ✏️ **Grid Remix Modal (`✏️ Remix`)**: Added a Remix button under every 4-image grid that opens an easy popup window with your prompt and seed pre-filled so you can tweak words and re-roll.
* 🧹 **Automatic Graphics Card (VRAM) Cleaning**: The bot now automatically frees up graphics card memory whenever you switch between different model types (like SDXL and Flux) to prevent slowdowns.

### Changed
* 🚀 **Streamlined Command Menu**: Removed cluttered and rarely-used commands (like the experimental LoRA builder and icon creator) to make the bot faster, cleaner, and much easier to navigate.

### Maintenance
* 🧪 **56 Automated Tests**: Expanded our automated test suite to 56 passing checks to guarantee prompt formatting, character presets, and buttons work reliably.
* 📖 **Novice-Friendly Documentation**: Refreshed user guides and project summaries to make all concepts easy for beginners to understand.

---
## [2026-09-03]

### Added
* 🔒 **Double-Launch Warning**: Added a guard that alerts you if the bot is already running in another window, preventing double generations or duplicate button clicks.

### Maintenance
* 🧹 **Codebase Cleanup**: Removed old LoRA builder files and streamlined Discord interaction handlers.

---
## [2026-09-01]

### Changed
* 🧅 **Project Renaming**: Standardized the project and folder name to **Shallot-cui-bot** and updated all setup scripts.

---
## [2026-08-31]

### Maintenance
* ⚙️ **Prompt Parser Polish**: Improved prompt modifier parsing and style reference handling.

---
## [2026-08-30]

### Maintenance
* ⚙️ **Performance & Stability Tuning**: Optimized WebSocket message handling between Discord and ComfyUI.

---
## [2026-08-29]

### Maintenance
* ⚙️ **Model Architecture Checks**: Added automatic compatibility verification for loaded LoRA models.

---
## [2026-08-28]

### Maintenance
* ⚙️ **Discord View Enhancements**: Polished button rows and interactive components under image grids.

---
## [2026-08-27]

### Maintenance
* ⚙️ **Error Handling Improvements**: Improved auto-recovery logic for network disconnects.

---
## [2026-08-26]
### Added
* 🎬 **AI Sound Effects for Videos (Wan 2.2 + MMAudio)**: Added automatic sound effect generation for AI videos.
* 📊 **Real-Time Progress Bars**: Added progress bars in Discord so you can watch your pictures generate in real time.
* 🛡️ **Admin Safety Controls**: Added permissions so only bot owners and admins can run server controls or delete messages.
* 📜 **Automated Testing Suite**: Added pre-flight test runner to catch issues before starting the bot.
* 🚀 **Standalone Bot Release**: Released Shallot-CUI Bot v2.3.0 standalone setup with documentation and badges.

### Changed
* **fix(diagnostics): quote batch title and update changelog** (`90ca0cc`)
* **feat(ui): implement single in-place transforming message for zero-scroll generation progress** (`8cd3ea6`)
* **feat(progress): upgrade monitor.py with live WebSocket HUD and implement real-time Discord image generation progress bars** (`df9c912`)
* **feat(security): implement bot owner and admin authorization guards on server commands and message deletion** (`36d890a`)

### Maintenance
* **docs: update changelog with security authorization guard implementation details** (`1296a3d`)
* **docs: consolidate into single novice-friendly README.md** (`1819575`)
* **chore: update changelog to reflect Shallot-CUI Bot v2.3.0 initialization and updated ignore rules** (`c567c7b`)
* **chore: expand .gitignore with safety rules** (`c8ddbc6`)
* Working tree modification: `M bot.py`
* Working tree modification: `M comfy_client.py`
* Working tree modification: `M db.py`
* Working tree modification: `M monitor.py`
* Working tree modification: `M suite_test.py`

---
## [2.3.0] - 2026-08-26

This release introduces the **Model & LoRA Architecture Classification System**, **SQLite Architecture Registry**, **LoRA Variant Auto-Routing**, and **Pre-Execution Compatibility Validation Guards**.

### Added
* **Model & LoRA Architecture Classification (`model_architecture.py`)**:
  * Added zero-overhead Safetensors header parsing (`read_safetensors_header`) to extract model architectures, subtypes, and training metadata without loading heavy model weights into memory.
  * Formalized architecture taxonomy covering **SDXL** (Illustrious, Pony, Realistic, Standard), **Flux**, **SD 1.5**, **SD 3.5**, **Wan 2.1/2.2** (High Noise, Low Noise), **LTX-Video**, and **Hunyuan**.
  * Added UI architecture badges (`🎨 [SDXL]`, `⚡ [FLUX]`, `🎬 [WAN]`, `🎥 [LTX]`) for Discord embeds and autocomplete dropdowns.
* **SQLite Model & LoRA Registry (`db.py`)**:
  * Created `model_registry` table in SQLite for managing checkpoints, LoRAs, UNets, default weights, and trigger keywords.
  * Added automatic registry seeding (`seed_default_model_registry`) during bot startup.
  * Added `/models` slash command allowing users to inspect and filter registered models by architecture and type.
* **LoRA Variant Auto-Routing & Compatibility Validator (`parsers.py`)**:
  * Enhanced `parse_loras` and `apply_loras_to_workflow` to resolve and auto-route LoRA requests to matching architecture variants (e.g., mapping `ogarla` to `ogarlaflux` on Flux, `ogarlapony` on Pony, and `ogarla` on SDXL).
  * Added pre-execution validation guard (`validate_workflow_loras`) to reject or warn against incompatible pairings before queuing jobs to ComfyUI, preventing PyTorch tensor mismatch crashes.
* **Externalized Quadrant Cache Storage**:
  * Migrated 2x2 grid temporary quadrant slices from repository `scratch/quadrants/` to `C:\ComfyUI\ComfyUI\output\Discord Bot\scratch\`.
* **Model Auto-Discovery Scanner (`/scan_models`)**:
  * Added `scan_and_register_comfyui_models()` to dynamically walk ComfyUI model folders, parse Safetensors metadata headers on the fly, and register all new models and LoRAs into SQLite.
  * Added `/scan_models` slash command for one-click discovery and tagging.
* **SQLite Automatic Space Reclaiming (`VACUUM`)**:
  * Added `db.vacuum_database()` and integrated automatic vacuuming into `db.prune_database()`, reclaiming fragmented database pages on disk.
* **Non-Blocking Async Image & File I/O**:
  * Added `save_quadrant_images_async`, `get_quadrant_bytes_async`, `embed_metadata_async`, and `create_grid_async` using `asyncio.to_thread` to protect the Discord gateway event loop from heavy disk operations.
* **Dataset Export Retention Pruning**:
  * Added `lora_dataset.cleanup_old_dataset_exports()` to prune old training export zip packages older than 14 days.
* **Standardized Repository Security (`.gitignore`)**:
  * Added root `.gitignore` safeguarding API tokens (`.env`), SQLite databases (`cache.db*`), error journals, and dataset export zips.
* **Expanded Test Suite (46 Modules)**:
  * Added automated tests for model auto-discovery scanning, database vacuuming, async image I/O, dataset retention pruning, architecture classification, and validation guards.

---
## [2.2.0] - 2026-08-26

This release introduces the **Impact Pack Face Detailer (`/imagine_det`)**, **Right-Click Message Context Menus**, **SQLite WAL Performance Optimizations**, **Robust Interaction Fallbacks**, **Static LoRA Node Pre-Wiring**, and expanded **38-Module Automated Test Suite**.

### Added
* **Impact Pack Face Detailer (`/imagine_det`)**: Added a dedicated SDXL generation command that automatically runs decoded latents through an `UltralyticsDetectorProvider` (`bbox/face_yolov8m.pt`) and `FaceDetailer` pass. Tuned specifically for 8GB VRAM cards using cropped facial inpainting (`guide_size=512`, `denoise=0.40`, 20 steps) while maintaining full parameter parity with `/imagine`.
* **Right-Click Message Context Menus (Apps)**:
  * **`Animate to Video`**: Right-click any Discord image attachment or generation to open an interactive `VideoPromptModal` (with customizable prompt, 5s/10s duration, smoothness interpolation, and seed) and render it directly via Wan 2.2 Image-to-Video diffusion.
  * **`Blend Image`**: Right-click any image message to instantly launch the Florence-2 powered Image Blend Studio.
  * **`Adopt Post / Image` & `Adopt Midjourney Post`**: Rapidly parse, extract, and adapt prompt parameters and styles from external image posts.
* **Resilient Interaction Fallback System**:
  * Added fallback dispatchers (`send_followup_fallback`, `send_error_fallback`, `edit_original_fallback`, `edit_message_fallback`) in `bot.py` to catch expired/invalid Discord interaction tokens and seamlessly deliver embeds and attachments directly to the channel.
  * Added large prompt attachment handling (`handle_copy_prompt`) that automatically packages prompts exceeding Discord's 2,000-character limit into clean `.txt` file attachments.
* **High-Performance Architecture & Database Upgrades**:
  * **SQLite WAL Mode**: Enabled Write-Ahead Logging (`PRAGMA journal_mode=WAL`) in `db.py` for concurrent, non-blocking generation cache and metric operations.
  * **User History Indexing**: Added indexed `get_user_generations(user_id)` query for rapid retrieval of user generation records.
  * **Bounded Execution Timings Cache**: Capped `ComfyClient` timing caches (`timings` and `last_timing`) to 100 entries to prevent memory accumulation during heavy generation sessions.
  * **Precompiled Regex Parsers**: Precompiled regular expression patterns in `parsers.py` (`RE_ASPECT_RATIO`, `RE_SEED`, `RE_LORAS`, etc.) for zero-latency prompt evaluation.
* **Standardized Output File Naming**:
  * Unified output filenames across all generation commands (`blend_<ckpt>_<idx>_seed<seed>_sref<sref>.png`, `imagine_<ckpt>_...`, `junji_<ckpt>_...`, `icon_<ckpt>_...`) with automatic checkpoint name abbreviation and embedded seed/sref metadata.
* **Expanded Test Suite (38 Modules)**:
  * Added automated tests `test_module29_reroll_lora_preservation` through `test_module38_animate_to_video_context_and_modal` in `suite_test.py` covering Face Detailer injection, context menus, large prompt exports, SQLite WAL performance, and Extra PNGInfo validation.

### Changed
* **Static LoRA Node Pre-Wiring**: Pre-wired static LoRA loader nodes (Node 75 & Node 76) across 12 workflow templates (`txt2img_lowres`, `sdxl_powerhouse_2stage`, `com_flux_gguf`, `blend_lowres`, etc.) enabling instantaneous LoRA swapping without dynamic graph re-generation.
* **Ogarla Model Upgrades**: Updated default character LoRAs to `ogarla_epoch_5.safetensors` (SDXL) and `ogarlaflux_epoch_5.safetensors` (Flux).
* **Pipeline Execution Chain**: Re-ordered workflow node dependencies so IP-Adapter connects prior to FreeU V2 frequency filtering before reaching the KSampler.
* **Re-Roll LoRA Preservation**: Ensured that re-rolling (`🔄`) active generations accurately preserves and wires injected character and semi-realism LoRAs.

---

## [2.1.0] - 2026-08-16

This release focuses on **UI Decluttering**, **Generation Speed Optimization**, **Wai Illustrious Photorealism Calibration**, **Clean Checkpoint Isolation**, and **Rebranding to Shallot-CUI Bot**.

### Added
* **Consolidated Enhancements Dropdown**: Consolidated multiple individual boolean parameters (`lightning`, `smart`, `magic_prompt`, `freeu`, `curved_edges`) across `/sdxl`, `/imagine`, `/com`, `/ico`, and `/junji` into unified, clean `enhancements` dropdown selectors with clear visual icon indicators.
* **ByteDance SDXL-Lightning 4-Step & 8-Step Turbo**: Integrated ByteDance SDXL-Lightning LoRAs (`sdxl_lightning_4step_lora.safetensors` and `8step`) allowing ~3-5s image generations on 8GB VRAM cards with zero trigger words required. Supports `--lightning`, `--lightning8`, `--lightning4` prompt shorthands and dynamic sampler/CFG adjustments.
* **Flux.1 Schnell GGUF Support**: Downloaded and integrated `flux1-schnell-Q4_K_S.gguf` (6.47 GB) as an ultra-fast ~8s 12B Flow-Matching option in `/com` with automated 4-step sampling and bypass guidance.
* **Strict Checkpoint Isolation**: Standardized `SDXL_CHECKPOINT_CHOICES` across all SDXL commands (`/imagine`, `/sdxl`, `/ico`, `/junji`, `/lora-build generate`) to strictly isolate SDXL models and eliminate confusion with non-SDXL (Flux, Wan, LTX) models.
* **Consolidated Enhancements Unit Tests**: Added `test_module26_sdxl_checkpoint_isolation`, `test_module27_lightning_lora_parsing`, and `test_module28_consolidated_enhancements` in `suite_test.py`.

### Changed
* **Bot Rebranding to Shallot-CUI Bot**: Renamed the bot suite to **Shallot-CUI Bot** across all startup scripts (`run_bot.bat`), console window titles, telemetry embeds, and documentation.
* **Wai Illustrious SDXL Default & Photorealism Calibration**: Set `waiIllustriousSDXL_v170.safetensors` as the default checkpoint for `/sdxl`. Configured enhanced anti-anime / anti-cartoon negative prompt filtering (`anime, anime girl, manga, comic, cartoon, cel shaded, lineart, drawing, illustration, 2d, 3d cgi render, sketch, anime face, big eyes, flat shading...`) to ensure high-fidelity Class-A female portraiture without anime bias.
* **Community Model Default for `/com`**: Configured `flux1-dev-Q4_K_S.gguf` as the recommended default model for `/com`.
* **Automated Instance Reset on `/cui-start`**: Updated `/cui-start` to automatically terminate any currently running ComfyUI process (tracked PID or port 8188 listener), release all VRAM/sockets, and launch a completely fresh, unblocked ComfyUI server instance.
* **Documentation**: Updated `README.md` and `README-LITE.md` with complete documentation for all new slash commands, enhancement presets, and high-performance workflows.

---

## [2.0.0] - 2026-08-16

Major milestone release introducing **12B Flow-Matching (Flux.1 GGUF)**, **2-Stage Powerhouse SDXL**, **Windows 11 Icon Engineering**, and **Wan 2.2 Video Diffusion**.

### Added
* **`/com` — Community Popular 12B Flow-Matching**: Added a dedicated community workflow running quantized Flux.1 GGUF models (`flux1-dev-Q4_K_S.gguf`, `fluxedUpFluxNSFW_71Q4GGUF.gguf`) tailored for 8GB VRAM GPUs with dual CLIP (T5-XXL + CLIP-L) and FluxGuidance control.
* **`/sdxl` — 2-Stage Powerhouse SDXL**: Created an advanced 2-stage workflow combining base generation, FreeU V2 frequency filtering, 1.35x bicubic latent upscaling, and a 2nd-pass refinement pass (`0.48` denoise) for maximal detail on 8GB VRAM cards.
* **Ogarla LoRA Dual-Architecture Routing**: Integrated automatic LoRA routing between SDXL (`ogarla_epoch_5.safetensors`) and Flux (`ogarlaflux_epoch_1.safetensors`) depending on the selected workflow engine.
* **`/ico` — Windows 11 Icon Generator & Converter**: Added dual-mode icon generator: text-to-icon 1:1 grid generation and direct image upload conversion creating true multi-resolution `.ico` binaries (7 embedded layers from 16x16 to 256x256) with optional curved squircle corners.
* **`/video` — Wan 2.2 Image-to-Video Diffusion**: Added image-to-video workflow using Wan 2.2 (14B GGUF) with automatic aspect ratio preservation and RIFE 60fps frame interpolation.
* **`/ltx` — LTX Video Diffusion**: High-speed lightweight video generation workflow.
* **`/junji` — Master Art Stylizer**: Dedicated dark fantasy and manga stylization generator featuring Junji Ito and Martine Johanna aesthetic presets.
* **Performance Telemetry**: Built-in execution time profiling (init, sampling, post-processing), VRAM safeguard alerts, and SQLite generation metric logs.

---

## [1.2.0] - 2026-08-01

This release focuses on **Prompt Intelligence**, **Style Reference Management**, and **LoRA Training Workflows**.

### Added
* **Smart Art Director (`--smart`)**: Intelligent prompt expansion system that analyzes subject context to generate harmonized descriptions and automatically pairs compatible `--sref` style reference codes.
* **Magic Prompt Enhancer (`--magic` / `--mp`)**: One-click cinematic lighting and atmospheric enhancement for prompts.
* **Style Reference Engine (`--sref`)**: Visual style extraction allowing users to transfer aesthetic styles from uploaded images onto generations.
* **User Favorites Management**: Added SQLite-backed user favorite prompts and favorite style codes with autocomplete integration (`/save_prompt`, `/list_prompts`, `/delete_prompt`, `/favorite_styles`).
* **Character LoRA Dataset Builder (`/lora-build`)**: Interactive dataset preparation tool with background removal, square cropping, and automatic caption generation.

---

## [1.1.0] - 2026-07-22

This release focuses on improving **Code Modularity**, **Developer Experience (DX)**, and **Database Resilience** for single-developer workflows.

### Added
* **SQLite Database Layer (`db.py`)**: Migrated the active generations cache from volatile memory / JSON files to a transaction-safe SQLite database (`cache.db`). Includes automatic pruning of records exceeding a limit of 2,000 entries and cascading filesystem cleanup of quadrant images.
* **Active Generations Proxy**: Added `ActiveGenerationsProxy` to seamlessly forward dictionary operations (access, assignments, containment checks) to the SQLite database without refactoring caller components.
* **Startup Pre-flight Check**: Added connection health checking for the ComfyUI server on bot startup. Alerts the developer in console logs if the server is offline.
* **Configurable Logging Level**: Added `LOG_LEVEL` environment variable support in the `.env` configuration (e.g. `LOG_LEVEL=DEBUG`), loading environment configs before logger initialization.
* **Automated SQLite Cache Tests**: Added unit tests to `suite_test.py` (`test_module7_sqlite_persistence`) verifying SQLite read/write proxy operations.

### Changed
* **Code Modularization**: Refactored the monolithic `bot.py` (down from 3,400+ lines to ~2,500 lines) by extracting prompt parsers, image utility functions, and custom button layouts:
    * Moved parsing helpers to `parsers.py`.
    * Moved Pillow operations to `image_utils.py`.
    * Moved custom `discord.ui.View` layout definitions to `views.py`.
* **Test Imports**: Updated `suite_test.py` imports to pull functions from the new `parsers.py` and `image_utils.py` modules.
* **Test Gated Startup**: Modified `run_bot.bat` to run the automated unit tests before initiating bot startup. Execution aborts immediately if any test fails.

### Fixed
* **Traceback Source Mismatch**: Fixed a mismatch in the error journal where `suite_test.py` tests logged their source as `test_suite.py`.
* **Fallback Message Typo**: Fixed a minor typo in the handle upscale error fallback: `Generation encounter` -> `Generation encountered`.
