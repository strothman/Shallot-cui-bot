# Changelog

All notable changes to **Shallot-CUI Bot** will be documented in this file.

---

## [2026-08-26]

### Added
* **feat: initialize Shallot-CUI Bot v2.3.0 standalone repository** (`9acba56`)

### Maintenance
* **docs: consolidate into single novice-friendly README.md** (`1819575`)
* **chore: update changelog to reflect Shallot-CUI Bot v2.3.0 initialization and updated ignore rules** (`c567c7b`)
* **chore: expand .gitignore with safety rules** (`c8ddbc6`)
* Working tree modification: `M README.md`
* Working tree modification: `M auto_changelog.py`

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
