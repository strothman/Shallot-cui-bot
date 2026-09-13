# Changelog

All notable changes to **Shallot-CUI Bot** will be documented in this file.

---

## [2026-09-13]

* 📚 **Technical Debt & Optimization Audit Reference (`docs/technical_debt_and_optimization_audit.md`)**:
  * Compiled and documented a comprehensive architectural audit covering the 6,100-line `bot.py` monolith, unchecked quadrant scratch cache accumulation, monolithic `parsers.py`, sequential media downloading, and queue concurrency limits, complete with priority matrices and solution blueprints for future sprints.

* 🔄 **Crash Recovery & SQLite Job Journaling (`services/recovery_service.py` & `db.py`)**:
  * **Persistent Job Journaling**: Added a WAL-mode SQLite schema (`pending_jobs`) to journal in-flight prompts with Discord `channel_id`, `message_id`, `user_id`, `command_type`, and prompt metadata upon initial dispatch to ComfyUI.
  * **Automated Startup Reconciliation**: Created `reconcile_pending_jobs` and `start_crash_recovery` in `services/recovery_service.py`. On bot startup (`on_ready()`), queries ComfyUI's `/history/{prompt_id}`, downloads finished media outputs (images, videos, gifs), and edits the original Discord status messages with a clean `🔄 Recovered after restart` badge.
  * **Interrupted & Error Handling**: If a render encountered an unrecoverable failure during the crash or downtime, the reconciliation worker automatically informs the user on Discord with an explanation rather than leaving them with a frozen status message.
  * **Non-Blocking Execution & Auto-Pruning**: Spawns asynchronously without delaying bot gateway heartbeats or command syncing. Automatically cleans up stale jobs older than 24 hours via `db.cleanup_stale_jobs()`.
  * **Automated Test Coverage**: Added 3 new unit and integration tests (`test_pending_jobs_db_crud`, `test_crash_recovery_media_reconciliation`, `test_crash_recovery_execution_error`) in `suite_test.py`; all 85 tests pass 100% green.

* 🛡️ **Semantic Workflow Adapter & Node Decoupling (`services/workflow_adapter.py`)**:
  * **Zero-Breakage Node Discovery**: Permanently insulates the bot from arbitrary ComfyUI node renumbering. Replaced brittle hardcoded numeric indexing (`wf["3"]`, `wf["5"]`, `wf["6"]`, `wf["75"]`, `wf["76"]`, `wf["822"]`) with semantic discovery based on `class_type`, title inspection, input parameter signatures, and graph connection tracing.
  * **Semantic Setters Library**: Added high-level manipulation helpers: `set_workflow_prompt`, `set_workflow_seed`, `set_workflow_dimensions`, `set_workflow_checkpoint`, `set_workflow_sampler_params`, `set_workflow_input_image`, `set_workflow_filename_prefix`, and `set_workflow_lora`.
  * **Renumbered Workflow Immunity**: Validated through `test_renumbered_workflow_immunity` in `suite_test.py`, where full workflows with completely scrambled 4-digit node IDs are successfully updated without runtime errors.
  * **Integration into Parsers & Services**: Refactored `apply_loras_to_workflow` and `prepare_bertflow_workflow` to locate LoRA loaders, checkpointers, and sampler nodes semantically while retaining 100% backward-compatible signatures.
  * **Suite Test Coverage**: Added 3 dedicated test suites to `suite_test.py`; all 82 automated tests pass green (100%).

* 🧠 **Engine-Aware Priority Queue & VRAM Thrashing Prevention (`services/engine_queue.py`)**:
  * **Model-Affinity Scheduling**: Automatically groups pending generation jobs by target model architecture (`SDXL`, `Flux.1`, `Krea 2`, `Wan 2.2`, `LTX Video`, `Florence-2`). When an engine is active in GPU memory, subsequent jobs targeting the same model run consecutively without unloading weights, eliminating 20–45 seconds of PCIe weight-swapping delays per render.
  * **Automatic VRAM Purge on Engine Switch**: When the scheduler transitions between different architectures (e.g. Wan 2.2 to Krea 2), it automatically invokes `comfy_client.free_memory(unload_models=True, free_memory=True)` to guarantee a clean VRAM state and eliminate out-of-memory crashes on 8GB GPUs.
  * **Dynamic Anti-Starvation Escalation**: Prevents job starvation by tracking waiting age; jobs waiting longer than 45 seconds automatically receive high priority to ensure fair execution regardless of active model affinity.
  * **Job Priority Hierarchy**: Categorizes tasks into `HIGH` (interactive buttons, rerolls, remixes, Florence-2 interrogations), `NORMAL` (standard `/imagine`, `/flux`, `/blend-krea`), and `LOW` (multi-minute video renders, batch runs, dataset generators).
  * **Seamless Non-Blocking Client Integration**: Updated `comfy_client.ComfyClient.generate()` to route through `EngineAwareQueue` with transparent fallback and direct execution support (`_execute_direct`).
  * **Upgraded `/queue` Dashboard**: Enhanced the Discord `/queue` command embed to display active engine status, active task duration, pending jobs breakdown by engine, and total VRAM weight swaps prevented.
  * **Comprehensive Test Suite**: Added 5 dedicated unit and integration tests to `suite_test.py` (`test_engine_queue_workflow_detection`, `test_engine_affinity_scheduling_and_thrashing_prevention`, `test_engine_queue_starvation_prevention`, `test_engine_queue_vram_purge_on_switch`, `test_engine_queue_cancellation_and_status`); all 79 automated tests pass green (100%).

* 🚀 **Event Loop Optimization: Offloaded PIL Operations & Disk I/O to Background Worker Threads (`asyncio.to_thread`)**:
  * **Zero Gateway Lag**: Replaced all synchronous PIL image operations (Lanczos upscaling, aspect ratio cropping, outpaint padding math, vibrancy boosting, metadata chunk embedding) and disk writes (`save_quadrant_images`, `get_quadrant_bytes`) across `bot.py`, `services/krea_service.py`, and `image_utils.py` with asynchronous non-blocking worker threads.
  * **Parallel Vibrancy Enhancement**: Multi-image color vibrancy and contrast enhancements now execute concurrently across CPU cores via `asyncio.gather(*[boost_image_vibrancy_and_contrast_async(...)])`.
  * **Async Helpers Library in `image_utils.py`**: Added non-blocking async variants for `crop_to_aspect_ratio_async`, `upscale_isolated_image_async`, `calculate_outpaint_padding_async`, `boost_image_vibrancy_and_contrast_async`, `crop_quadrant_from_grid_bytes_async`, `create_thumbnail_bytes_async`, and `convert_image_to_ico_async`.
  * **Suite Test Validation**: Expanded `test_async_image_io_and_quadrant_operations` in `suite_test.py` to cover all new async utilities; all 74 automated tests pass green.

* 🎯 **Dedicated `--sref random` 1-Click Toggle for `/blend-sdxl` (`views.py` & `bot.py`)**:
  * **Simplified Style Controls**: Replaced the multi-preset and saved-style cycle button on Row 3 with a dedicated 1-click `--sref random` toggle (`toggle_blend_sref`).
  * **Clean Visual States**: Displays `[🎲 --sref random: ON]` (blurple `primary` style) when active, and `[🎲 --sref random: OFF]` (grey `secondary` style) when disabled.
  * **Embed Dashboard Synchronization**: Streamlined the `Aesthetics -> Style:` field in `build_blend_embed` to display `🎲 --sref random` when ON and `OFF` when disabled.
  * **Backward Compatibility**: Updated interaction handlers in `bot.py` to route both `toggle_blend_sref:` and legacy `cycle_blend_style:` interactions to the toggle handler.
  * **Full Automated Test Coverage**: Added `test_blend_sdxl_sref_random_toggle_interaction` in `suite_test.py`; all 74 automated tests pass green.

* 🗄️ **Dataset Storage Offload to ComfyUI Output (`C:\ComfyUI\ComfyUI\output\Discord Bot\datasets`)**:
  * **Workspace Size Reduction**: Safely migrated 185.5 MB of character training datasets, image/caption pairs, YAML configurations, and archives (`valerie_krea2/`, `ogarla_krea2/`, `palgirl/`, `favorites/`, `valerie_krea2_dataset.zip`) out of the local Git workspace to `C:\ComfyUI\ComfyUI\output\Discord Bot\datasets\`. Reduced repository folder footprint from **257.5 MB down to 67.1 MB (74% disk reclamation)**.
  * **Centralized Configuration**: Defined `DATASETS_DIR = os.getenv("DATASETS_DIR", r"C:\ComfyUI\ComfyUI\output\Discord Bot\datasets")` in `config.py`.
  * **Tools Updated**: Updated `tools/build_character_dataset.py` and `tools/create_character_dataset_from_photos.py` to target `DATASETS_DIR` by default for image generations, captioning, YAML configs, and zip files.

* 🏗️ **Modular Krea Cog & Service Architecture (`cogs/krea_cog.py` & `services/krea_service.py`)**:
  * **Service Extraction**: Moved core Krea 2 workflow execution logic (`execute_bertflow`, `handle_bertflow_reroll`, `handle_bertflow_remix`, `handle_bertflow_toggle_char`, `handle_bertflow_upscale`, `handle_update_blend_krea_view`, `handle_submit_edit_blend_krea_prompt`, `handle_generate_blend_krea`, `execute_blend_krea_core`) into a dedicated `services/krea_service.py`.
  * **Discord Cog Layer**: Grouped `/bertflow` and `/blend-krea` slash commands, along with all associated dynamic autocompletes (`character`, `celebrity`, `favorite_prompt`), into `cogs/krea_cog.py`.
  * **Codebase Slimming**: Extracted ~800 lines of Krea 2 execution and UI interaction logic out of monolithic `bot.py` while maintaining 100% backward-compatibility via clean re-exports.
  * **Suite Test Validation**: Added `test_module73_krea_cog_modular_architecture` in `suite_test.py`; all 73 automated tests pass green.

* 🏗️ **Pilot Modular Cog Architecture (`cogs/vision_cog.py` & `services/vision_service.py`)**: Extracted `/blend-sdxl`, `/blend` alias, and `Blend Image (SDXL)` context menu into a standalone modular Discord Cog and pure execution service layer, removing 468 lines from `bot.py` while maintaining 100% backwards compatibility and passing all 71 automated test suite validations.

* ⚡ **100% SDXL Dedicated Blend Studio (`/blend-sdxl`)**:
  * **Primary Slash Command & Context Menu**: Promoted `/blend-sdxl` as the exclusive SDXL blend command and registered right-click context menu `Blend Image (SDXL)`. Removed the redundant legacy `/blend` slash command to eliminate Discord autocomplete clutter and provide clean symmetry with `/blend-krea`.
  * **Single-Parameter Slash Invocation**: Stripped optional parameters (`prompt`, `style`, `secondary_style`) so invoking `/blend-sdxl` only requires uploading an image, eliminating initial parameter friction.
  * **Unified 1-Page Interactive Dashboard (Zero Tabs)**: Merged the previous 2-tab navigation (`Canvas & Model` vs `Characters & Styles`) into a single consolidated 5-row screen. No tab-switching required to configure all generation settings.
  * **1-Click Semi-Realism Toggle**: Replaced the 6-item dropdown with a single-click button `[✨ Semi-Realism: ON (.75)]` / `[✨ Semi-Realism: OFF]`.
  * **1-Click Aspect Ratio Cycle**: Replaced the AR dropdown with a single-click cycle button `[📐 AR: {ar}]` cycling through all 6 supported aspect ratios.
  * **1-Click Style Preset Cycle**: Replaced the style dropdown with a clean preset cycle button `[🎨 Style: {name}]`, eliminating style batch queue bloat (`batch:5`, `batch:10`, `batch:15`).
  * **Consolidated Generation Launcher**: Replaced dual blend buttons (`[🏷️ Blend Tags]` and `[✨ Blend Scene]`) with a single prominent `[🎨 Blend Image]` button and `[✏️ Edit Prompt]`.
  * **Hardwired Florence-2 for Instant Interrogation**: Replaced slow JoyCaption (which suffered 9+ minute VRAM thrashing on 8GB RTX 5060 Ti) with high-efficiency Florence-2 vision analysis (~1.5–2.5s generation time, ~1.2GB VRAM).
  * **Dual SDXL CLIP Prompt Synthesis**: Generates both concise comma-separated tags (`sdxl_prompt` for CLIP-G/CLIP-L) and structured spatial scene composition (`sdxl_detailed_prompt`) formatted directly for SDXL's 77-token windows without wordy filler.
  * **Pure SDXL Architecture**: Studio dashboard is restricted 100% to SDXL checkpoints (`wai`, `illustrious_realism`, `realvis`, `juggernaut`, `copax`, `ultra`, `hyphoria`, `nova`) and verified SDXL LoRAs.

* ⚡ **Decoupled `/blend` and `/blend-krea` Workflows**:
  * **`/blend` (Speed & Purity)**: Completely isolated from Krea 2. Removed `[⚡ Blend in Krea 2]` launcher button, Krea 2 models (`Muse v3.5 Extended`, `Pornmaster v2`) from checkpoint dropdown, and all Krea 2 prompting/generation routing.
  * **Dedicated SDXL Vision Workflow (`workflows/DESCRIBE_blend.json`)**: Built a streamlined Florence-2 workflow executing only the 2 essential SDXL passes (Short Tags `caption` and Detailed Scene `detailed_caption`), cutting out the 400-token beam search pass previously dedicated to Krea 2 for significantly faster `/blend` initialization.
  * **`/blend-krea` (Single-Parameter Invocation)**: Stripped all 7 optional parameters (`prompt`, `aspect_ratio`, `character`, `celebrity`, `model`, `wetness`, `composition`) so invoking `/blend-krea` only requires uploading an image, matching `/blend-sdxl` and leaving all customization to the interactive Phase 2 studio dashboard.

### Fixed
* 🐛 **Discord Command Description Length Limit (HTTP 400 Error 50035)**: Shortened `/blend-krea` description to 93 characters to comply with Discord's strict 100-character maximum limit, and enhanced `suite_test.py:test_discord_command_description_lengths` to dynamically validate all live registered command objects.
* 🐛 **SDXL LoRAs and Semi-Realism (--sr) Flag Wiping Bug (`bot.py`)**: Fixed an issue in `execute_imagine` where a redundant second call to `parse_loras` was executed on an already cleaned prompt string after LoRA flags were stripped, which caused `loras` to be wiped out to `[]` and disabled all character LoRAs and `--sr` / `--semi-realism` weights.
* 🐛 **Checkpoint Misclassification as LoRA (`model_architecture.py`)**: Fixed a false-positive substring check `"sr" in clean_name` which mistakenly classified `illustriousRealismBy_v10VAE.safetensors` as a LoRA rather than a Checkpoint due to the letters `sr` inside `illustriousRealism`. Replaced with regex word-boundary matching `(?:^|[-_.\s])sr(?:$|[-_.\s0-9])`.
* 🐛 **`/describe` Action Buttons Execution**: Fixed `handle_generate_described` passing `model=` instead of `checkpoint=` to `execute_imagine()`, which caused a `TypeError: execute_imagine() got an unexpected keyword argument 'model'` when clicking **Generate Caption** or **Generate Detailed**. Also added `model` parameter alias support to `execute_imagine` for robust backwards/cross compatibility.

## [2026-09-11]

### Added
* 📐 **Auto-Detected Source Image Aspect Ratio in `/blend-krea` (`image_utils.detect_closest_krea_aspect_ratio`)**: Implemented automatic aspect ratio detection for `/blend-krea` from the uploaded source image dimensions, mapping seamlessly to the Krea 2 Studio button set (`21:9`, `16:9`, `1:1`, `3:4`, `9:16`). Eliminated the rigid default of `16:9` so portraits, square images, and ultrawide uploads are immediately detected and pre-selected in the interactive dashboard without manual button clicking.
* 🌟 **Krea 2 Curated Celebrity Presets (`celebrities.py`)**: Added a dedicated celebrity registry and prompt injection system for 16 favorite actresses/icons (Audrey Hepburn, Grace Kelly, Nicole Kidman, Margot Robbie, Sandra Bullock, Emma Stone, Anya Taylor-Joy, Zendaya, Cameron Diaz, Saoirse Ronan, Emma Watson, Michelle Pfeiffer, Gal Gadot, Taylor Swift, Ariana Grande, Keira Knightley), taking advantage of Krea 2 Turbo's inherent facial knowledge without requiring extra LoRAs.
* 🎭 **Independent Character & Celebrity Selection in `/blend-krea` (`views.BlendKreaButtons`)**:
  * Retained the dedicated **Character LoRA selector** (Ogarla, Valerie) on Row 2.
  * Added a dedicated 1-click **Celebrity selector** dropdown on Row 3 with intuitive emoji badges and descriptions.
  * Re-organized Action & Engine buttons on Row 4 (`[🤖 Engine]`, `[💧 Skin]`, `[✏️ Edit Prompt]`, `[⚡ Generate Krea 2]`) within Discord's 5-row architecture.
  * Displayed both Character and Celebrity status in the `/blend-krea` Studio pipeline settings embed.
* 📸 **Slash Command Autocomplete & Prompt Flags for `/bertflow` and `/blend-krea`**:
  * Added `celebrity` options to `/bertflow` and `/blend-krea` with live autocomplete.
  * Added instant prompt shorthand flag parsing (e.g. `--audrey`, `--zendaya`, `--margot`, `--keira`, `--celeb <name>`) in `parsers.prepare_bertflow_workflow`.
* 📖 **Curated Favorites Reference Guide (`docs/krea2_celebrity_reference.md`)**: Updated the community recognition reference guide with a dedicated *Curated Favorites (Bot Presets)* table detailing recognition tiers and iconic visual markers.
* 🛡️ **Comprehensive LoRA & Workflow Architecture Audit & Zero-Bleed Guardrails (`scripts/audit_loras.py`, `suite_test.py`)**:
  * Added automated architectural auditing verifying that all 25+ LoRA references across 20 workflow templates, character profiles (`characters.py`), and UI dropdowns strictly match their base engine architecture (SDXL, Flux, Krea 2, Wan).
  * Added active architecture guardrails in `parsers.apply_loras_to_workflow`: Incompatible LoRAs (e.g., attempting to append an SDXL LoRA onto a Flux or Krea 2 graph) are automatically intercepted and safely excluded, preventing ComfyUI `lora key not loaded` warnings.
  * Enhanced `model_architecture.detect_model_architecture` with direct Safetensors header inspection and model directory path resolution across `checkpoints`, `loras`, and `diffusion_models` for SDXL, Flux, Krea 2, Lumina 2, Wan, SD1.5, and SD3.5.
* 🕒 **Proactive Discord Interaction Expiration Protection (`bot.py`, `core_helpers.is_interaction_expired`)**:
  * Implemented proactive 14.5-minute timestamp checking against `interaction.created_at` to detect expired tokens before attempting doomed webhook calls.
  * Fast-routes expired generations directly to `channel.send` without logging noisy 50027 interaction token error traces.
* 🎨 **Dedicated Valerie SDXL Photorealism Workflow (`workflows/valerie_sdxl_photorealism.json`)**: Created a production-ready ComfyUI workflow utilizing Valerie's trained SDXL LoRA (`jen_epoch_5.safetensors`) with `RealVisXL_V5.0_fp16.safetensors`, 832x1216 portrait resolution, and DPM++ 2M Karras sampling.
* **Compact studio embed and add unified prompt editing modal**

### Fixed
* 🖼️ **Initial Source Thumbnail in `/blend-krea` and `/blend` Embeds (`bot.edit_original_fallback`, `image_utils.create_thumbnail_bytes`)**: Resolved an issue where Discord's embed media proxy failed to render the initial uploaded image thumbnail because slash command attachment URLs are hosted on restricted/ephemeral CDN paths. Added automatic creation of a lightweight thumbnail attachment (`source_thumb.jpg`) delivered directly via message attachments (`attachment://source_thumb.jpg`), guaranteeing that the source image thumbnail always displays immediately and persists across view button clicks.
* ⚡ **Eliminated Dashboard Event Loop Freezes & Lag (`db.save_generation`, `image_utils.create_thumbnail_bytes`)**:
  * Offloaded `db.prune_cache()` to a background daemon thread so periodic file-deletion cycles over the 5,900+ scratch files never stall the asyncio event loop or block Discord button interactions.
  * Offloaded thumbnail generation in `/blend-krea` and `/blend` via `asyncio.to_thread(create_thumbnail_bytes, image_bytes)` with fast `BILINEAR` resampling.
  * Added in-memory JSON workflow template caching (`parsers.load_workflow_template`) to eliminate repetitive disk file reads and JSON deserializations on every slash command.
* 🧠 **Plugged Unbounded In-Memory Caches & Leaks (`bot.ActiveGenerationsProxy`, `comfy_client.ComfyClient`, `error_handler.ErrorHandler`)**:
  * Converted `ActiveGenerationsProxy._cache` from an unbounded dictionary to an LRU `OrderedDict` capped at 500 items to prevent infinite RAM growth over extended bot uptime.
  * Ensured `ComfyClient.timings` entries are always popped in the `finally:` block of `generate()` even during timeouts, interruptions, or transient errors.
  * Capped `ErrorHandler._retry_tracker` to prevent unbounded error key accumulation.

---
## [2026-09-10]

### Added
* 🧬 **Dual-Engine Synthetic Dataset Builder (`tools/build_character_dataset.py`)**: Enhanced the dataset builder to seamlessly support both **Flux GGUF** and **SDXL LoRAs** with automatic engine detection (`--engine auto|flux|sdxl`). Characters with SDXL LoRAs (like Valerie's `jen_epoch_5.safetensors`) can now generate diverse 30-sample 1024x1024 training datasets without relying on IP-Adapter.
* 🎭 **Smart Generation vs. Caption Triggering**: Separated the LoRA activation trigger (e.g. `jen`) from the training caption trigger (e.g. `valerie`), allowing new Krea 2 LoRAs to be cleanly captioned with user-facing trigger words.
* 📁 **Automated AI-Toolkit YAML & One-Click Runner (`generate_valerie_dataset.bat`)**: Automatically generates ready-to-train AI-Toolkit `.yaml` configurations pointing to the dataset path and includes a one-click batch launcher for generating Valerie's Krea 2 dataset.
* 📦 **Valerie Krea 2 Dataset (`datasets/valerie_krea2`)**: Generated complete 30-image photorealistic modeling dataset with Florence-2 captions and created `datasets/valerie_krea2.zip` ready for RunPod training.
* 🧬 **Character Dataset Generator from Reference Photos (`tools/create_character_dataset_from_photos.py`)**: Built an automated identity-expansion dataset generator allowing users to create full 30-image Krea 2 LoRA training datasets from just 1 to 12 reference photos using IP-Adapter identity projection, a balanced prompt matrix (full-body, medium, portrait), Florence-2 auto-captioning, and seamless resume/job continuation logic. Includes 1-click batch launcher `generate_dataset_from_photos.bat`.
* 👗 **High-Fashion & Modeling Prompt Matrix**: Scripted dedicated modeling categories (~25% Swimwear, ~25% Glamour Couture/Evening Dresses, ~20% Athletic Activewear, ~15% Streetwear Chic, ~15% Beauty Portraits) to guarantee publication-grade versatility for LoRA training. Tuned IP-Adapter weights and steps (`end_at: 0.65`, dynamic weights) and added clothing-suppression negatives to prevent reference sweater/wardrobe bleed into modeling shots.
* 🎥 **10-Second Natural Speed Video Pipeline**: Rebuilt `/video` duration and frame scaling to generate 161 true native diffusion frames in Wan 2.2 for 10-second clips (eliminating the 0.5x slow-motion temporal dilation caused by stretching 81 frames). Defaulted duration to 10 seconds across `/video`, remix modals, and toggles.
* ⚡ **Natural Speed Conditioning & `--realtime` Flag**: Replaced sluggish `"smooth video motion"` positive prompt template with `"natural motion, real-time speed"`, and added `--realtime`, `--natural`, and `--normal-speed` prompt flags to `parse_video_motion_flags` for authentic pacing without slow-motion lag.
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
