# 🔍 Technical Debt & Optimization Audit — Shallot-CUI Bot

> **Target Version:** `v2.6.0+`  
> **Date Compiled:** September 13, 2026  
> **Status:** Active Architectural Reference & Optimization Backlog  

This document preserves the comprehensive architectural audit of Shallot-CUI Bot's current shortcomings, bottlenecks, and structural inefficiencies to guide future development and maintenance.

---

## 📋 Executive Summary of Areas Audited

| # | Domain | Core Issue | Priority | Status / Version |
| :-: | :--- | :--- | :-: | :--- |
| **1** | **Monolithic Structure** | `bot.py` originally 6,133 lines; down to 5,111 lines with 5 cogs extracted | **High** | 🟡 **In Progress** (`v2.7.2` - Video, System, & Upscale Cogs extracted) |
| **2** | **Disk / Scratch Cache** | `QUADRANT_CACHE_DIR` accumulates thousands of PNGs without auto-pruner | **High** | ✅ **Resolved** (`v2.6.5` - 6h pruner + vacuum) |
| **3** | **Upscaling Engine** | 1-pass naive fixed-1920px filter & 1.25x grid scale factor | **High** | ✅ **Resolved** (`v2.7.2` - Fast Clean + Generative Clarity 2K/4K) |
| **4** | **Outpaint & Zoom** | Hardcoded SDXL anime checkpoint, naive 0.95 inpainting, undefined variable bug | **High (Next Week #1)** | 🔴 **Scheduled Backlog** (Multi-arch outpaint + seam blending) |
| **5** | **Interaction Dispatcher** | 670-line `on_interaction` string prefix `if/elif` chain in `bot.py` | **High (Next Week #2)** | 🔴 **Scheduled Backlog** (Router table / persistent views) |
| **6** | **Button Session DNA** | In-memory/SQLite temporary IDs expire on bot restart | **Medium (Next Week #3)** | 🔴 **Scheduled Backlog** (Embed metadata persistence) |
| **7** | **Module Coupling** | `parsers.py` (2,100 lines) mixes text, math, PNG chunks, & workflows | **Medium (Next Week #4)** | 🔴 **Scheduled Backlog** (Submodule segregation) |
| **8** | **Network Latency** | Sequential `session.get` calls when downloading multi-image batches | **Medium** | ✅ **Resolved** (`v2.6.5` - concurrent `asyncio.gather`) |
| **9** | **Queue Fair-Share** | No per-user active job cap; single users can spam queue | **Low** | Single-user workstation (Intentionally bypassed) |
| **10** | **Config Distribution** | Hardcoded denoise floats, CFG values, and checkpoint filenames in code | **Low** | ✅ **Resolved** (`v2.6.6` - `PipelineDefaults` & display resolution) |

---

## 1. 🐘 The `bot.py` Monolith & `on_interaction` Switch

### The Problem
While `cogs/vision_cog.py` and `cogs/krea_cog.py` were established as initial extractions, over **80% of all Discord command and interaction logic** was concentrated directly inside [`bot.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py) (originally 6,133 lines).

Furthermore, the [`on_interaction`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py#L2446) listener spans over 500 lines of chained string-prefix matching:
```python
if custom_id.startswith("stasis_pause:"): ...
elif custom_id.startswith("stasis_resume:"): ...
elif custom_id.startswith("cancel_gen:"): ...
elif custom_id.startswith("remix:"): ...
elif custom_id.startswith("upscale:"): ...
elif custom_id.startswith("variation:"): ...
elif custom_id.startswith("vary_subtle:"): ...
elif custom_id.startswith("vary_strong:"): ...
# ... over 30 more elif branches
```

### Architectural Risk & Inefficiency
- **Single Point of Failure**: Any syntax error, broken import, or unhandled exception in `bot.py` crashes the entire bot service.
- **Maintainability Burden**: Adding or updating any interaction requires navigating thousands of lines of unrelated code.
- **Testing Impediment**: Testing interaction logic requires mocking the entire Discord bot client rather than testing isolated cog classes.

### Solution Blueprint
1. Decompose `bot.py` into dedicated domain cogs following the established pattern in `cogs/`:
   - `cogs/vision_cog.py` & `services/vision_service.py`: ✅ **Completed (`v2.6.9`)** — `/describe`, `/blend-sdxl`, context menus, multi-architecture JoyCaption/Qwen2.5-VL/Florence-2 interrogation, and interactive prompt generation callbacks extracted.
   - `cogs/video_cog.py` & `services/video_service.py`: ✅ **Completed (`v2.6.7`)** — `/video`, `/ltx`, `"Animate to Video"`, Wan 2.2 I2V, RIFE interpolation, and video button handlers extracted (~800 lines offloaded).
   - `cogs/system_cog.py` & `services/system_service.py`: ✅ **Completed (`v2.6.8`)** — Server controls (`/cui-start`, `/cui-stop`, `/cui-status`), memory (`/free`, `/purge-vram`), `/queue`, `/diagnostics`, `/models`, `/scan_models`, `/negative`, `/variation_mode`, `/prompt`, and `/style` (~900 lines offloaded).
   - `cogs/imagine_cog.py` & `services/imagine_service.py`: `/imagine`, `/flux`, grid U1–U4, V1–V4, and remix callbacks.
2. Replace raw string matching in `on_interaction` with:
   - A centralized, table-driven interaction router in `services/interaction_dispatcher.py`, or
   - Persistent `discord.ui.View` classes registered with `bot.add_view()`.

---

## 2. 🗄️ Unchecked Scratch Disk Sprawl (`QUADRANT_CACHE_DIR` Accumulation)

### The Problem
Every single 4-image grid generation saves 4 uncompressed PNG files to `C:\ComfyUI\ComfyUI\output\Discord Bot\scratch` (`{generation_id}_{1..4}.png`) via [`save_quadrant_images_async()`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/image_utils.py#L251) so users can isolate, upscale, or create variations later.

While a cleanup utility [`cleanup_orphaned_quadrants()`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/db.py#L227) exists in `db.py`:
```python
def cleanup_orphaned_quadrants():
    # Removes any quadrant cache files that are no longer in the database
```
**It is never called, scheduled, or triggered anywhere in the bot.**

### Architectural Risk & Inefficiency
- On an active server generating 50 grids daily, 200 high-res PNG files are written to disk every single day.
- Over weeks and months, the folder silently consumes dozens to hundreds of gigabytes of disk space until the host runs out of drive capacity.

### Solution Blueprint
1. Implement a scheduled background maintenance worker in `bot.py` using `discord.ext.tasks`:
   ```python
   @tasks.loop(hours=6)
   async def scheduled_maintenance():
       # 1. Prune orphaned quadrant cache files older than 48 hours
       await asyncio.to_thread(db.cleanup_orphaned_quadrants)
       # 2. Reclaim SQLite pages
       await asyncio.to_thread(db.vacuum_database)
   ```
2. Automatically invoke this maintenance check once on bot startup after `on_ready()`.

---

## 3. 📦 `parsers.py` is a 2,100-Line "Junk Drawer"

### The Problem
[`parsers.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/parsers.py) currently contains over 2,088 lines performing multiple disparate responsibilities:
1. **Prompt & Flag Parsing**: Regex extraction of `--ar`, `--sref`, `--cref`, `--smart`, `--magic`, and dynamic wildcards `{a|b|c}`.
2. **Workflow Graph Manipulation**: Injecting nodes for LoRAs (`apply_loras_to_workflow`), Face Detailer (`apply_face_detailer_to_workflow`), IPAdapter (`apply_ipadapter_to_workflow`), and Krea2 Bertflow (`prepare_bertflow_workflow`).
3. **PNG Metadata Extraction**: Reading raw PNG chunks, parsing Automatic1111 parameter strings, and extracting embedded ComfyUI workflows.
4. **Resolution Math & Prompt Fusion**: Calculating Wan 2.2 aspect ratio bounding boxes and synthesizing multi-modal blend prompts.

### Architectural Risk & Inefficiency
- Violates the Single Responsibility Principle (SRP).
- Whenever another module only needs simple text parsing (e.g. `parse_aspect_ratio`), Python is forced to evaluate heavy image dependencies, PIL plugins, workflow caches, and model resolvers, increasing memory footprint and risking circular imports.

### Solution Blueprint
Split `parsers.py` into clear, single-responsibility submodules under a `parsers/` package:
- `parsers/prompt_parser.py`: Pure prompt parsing, aspect ratio math, flags, and wildcard expansion.
- `parsers/metadata_parser.py`: PNG chunk inspection, A1111 parameter extraction, and ComfyUI workflow recovery.
- Delegate all workflow node graph manipulation strictly to `services/workflow_adapter.py`.

---

## 4. 🌐 Sequential Image Downloading on Grid Completion

### The Problem
In [`comfy_client.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/comfy_client.py#L464), when ComfyUI finishes executing a prompt and returns the output node dictionary, the bot downloads output files one-by-one:
```python
for node_id, output in results_dict.items():
    if output:
        for key in ["images", "gifs", "videos"]:
            if key in output:
                has_outputs = True
                for item in output[key]:
                    file_bytes = await self.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                    output_bytes_list.append(file_bytes)
```

### Architectural Risk & Inefficiency
- For standard 4-image grids or multi-frame video outputs, the bot awaits 4 separate, sequential HTTP GET roundtrips over localhost.
- Each image must complete downloading before the next download starts, adding 200–500ms of avoidable latency before the Discord response is prepared.

### Solution Blueprint
Gather all image download coroutines and fetch them concurrently:
```python
download_tasks = [
    self.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
    for item in media_items
]
output_bytes_list = await asyncio.gather(*download_tasks)
```

---

## 5. 👥 Lack of Per-User Queue Concurrency Limits

### The Problem
While `services/engine_queue.py` prioritizes tasks and prevents GPU VRAM thrashing on the backend, there is no guard on **how many active jobs a single user can submit**.

### Architectural Risk & Inefficiency
- A single user in a Discord channel can run `/video` 5 times or `/imagine` 10 times consecutively.
- On an 8GB GPU, Wan 2.2 video generations take 2–4 minutes each. One user can completely monopolize the GPU for 15–20 minutes, locking out other server members who just want a fast 5-second `/imagine`.

### Solution Blueprint
1. Add an active job tracker in `services/engine_queue.py` mapping `user_id -> count_of_active_or_pending_jobs`.
2. Configurable threshold in `config.py` (e.g. `MAX_USER_CONCURRENT_JOBS = 2`, `MAX_USER_VIDEO_JOBS = 1`).
3. If exceeded, return a friendly Discord ephemeral notification:
   > *"⚠️ You already have 2 jobs in the creation queue. Please wait for them to finish before queueing more!"*

---

## 6. 🪄 Scattered Magic Numbers & Hardcoded Checkpoint Strings

### The Problem
Several default settings, denoise floats, and checkpoint filenames are hardcoded directly inside command bodies across `bot.py` and `services/`:
- Denoise values: `0.55` (SDXL detail upscale), `0.26` / `0.35` (Flux upscale), `0.70` (subtle variation), `0.85` (strong variation).
- Checkpoint filenames: `"waiIllustriousSDXL_v170.safetensors"`, `"wan2.1_i2v_720p_14B_fp8.safetensors"`, `"flux1-dev-fp8.safetensors"`.
- Sampler parameters: Steps `20` / `4`, CFG `4.0` / `1.0`.

### Architectural Risk & Inefficiency
- Updating a checkpoint to a newer version (e.g. `v170` $\rightarrow$ `v180`) requires multiple manual replacements across disparate functions.
- Modifying default denoise strengths requires editing core command functions.

### Solution Blueprint
- Consolidate all pipeline defaults and denoise profiles into [`config.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/config.py) and [`model_architecture.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/model_architecture.py).
- Reference presets via semantic constants (e.g. `PipelineDefaults.UPSCALE_DENOISE_SDXL`, `PipelineDefaults.VARIATION_DENOISE_SUBTLE`).

### ✅ Resolution (`v2.6.6`)
1. Centralized generation defaults, upscale denoise constants (`UPSCALE_DENOISE_SDXL`, `UPSCALE_DENOISE_FLUX_SUBTLE`, `UPSCALE_DENOISE_FLUX_MODERATE`), and variation mapping profiles into `class PipelineDefaults` in [`config.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/config.py).
2. Implemented `get_checkpoint_display_name(checkpoint)` in [`config.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/config.py) with alias resolution for shorthand codes (`wai`, `realvis`, `juggernaut`, etc.), eliminating duplicate 18-line dictionaries in [`views.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/views.py).
3. Replaced raw magic floats throughout [`bot.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py) with semantic `PipelineDefaults` references.

---

## 7. 🖼️ Outpaint Architecture Amnesia & Naive Inpainting (Priority #1 — Next Week)

### The Problem
In [`handle_outpaint`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py#L1879) and [`workflows/outpaint_lowres.json`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/workflows/outpaint_lowres.json):
1. **Model Amnesia**: `outpaint_lowres.json` hardcodes `waiIllustriousSDXL_v170.safetensors`. If an image was generated with Flux or a photoreal checkpoint (RealVisXL / Juggernaut), clicking Outpaint/Zoom forces it through the WaiIllustrious anime model, corrupting style, faces, and palette.
2. **Naive Inpainting (Ghost Limbs & Seams)**: Uses standard `VAEEncodeForInpaint` at a high denoise (`0.95`). Without differential diffusion or feathered seam blending, outpainted borders frequently invent duplicate bodies or visible square edge artifacts.
3. **Hidden Runtime Bug**: Line 1908 references `prompt` (`parse_loras(original_prompt or prompt)`), but only `raw_prompt` is defined in scope.

### Solution Blueprint
1. Extract outpaint into `services/outpaint_service.py` and `cogs/outpaint_cog.py`.
2. Inspect `checkpoint` and `is_flux` from generation metadata to route outpaint through the matching model architecture.
3. Apply feathered boundary masks with seam blend denoising (`0.75–0.85` instead of flat `0.95`) to produce cohesive, seamless canvas extensions.
4. Fix variable reference bug on line 1908.

---

## 8. 🖲️ The 670-Line `on_interaction` String Switch Monolith (Priority #2 — Next Week)

### The Problem
In [`bot.py` (L2486–L3156)](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/bot.py#L2486-L3156), all button clicks, modals, and select menus across Discord are caught in a massive, single-threaded `if/elif/elif` string prefix parsing block over 670 lines long.

### Architectural Risk & Inefficiency
- Any uncaught exception in a single handler can disrupt the global listener.
- Difficult to unit test without constructing heavy mock `discord.Interaction` objects.
- Violates Open/Closed Principle: every new interactive button requires modifying the core `bot.py` listener.

### Solution Blueprint
1. Build `services/interaction_dispatcher.py` with a decorator-based or dictionary-driven handler registry:
   ```python
   INTERACTION_HANDLERS = {
       "upscale_run": handle_upscale,
       "vary_subtle": handle_variation_subtle,
       "vary_strong": handle_variation_strong,
       "remix": handle_remix,
       ...
   }
   ```
2. Convert `on_interaction` into a clean 15-line router that extracts the prefix key and dispatches safely with isolated error boundaries.

---

## 9. ⏳ Ephemeral Session Expiration / Restart Amnesia (Priority #3 — Next Week)

### The Problem
Buttons on grids and isolated images rely on `generation_id` stored in a temporary dictionary or SQLite row (`get_generation(generation_id)`). If the bot restarts or 24 hours pass, clicking `U1–U4`, `V1–V4`, `Remix`, or `Reroll` fails with:
> *"Could not find generation session data. It may have expired or the bot was restarted."*

### Solution Blueprint
1. Encode essential generation DNA directly into Discord UI components or persistent embed footers:
   - Base seed, model alias, aspect ratio, prompt hash.
2. If `generation_id` is missing from the cache on button click, reconstruct generation parameters on the fly from embed metadata and re-download the quadrant from the message attachment.
3. Completely eliminates dead buttons after bot restarts.

---

## 10. 📦 `parsers.py` Single-Responsibility Modularization (Priority #4 — Next Week)

### The Problem
[`parsers.py`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/parsers.py) spans over 2,100 lines and combines four distinct domains:
1. Discord prompt & flag parsing (`--ar`, `--sref`, `--cref`, `--smart`, `--magic`, wildcards).
2. ComfyUI workflow graph mutation (`apply_loras_to_workflow`, `apply_face_detailer_to_workflow`).
3. Binary PNG metadata reading and A1111 parameter parsing.
4. Aspect ratio math and bounding box calculations.

### Solution Blueprint
Decompose `parsers.py` into a focused `parsers/` package:
- `parsers/flags.py`: Prompt flags, regexes, and wildcards.
- `parsers/workflow_graph.py`: Node graph transformations and LoRA injection.
- `parsers/metadata.py`: PNG chunk reading and embedded workflow extraction.
- `parsers/dimensions.py`: Aspect ratio resolution and Wan dimension math.
- Re-export all functions from `parsers/__init__.py` for 100% backward compatibility.

