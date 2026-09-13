# Agent Instructions — Shallot-CUI Bot

This document defines core engineering standards, architectural patterns, and behavioral guidelines for AI assistants and contributors working on **Shallot-CUI Bot**.

---

## 🛡️ Rule 1: Async Event Loop Hygiene (Never Block the Discord Gateway)

* **Never run blocking PIL operations on the main asyncio event loop:**
  Heavy image operations (Lanczos upscaling, 2x2 grid stitching, quadrant cropping, color vibrancy/contrast enhancement, metadata chunk embedding, format conversions) and disk file I/O (`open()`, `.save()`, `.read()`, `.write()`) **MUST ALWAYS** be dispatched to background worker threads via `asyncio.to_thread()` or by calling the dedicated `*_async` functions in [`image_utils.py`](image_utils.py).
* **Always use or provide `_async` variants:**
  When interacting with [`image_utils.py`](image_utils.py), use non-blocking async functions:
  * `save_quadrant_images_async(generation_id, images)`
  * `get_quadrant_bytes_async(generation_id, index)`
  * `create_grid_async(image_bytes_list, ...)`
  * `crop_to_aspect_ratio_async(image_bytes, target_w, target_h)`
  * `upscale_isolated_image_async(image_bytes, target_w, target_h)`
  * `calculate_outpaint_padding_async(image_bytes, mode_or_ratio)`
  * `boost_image_vibrancy_and_contrast_async(image_bytes, ...)`
  * `crop_quadrant_from_grid_bytes_async(grid_bytes, index)`
  * `create_thumbnail_bytes_async(image_bytes, max_dim)`
  * `convert_image_to_ico_async(image_bytes, ...)`
* **Concurrent Multi-Image Enhancements:**
  When post-processing multiple images (e.g. In `complete_grid_generation` or batch outputs), use `asyncio.gather(*[boost_image_vibrancy_and_contrast_async(...) for ...])` to run enhancements concurrently across available CPU cores.
* **Preserve Gateway Heartbeats:**
  Any synchronous operation exceeding 50ms must be offloaded to a thread to prevent Discord gateway disconnects ("Heartbeat missed / connection reset") and UI button latency.

---

## 🏗️ Rule 2: Modular Cog & Service Architecture

* Keep `bot.py` focused strictly on bot lifecycle, global events, and top-level error trapping.
* When adding or refactoring commands, follow the established modular cog pattern:
  * **Discord UI / Slash Commands:** Place in `cogs/<feature>_cog.py`.
  * **Core Execution / Business Logic:** Place in `services/<feature>_service.py`.
* Ensure service functions remain decoupled from `discord.Interaction` where practical to facilitate automated unit testing.

---

## 🎨 Rule 3: Single-Source Style & Character Registry

* Character triggers, LoRAs, display badges, and privacy filters must be registered in [`characters.py`](characters.py) rather than hardcoded in commands or view callbacks.
* Checkpoints and model architectures must be registered in [`model_architecture.py`](model_architecture.py).
* All automated tests in [`suite_test.py`](suite_test.py) must pass 100% green before completing any changes.
