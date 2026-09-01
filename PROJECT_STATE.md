# 🧅 PROJECT STATE — Shallot-CUI Bot

> **Project Name:** Shallot-CUI Bot (*Discord AI Creation Studio*)  
> **Current Version:** `v2.3.0`  
> **Repository Target:** `Shallot-CUI-Bot`  
> **Last Updated:** September 1, 2026  

---

## 📌 Executive Summary
**Shallot-CUI Bot** is a feature-rich, asynchronous Discord bot bridging Discord users with a local or remote **ComfyUI** instance. It provides midjourney-style generation workflows (2x2 grids, upscale buttons, variations, pan, zoom, inpaint, outpaint, character & style references) as well as video generation (Wan 2.2 with MMAudio, Hunyuan, LTX-Video) and Windows 11 icon creation.

---

## 🏗️ Architecture & Core Modules

| Module / File | Responsibility |
| :--- | :--- |
| [`bot.py`](bot.py) | Main Discord bot initialization, application slash commands, event listeners, workflow assembly, and execution pipeline. |
| [`parsers.py`](parsers.py) | Prompt syntax parser: aspect ratios (`--ar`), wildcards (`{a\|b}`), Smart Art Director (`--smart`), Magic Prompt (`--magic`), LoRA triggers (`<lora:...>`), Style Reference (`--sref`), and Character Reference (`--cref`). |
| [`comfy_client.py`](comfy_client.py) | REST API and WebSocket client managing prompt queuing, node execution tracking, image/video binary downloads, and error capture. |
| [`views.py`](views.py) | Discord UI components: dynamic action buttons (U1–U4, V1–V4, Pan, Zoom, Inpaint), modals for custom prompts/seeds, and dropdown selectors. |
| [`image_utils.py`](image_utils.py) | Image manipulation: 2x2 grid stitching and splitting, outpaint canvas expansion, color-matching, metadata injection, and multi-resolution Windows ICO generation. |
| [`db.py`](db.py) | SQLite database layer (`cache.db`) for caching prompt runs, generation histories, user preferences, and usage analytics. |
| [`error_handler.py`](error_handler.py) | Centralized error recovery system categorizing failures, proposing automated recipes, and logging structured diagnostics (`error_log.json`). |
| [`monitor.py`](monitor.py) | System & GPU telemetry: live VRAM usage, queue depth monitoring, and WebSocket generation progress HUD. |
| [`config.py`](config.py) | Environment variables, checkpoint presets, admin/owner permission guards, and VRAM safety thresholds. |
| [`suite_test.py`](suite_test.py) | Comprehensive test suite (47+ unit & integration tests) covering parsers, workflows, dimension math, and error handlers. |
| [`workflows/`](workflows/) | 19 modular ComfyUI JSON API workflow templates (SDXL, Flux GGUF, Wan 2.2 + MMAudio, Upscalers, Blend, Inpaint/Outpaint). |

---

## ⚡ Active Features & Slash Commands

### 🎨 Image Generation
* **`/imagine`**: Generates a 2x2 grid (4 variations) using SDXL 2-stage pipeline with support for `--ar`, `--sref`, `--cref`, `--magic`, `--smart`, and `--sr.85`.
* **`/imagine-flux`**: High-fidelity generation via Flux.1 (GGUF optimized).
* **`/blend`**: Synthesizes and merges visual features of 2 to 5 reference images.
* **`/edit`**: Image-to-image modification with configurable denoise and prompt guidance.
* **`/outpaint`**: Expands canvas left, right, top, or bottom with auto-matched diffusion fill.

### 🎬 Video & Motion
* **`/video` / `/animate`**: Image-to-video synthesis powered by **Wan 2.2** with automated **MMAudio** foley sound generation, as well as Hunyuan and LTX-Video alternatives.

### 🛠️ Utilities & System
* **`/icon`**: Converts prompts or reference images into crisp, multi-size Windows 11 `.ico` desktop icon bundles.
* **`/describe`**: Reverse-engineers prompts and stylistic keywords from uploaded images.
* **`/upscale`**: High-resolution 2x/4x enhancement with detail recovery.
* **`/cui-status` & `/vram`**: Real-time ComfyUI health check and GPU VRAM monitoring.
* **`/cui-errors`**: AI-ready diagnostic export of recent ComfyUI errors.

---

## 🧪 Testing & Quality Assurance
* **Automated Test Suite:** Run with `python suite_test.py` (pre-flight checks enforced on startup in `run_bot.bat` — **49/49 tests passing**).
* **Test Coverage:**
  * Aspect ratio parsing & dimension quantization (multiples of 64).
  * Wildcard combinatorial expansion & RNG seeds.
  * LoRA injection & weight extraction.
  * IP-Adapter (`--sref` / `--cref`) graph mutation.
  * Outpaint canvas padding math.
  * Error auto-fix mechanisms and structured logging.
  * Documentation sync validation between commands and README.

---

## 🚀 Development & Operational Scripts

* **`run_bot.bat`**: Runs pre-flight test suite, updates daily changelog, and launches the Discord bot.
* **`run_monitor.bat`**: Launches the real-time GPU/VRAM hardware monitoring dashboard.
* **`check_comfy_errors.bat`**: CLI analyzer for inspecting and copying formatted error logs for debugging.
* **`auto_changelog.py`**: Automated changelog generator maintaining [`CHANGELOG.md`](CHANGELOG.md).

---

## 📋 Current Considerations & Upcoming Tasks

1. **Repository & Folder Renaming**:
   * [x] Standardized local folder name: `Shallot-cui-bot`.
   * [x] Upgraded and fixed `.venv` script wrappers and paths for the new directory.
2. **Resource Management**:
   * Continuous tuning of VRAM threshold alerts (`VRAM_CAUTION_THRESHOLD_PERCENT = 85.0%`).
3. **Workflow Enhancements**:
   * Further optimize audio-video synchronization in Wan 2.2 + MMAudio pipeline.
