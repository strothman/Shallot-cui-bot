# 📋 Architectural Action Plan & Optimization Backlog — Shallot-CUI Bot

> **Created:** September 16, 2026  
> **Target Version:** `v2.8.0+`  
> **Status:** Active Reference & Roadmap  
> **Baseline Test Suite:** 101/101 Tests Passing (`suite_test.py`)

This document preserves the prioritized action plan and technical debt analysis for the bot to guide future implementation sessions.

---

## 🧭 Phase-by-Phase Implementation Roadmap

### 🔴 Phase 1: Event Loop Hygiene & Concurrency Safety (AGENTS.md Rule 1)
* **Goal:** Eliminate all blocking I/O and unmanaged threads on the asyncio event loop to prevent Discord gateway disconnects ("Heartbeat missed / connection reset") and database locks.
1. **Async Error Log Persistence (`error_handler.py`)**:
   - Move `json.dump()` in `ErrorHandler._save()` off the main event loop to a background worker thread via `asyncio.to_thread()` or debounce writes with a dirty flag.
2. **Managed Threading / Task Execution (`db.py`)**:
   - Replace `threading.Thread(target=prune_cache, daemon=True).start()` in `db.save_generation()` with an `asyncio.create_task()` or bounded threadpool executor to prevent concurrent SQLite write lock collisions.
3. **Loop API Modernization (`comfy_client.py`)**:
   - Replace deprecated `asyncio.get_event_loop().time()` with `asyncio.get_running_loop().time()` or `self.loop.time()`.
4. **WebSocket Progress Throttling (`comfy_client.py`)**:
   - Throttle `progress_callbacks` to only spawn tasks at meaningful intervals (e.g. Every 10% or minimum 500ms elapsed) to avoid task flooding.

---

### 🟠 Phase 2: Deduplication & Configuration Alignment
* **Goal:** Resolve code drift between `bot.py`, `core_helpers.py`, and `config.py`.
1. **Helper Alignment (`core_helpers.py` vs `bot.py`)**:
   - Upgrade `core_helpers.safe_defer()` to include the 3-attempt exponential backoff retry loop currently present in `bot.py`.
   - Remove redundant local helper definitions (`safe_defer`, `download_image`, `send_followup_fallback`, `send_error_fallback`, `edit_original_fallback`, `edit_message_fallback`) from `bot.py` and import directly from `core_helpers.py`.
2. **Choice Registry Unification (`config.py` vs `bot.py`)**:
   - Remove local copies of `SDXL_CHECKPOINT_CHOICES`, `SDXL_ENHANCEMENT_CHOICES`, and `CHECKPOINT_CONFIGS` from `bot.py` and import them from `config.py`.

---

### 🟡 Phase 3: Idiomatic Cog Registration Lifecycle
* **Goal:** Eliminate top-level `asyncio.run(bot.add_cog(...))` at module import time.
1. **Discord.py 2.x `setup_hook()`**:
   - Subclass `commands.Bot` (or define `bot.setup_hook`) to register cogs asynchronously during bot startup.
2. **Test Suite Compatibility (`suite_test.py`)**:
   - Ensure test suite uses an explicit synchronous loader (e.g. `load_all_cogs_sync(bot)`) so `bot.tree.get_commands()` remains testable without a live Discord connection.

---

### 🟢 Phase 4: Interaction Dispatcher & Dynamic Items
* **Goal:** Replace the 670-line `on_interaction` string-matching switch statement.
1. **Router Table / Dispatcher (`services/interaction_dispatcher.py`)**:
   - Map `custom_id` prefixes to structured async handlers with schema validation.
   - Provide centralized error boundaries so bad parameters or unknown IDs show user-friendly ephemeral alerts rather than silent timeouts.
2. **Embrace `discord.ui.DynamicItem`**:
   - Transition high-frequency buttons (like variations, rerolls, and isolates) to discord.py native dynamic item pattern.

---

### 🔵 Phase 5: Core Generation Extraction (`imagine_cog` & `imagine_service`)
* **Goal:** Reduce `bot.py` from 5,200 lines to ~500 lines.
1. **`cogs/imagine_cog.py`**:
   - Move `/imagine`, `/study`, and post adoption context menus.
2. **`services/imagine_service.py`**:
   - Move `execute_imagine()`, grid lifecycle post-processing, quadrant isolation, variation logic, and reroll workflows.

---

### 🟣 Phase 6: Submodule Segregation of `parsers.py` [COMPLETED]
* **Goal:** Break the 2,158-line multi-responsibility module into focused packages:
- `parsers/prompts.py`: Regex extraction (`--ar`, `--sref`, `--cref`, `--seed`, `--magic`, dynamic wildcards).
- `parsers/workflows.py`: ComfyUI graph AST manipulation (`apply_loras_to_workflow`, `apply_face_detailer_to_workflow`).
- `parsers/styles.py`: SREF style constants, lighting, palettes, and preset definitions.
- `parsers/dimensions.py`: Resolution math and aspect ratio bounding box logic.
- `parsers/__init__.py`: Master re-exporter providing 100% backward compatibility for all imports.
* **Status:** ✅ Completed. 103/103 tests and 4 architectural audits passing green.

---

## 🗑️ Feature Retirement: Deprecating `/flux`

* **Status:** Proposed for removal in `v2.8.0`.
* **Rationale:**
  - Flux GGUF is rarely used, consumes 15–25 GB of disk space, and incurs 20–45s VRAM model swap overhead on an 8GB GPU.
  - SDXL (`/imagine`) provides 4-quadrant grids in 8–15 seconds, making it far superior for rapid synthetic character dataset creation.
* **Retirement Steps:**
  1. Remove `/flux` command from `bot.py`.
  2. Remove `FLUX_CHECKPOINT_CHOICES`, `FLUX_ENHANCEMENT_CHOICES`, and `CHARACTER_CHOICES_FLUX`.
  3. Remove Flux lane from `services/engine_queue.py`.
  4. Archive `workflows/com_flux_gguf.json` and `workflows/flux_lowres.json`.
  5. Remove Flux-specific test assertions from `suite_test.py`.
