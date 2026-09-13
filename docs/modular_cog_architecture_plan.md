# 🏗️ Phase 3 Architecture Blueprint: Modular Cog Overhaul

> **Document Version:** `v2.4.2` (Updated September 13, 2026)  
> **Target Architecture:** Discord.py Modular Cogs (`commands.Cog`) & Extracted Execution Services  
> **Status:** 📋 Validated Blueprint — Ready for Phased Execution (71/71 Tests Passing)  
> **Objective:** Decompose the 7,800+ line monolithic [`bot.py`](bot.py) into specialized, maintainable modules without breaking existing commands, button callbacks, database persistence, context menus, or test suites.

---

## 1. High-Level Vision: Before vs. After

### Current State (Monolith)
* [`bot.py`](bot.py) (7,849 lines) contains:
  * Global variables & environment config (~200 lines)
  * ComfyUI client lifecycle & memory hooks (~150 lines)
  * Central interaction router `on_interaction` (~800 lines)
  * Slash command definitions (~2,500 lines)
  * Context menu handlers (~300 lines)
  * Modal callbacks & button responses (~1,500 lines)
  * Heavy generation pipelines (`execute_imagine`, `execute_bertflow`, `execute_video`, `execute_blend_core`) (~2,500 lines)
* **Pain Points:** Hard to navigate, high risk of git merge conflicts, difficult to isolate issues in one engine (e.g. Krea 2 vs. SDXL) from another (e.g. Wan Video vs. Florence-2).

### Target State (Modular Cogs + Service Layer)
```
Shallot-cui-bot/
├── bot.py                     # Slim entry point (~350 lines): client init, cog loader, error hooks
├── cogs/                      # Discord Slash Command, Context Menu & UI Definitions
│   ├── __init__.py
│   ├── imagine_cog.py         # /imagine, /variation_mode, /negative; Context: "Adopt Post / Image", "Adopt Midjourney Post"
│   ├── krea_cog.py            # /bertflow, /blend-krea; Krea 2 action controls; Celebrities & Character autocompletes
│   ├── flux_cog.py            # /flux flow-matching generation & character selector
│   ├── video_cog.py           # /video, /ltx Wan 2.2 animation pipeline; Context: "Animate to Video"
│   ├── vision_cog.py          # /blend-sdxl, /blend (alias), /describe, /study; Context: "Blend Image (SDXL)"
│   └── admin_cog.py           # /cui-start, /cui-stop, /cui-status, /free, /purge-vram, /upscale, /queue, /models, /scan_models, /diagnostics, /prompt, /style
├── services/                  # Pure Python Business & ComfyUI Execution Logic
│   ├── __init__.py
│   ├── imagine_service.py     # SDXL multi-stage generation, Face Detailer, Powerhouse refiner, grid creation, upscaling
│   ├── krea_service.py        # Bertflow & Krea 2 flow-matching pipelines, direct composition locking
│   ├── video_service.py       # Wan 2.2 GGUF + RIFE interpolation & LTX pipeline
│   ├── vision_service.py      # Florence-2 interrogation, prompt synthesis (SDXL, Krea 2, Flux), blend execution
│   └── admin_service.py       # ComfyUI process management, memory purging, model scanner
├── characters.py              # Central character registry & autocomplete provider
├── celebrities.py             # Curated celebrity registry & prompt injection
├── model_architecture.py      # Cross-architecture taxonomy & safetensors guard
├── parsers.py                 # Workflow mutation & prompt parsing
├── views.py                   # Discord interactive views (Buttons, Modals, Selects)
├── comfy_client.py            # Async WebSocket ComfyUI client
├── db.py                      # SQLite persistence & metrics
└── suite_test.py              # Automated test harness (71 tests)
```

---

## 2. Critical Gotchas & Risk Mitigations

### Gotcha A: Circular Imports
* **The Danger:** A cog in `cogs/` needs `comfy_client` from `bot.py`, while `bot.py` imports the cog. Python will throw `ImportError: cannot import name ... from partially initialized module`.
* **The Solution:** 
  1. Pass the `bot` instance into every Cog's `__init__(self, bot: commands.Bot)`.
  2. Attach shared singletons directly to `bot` (e.g., `bot.comfy_client = comfy_client`, `bot.db = db`).
  3. Keep service functions pure: pass parameters explicitly rather than relying on module-level globals.

### Gotcha B: Context Menus Inside Cogs
* **The Danger:** Right-click context menus (`Blend Image (SDXL)`, `Animate to Video`, `Adopt Post / Image`, `Adopt Midjourney Post`) are currently decorated with `@bot.tree.context_menu()`. If moved incorrectly, they fail to register in `bot.tree`.
* **The Solution:**
  * Discord.py supports `@app_commands.context_menu(name="...")` directly inside a `commands.Cog`:
    ```python
    class VisionCog(commands.Cog):
        def __init__(self, bot):
            self.bot = bot

        @app_commands.context_menu(name="Blend Image (SDXL)")
        async def blend_image_context(self, interaction: discord.Interaction, message: discord.Message):
            ...
    ```

### Gotcha C: Test Suite Synchronization & Sync Cog Loading (`suite_test.py`)
* **The Danger:** In `discord.py 2.x`, `bot.load_extension()` is asynchronous (`await bot.load_extension(...)`). However, `suite_test.py` imports `bot` and checks `bot.tree.get_commands()` **synchronously** without launching the Discord gateway event loop.
* **The Solution:**
  * Provide a synchronous initialization helper `load_all_cogs_sync(bot)` or trigger cog registration during module import / setup so that `suite_test.py` immediately discovers all 28+ commands and context menus upon `from bot import bot`.

### Gotcha D: Custom ID Button Callbacks (`on_interaction`)
* **The Danger:** `bot.py` handles persistent custom IDs (like `bertflow_reroll:{id}`, `set_blend_krea_ar:{id}:{ar}`, `set_blend_sdxl_ckpt:{id}`, `U1-U4`) inside `on_interaction`. Moving command code must not break active button listeners.
* **The Solution:** 
  * Keep the central `on_interaction` delegator in `bot.py` (or a dedicated `interaction_router.py`), and have it call the extracted service functions (`krea_service.handle_bertflow_reroll()`, `vision_service.execute_blend_core()`, etc.).
  * Because custom IDs are stored in SQLite and stateless, button routing remains 100% backward-compatible.

### Gotcha E: Test Suite Backward Compatibility
* **The Danger:** `suite_test.py` imports directly from `bot.py` (`from bot import execute_bertflow, execute_imagine, handle_reroll, run_vision_interrogate`).
* **The Solution:** 
  * Re-export all extracted service functions in `bot.py`:
    ```python
    # Backward compatibility re-exports for tests and legacy callers
    from services.krea_service import execute_bertflow, execute_blend_krea_core, handle_bertflow_reroll, handle_bertflow_toggle_char
    from services.imagine_service import execute_imagine, handle_upscale, handle_variation
    from services.vision_service import run_vision_interrogate, execute_blend_core
    from services.video_service import execute_video_generation, execute_ltx_generation
    ```
  * Running `suite_test.py` will continue to pass without changing a single line of test code.

### Gotcha F: Dual Autocomplete Handlers Inside Cogs
* **The Danger:** In `KreaCog`, both character and celebrity autocompletes must be declared on `self`:
  ```python
  class KreaCog(commands.Cog):
      def __init__(self, bot):
          self.bot = bot

      @app_commands.command(name="bertflow", description="...")
      async def bertflow_command(self, interaction: discord.Interaction, prompt: str, ...):
          ...

      @bertflow_command.autocomplete("character")
      async def bertflow_character_autocomplete(self, interaction: discord.Interaction, current: str):
          return get_character_autocomplete_choices(current, Architecture.KREA2)

      @bertflow_command.autocomplete("celebrity")
      async def bertflow_celebrity_autocomplete(self, interaction: discord.Interaction, current: str):
          return get_celebrity_autocomplete_choices(current)
  ```

---

## 3. Step-by-Step Execution Playbook

Execute the overhaul in **5 incremental, test-verified phases**. Do NOT attempt all steps in one go.

```mermaid
graph TD
    S0[Current Monolith v2.4.2] --> S1[Step 1: Extract Services Layer]
    S1 --> T1[Verify 71/71 Tests Pass]
    T1 --> S2[Step 2: Pilot Cog - KreaCog]
    S2 --> T2[Verify 71/71 Tests Pass]
    T2 --> S3[Step 3: Video & Vision Cogs]
    S3 --> T3[Verify 71/71 Tests Pass]
    T3 --> S4[Step 4: Imagine & Flux Cogs]
    S4 --> T4[Verify 71/71 Tests Pass]
    T4 --> S5[Step 5: Admin Cog & Final bot.py Cleanup]
    S5 --> T5[Final Verification & Release v3.0.0]
```

### Step 1: Create `services/` and Extract Business Logic (Zero Discord Changes)
1. Create `services/krea_service.py`:
   - Move `execute_bertflow`, `execute_blend_krea_core`, `handle_bertflow_reroll`, `handle_bertflow_toggle_char`, `handle_bertflow_upscale`.
2. Create `services/video_service.py`:
   - Move `execute_video_generation`, `execute_ltx_generation`.
3. Create `services/vision_service.py`:
   - Move `run_vision_interrogate`, `execute_blend_core`.
4. Create `services/imagine_service.py`:
   - Move `execute_imagine`, `handle_upscale`, `handle_variation`, preserving the bugfix where `parse_loras` is not run redundantly.
5. In `bot.py`, import and re-export all functions from `services/`.
6. **Verification Checkpoint:** Run `python suite_test.py`. Must pass 71/71 tests cleanly.

### Step 2: Pilot Cog — `cogs/krea_cog.py`
1. Create `cogs/krea_cog.py`:
   - Implement `class KreaCog(commands.Cog)`.
   - Move `/bertflow` and `/blend-krea` commands into `KreaCog`.
   - Move their autocomplete handlers (`character` and `celebrity`) into the cog.
   - Add async setup hook:
     ```python
     async def setup(bot: commands.Bot):
         await bot.add_cog(KreaCog(bot))
     ```
2. In `bot.py`, add cog loader in startup and synchronous discovery helper for tests.
3. **Verification Checkpoint:** Run `python suite_test.py`. Verify `bot.tree.get_commands()` contains `bertflow` and `blend-krea`.

### Step 3: Extract `cogs/video_cog.py` and `cogs/vision_cog.py`
1. Move `/video`, `/ltx`, and `Animate to Video` context menu into `cogs/video_cog.py`.
2. Move `/blend-sdxl`, `/blend` (alias), `/describe`, `/study`, and `Blend Image (SDXL)` context menu into `cogs/vision_cog.py`.
3. Load both extensions in `bot.py`.
4. **Verification Checkpoint:** Run `python suite_test.py`. Must pass 71/71 tests cleanly.

### Step 4: Extract Core Image Generation (`cogs/imagine_cog.py` & `cogs/flux_cog.py`)
1. Move `/imagine`, `/variation_mode`, `/negative`, and context menus (`Adopt Post / Image`, `Adopt Midjourney Post`) into `cogs/imagine_cog.py`.
2. Move `/flux` into `cogs/flux_cog.py`.
3. Load both extensions in `bot.py`.
4. **Verification Checkpoint:** Run `python suite_test.py`. Must pass 71/71 tests cleanly.

### Step 5: Extract `cogs/admin_cog.py` & Slim Down `bot.py`
1. Move ComfyUI lifecycle commands (`/cui-start`, `/cui-stop`, `/cui-status`, `/free`, `/purge-vram`, `/upscale`, `/queue`, `/models`, `/scan_models`, `/diagnostics`) and prompt/style management commands into `cogs/admin_cog.py`.
2. Reduce `bot.py` to:
   - Environment and client initialization
   - Global error handling
   - `on_ready` event and dynamic `bot.load_extension()` loop across `cogs/`
   - `on_interaction` central router
   - `acquire_instance_lock()` singleton guard
3. **Verification Checkpoint:** Run `python suite_test.py`. Ensure total file size of `bot.py` is under 400 lines and all 71 tests pass.

---

## 4. Rollback & Contingency Plan
* If any step introduces unexpected Discord command registration errors:
  1. Simply comment out the `await bot.load_extension("cogs.<failing_cog>")` in `bot.py`.
  2. The backward-compatible fallback implementations in `bot.py` remain untouched during each step until that specific step is verified.
  3. Git branch protection: Perform this refactoring on a dedicated branch (`git checkout -b refactor/modular-cogs`) so main remains 100% production-ready at all times.
