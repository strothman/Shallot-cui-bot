# 📦 Packaging & Client Distribution Feasibility Plan

> **Document Version:** `v1.0.0` (Created September 13, 2026)  
> **Target Scenario:** Packaging Shallot-CUI Bot for standalone deployment on an external user's PC with their own ComfyUI server, custom checkpoints, and custom LoRAs.  
> **Target Implementation Timeline:** ⏳ **Next Month (October 2026)** — Documented & Parked for Future Execution.  
> **Overall Feasibility Rating:** 🟢 **Highly Feasible** (8.5/10 — requires 4 targeted adaptability modules).

---

## 1. Executive Summary

Packaging Shallot-CUI Bot so a friend can run it on their own PC with their own ComfyUI installation is **fully feasible**. The bot already connects to ComfyUI via standard REST and WebSocket protocols (`127.0.0.1:8188`) and has database model registry capabilities.

However, because the current repository was developed for a specific single-machine environment, **4 specific hardcoded dependencies** must be adapted before shipping:
1. **Dynamic Model Discovery**: Replacing hardcoded checkpoint/LoRA choices with dynamic ComfyUI API querying.
2. **Path & Environment Decoupling**: Removing hardcoded paths (`C:\ComfyUI\...`) in favor of relative paths and ComfyUI REST introspection.
3. **Pre-Flight ComfyUI Node Validator**: Detecting missing custom nodes on the friend's ComfyUI before workflows are queued.
4. **Turnkey Installer Package**: Providing a self-contained `setup.bat` and configuration wizard.

---

## 2. Gap Analysis: Current System vs. Portable System

| Component | Current State | Required Portable State | Feasibility |
| :--- | :--- | :--- | :--- |
| **ComfyUI Connection** | `COMFYUI_ADDRESS=127.0.0.1:8188` in `.env` | Keep standard `127.0.0.1:8188` default; works out of the box on any PC running local ComfyUI. | 🟢 Ready Now |
| **Checkpoints** | Hardcoded static choices in `SDXL_CHECKPOINT_CHOICES` (`waiIllustriousSDXL_v170`, `RealVisXL_V4.0`, etc.) | **Dynamic Autocomplete**: Query ComfyUI's `/object_info/CheckpointLoaderSimple` to automatically populate Discord autocomplete with whatever checkpoints exist in their `models/checkpoints/` folder. | 🟡 Needs Minor Adapter |
| **Character LoRAs** | Mapped to specific author filenames in `characters.py` (`ogarla_epoch_5`, `jen_epoch_5`, `susa_epoch_6`) | Allow custom LoRAs dropped in `models/loras/` to be auto-discovered via ComfyUI `/object_info/LoraLoader` with `--loraname` prompt support. | 🟡 Needs Minor Adapter |
| **Local ComfyUI Paths** | `COMFYUI_BATCH_PATH = r"C:\ComfyUI\run_nvidia_gpu.bat"` hardcoded in `config.py` | Auto-detect standard install locations (`C:\ComfyUI`, `ComfyUI_windows_portable`, or custom path in `.env`). | 🟢 Low Effort |
| **ComfyUI Custom Nodes** | Assumes `ComfyUI-Florence2`, `ComfyUI-VideoHelperSuite`, `ComfyUI-GGUF`, and `FreeU` are installed | **Pre-flight Validator (`check_comfy_env.py`)**: Runs on startup, checks node availability via `/object_info`, and provides 1-click ComfyUI Manager install links for any missing nodes. | 🟡 High Value |
| **Python Environment** | Active Python 3.10+ installation on developer PC | Turnkey `setup.bat` creating an isolated virtual environment (`.venv`) and installing pinned dependencies from `requirements.txt`. | 🟢 Low Effort |
| **Discord Bot Account** | Single shared bot token in `.env` | Friend creates their own free Discord Bot in the Discord Developer Portal and pastes token into `.env`. | 🟢 Documented Flow |

---

## 3. The 4 Key Adaptations Required for Distribution

### Adaptation 1: Dynamic Model Autocomplete via ComfyUI REST API
* **Why it matters:** If your friend doesn't own `waiIllustriousSDXL_v170.safetensors` or `RealVisXL_V4.0.safetensors`, generations will crash with `FileNotFoundError` in ComfyUI.
* **The Solution:**
  * ComfyUI's `/object_info/CheckpointLoaderSimple` endpoint returns the live list of all `.safetensors` files inside their `models/checkpoints/` directory.
  * In `bot.py`, switch `@app_commands.choices(checkpoint=...)` to `@app_commands.autocomplete("checkpoint")`.
  * The bot caches their installed models on startup, and when they type `/imagine checkpoint:`, Discord automatically suggests *their* models.

### Adaptation 2: Pre-Flight Environment Validator (`tools/check_comfy_env.py`)
* **Why it matters:** If the friend's ComfyUI is missing `ComfyUI-Florence2` or `ComfyUI-VideoHelperSuite`, features like `/describe`, `/blend-sdxl`, and `/video` will return a raw ComfyUI 400 error.
* **The Solution:**
  * A lightweight validator script that checks:
    1. Is ComfyUI running at `http://127.0.0.1:8188`?
    2. Are required core nodes present? (`KSampler`, `VAEDecode`, `CLIPTextEncode`)
    3. Are extension nodes present? (`Florence2Run`, `VHS_VideoCombine`, `UnetLoaderGGUF`)
    4. Are basic checkpoints present? (Flags if `models/checkpoints/` is empty).
  * Outputs a friendly colorized diagnostic report in terminal showing: `[PASS] ComfyUI Connected`, `[MISSING] ComfyUI-Florence2 (Install via ComfyUI Manager)`.

### Adaptation 3: Default Workflow Fallback Logic
* **Why it matters:** The JSON workflow files in `workflows/` have default node values pointing to specific model names.
* **The Solution:**
  * In `parsers.py`, when loading a workflow template:
    * If the requested checkpoint is not specified by the user, dynamically substitute node `ckpt_name` with the user's default checkpoint configured in `.env` or the first available SDXL model found in their local registry.

### Adaptation 4: Turnkey Distribution Archive
* **Package Structure for Distribution:**
  ```text
  Shallot-CUI-Bot-Release/
  ├── .env.example               # Pre-configured template
  ├── setup.bat                  # 1-click environment installer (creates .venv, installs wheels)
  ├── run_bot.bat                # 1-click launcher with auto-restart
  ├── check_comfy.bat            # Quick validator for ComfyUI nodes & models
  ├── README_FRIEND.md           # 5-minute setup guide with screenshots
  ├── bot.py
  ├── config.py
  ├── ... (all bot files & workflows)
  ```

---

## 4. Implementation Effort & Roadmap

| Step | Scope | Estimated Effort |
| :--- | :--- | :--- |
| **Step 1: ComfyUI REST Model Synchronizer** | Query ComfyUI `/object_info` on startup to populate `model_registry` with friend's local checkpoints and LoRAs. | ~2 hours |
| **Step 2: Dynamic Checkpoint Autocomplete** | Convert `/imagine checkpoint` from static list to dynamic autocomplete. | ~1 hour |
| **Step 3: Pre-Flight Validator Script** | Build `tools/check_comfy_env.py` to inspect nodes and models via REST. | ~1.5 hours |
| **Step 4: Turnkey Packaging (`setup.bat` & Guide)** | Create clean distribution zip with setup script and step-by-step setup guide. | ~1 hour |
| **Total Effort** | Fully packaged, plug-and-play distribution | **~5.5 hours** |

---

## 5. Conclusion

Distributing the bot to an external user is **highly practical**. Because the bot communicates with ComfyUI purely through standard HTTP REST and WebSockets, the architecture is inherently client-server and decoupled from the local hardware. Adding dynamic model discovery and a pre-flight node validator will turn Shallot-CUI Bot into a true plug-and-play desktop AI studio for any ComfyUI user.
