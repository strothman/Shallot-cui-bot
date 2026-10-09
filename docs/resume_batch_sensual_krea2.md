# Resume Guide: Star Board Sensual Krea 2 Remaster Batch

This document outlines how to safely resume the **Sensual & Passionate Krea 2 Remaster Batch** after a system reboot or pause.

---

## 📊 Current Job Status (At Pause)

* **Source Directory**: `C:\Users\strot\Pictures\star board\picture` (753 total images)
* **Output Directory**: `C:\Users\strot\Pictures\star board\krea2_sensual`
* **Progress File**: `C:\Users\strot\Pictures\star board\krea2_sensual\krea2_sensual_progress.json`
* **Completed Images**: **187 / 753** (24.8% complete)
* **Remaining Images**: **566 / 753**
* **VRAM Status**: Purged and freed via `/free` endpoint. Background worker stopped cleanly.

---

## 🔄 How to Resume After Reboot

### Method 1: Double-Click Launcher (Easiest)
1. Start your local ComfyUI server (e.g., via `C:\ComfyUI\run_nvidia_gpu.bat` or your usual shortcut).
2. Double-click the launcher script:
   ```
   c:\Users\strot\Antigravity IDE\Shallot-cui-bot\scripts\run_batch_sensual_star_board.bat
   ```
3. The runner will inspect `krea2_sensual_progress.json`, automatically skip the 187 finished images, and resume processing image #188 (`12_stars_img2img.evening-...`).

---

### Method 2: Command Line (PowerShell / Terminal)
1. Open PowerShell or Command Prompt.
2. Navigate to the project root:
   ```powershell
   cd "C:\Users\strot\Antigravity IDE\Shallot-cui-bot"
   ```
3. Run the Python runner:
   ```powershell
   python scripts/batch_sensual_star_board_krea2.py
   ```

---

## ⚙️ Optional Command-Line Flags

* **Dry Run (Check remaining files without generating)**:
  ```powershell
  python scripts/batch_sensual_star_board_krea2.py --dry-run
  ```
* **Process a small test batch (e.g. Next 10 images)**:
  ```powershell
  python scripts/batch_sensual_star_board_krea2.py --limit 10
  ```
* **Target specific files only**:
  ```powershell
  python scripts/batch_sensual_star_board_krea2.py --files "image1.jpg" "image2.png"
  ```
* **Disable 4x-UltraSharp 1080p Upscaling (Faster render-only)**:
  ```powershell
  python scripts/batch_sensual_star_board_krea2.py --no-upscale
  ```

---

## 🛡️ Architecture & Safety Notes
* **Atomic Resume Tracking**: Every completed image is committed immediately to `krea2_sensual_progress.json` via temporary atomic file swaps (`.tmp` -> `.json`). No progress will be lost if you reboot, cancel, or encounter a power cycle.
* **Auto-Reconnection**: If ComfyUI is restarted or takes a few seconds to launch, the runner automatically attempts to connect or launch `C:\ComfyUI\run_nvidia_gpu.bat`.
