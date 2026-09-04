# 🧅 PROJECT STATE — Shallot-CUI Bot

> **Project Name:** Shallot-CUI Bot (*Your Discord AI Creation Studio*)  
> **Current Version:** `v2.4.0`  
> **Last Updated:** September 4, 2026  
> **Status:** 🟢 Stable & Healthy (54/54 Automated Tests Passing)  

---

## 📌 1. What Is Shallot-CUI Bot? (In Plain English)

Think of **Shallot-CUI Bot** as your own private version of Midjourney that runs right inside your Discord server, powered by your computer's graphics card!

* **How it works:**
  1. You type a command in Discord, like `/imagine prompt: a cozy cottage in autumn woods --ar 16:9`.
  2. The bot translates your text and sends it to **ComfyUI** (the AI art generator running on your PC).
  3. Your graphics card paints the pictures.
  4. The bot posts a neat **4-image grid** back to Discord, with easy buttons to upscale, create variations, or remix your prompt!

You don't need to know any programming or complex AI jargon to use it. The bot handles all the complicated math and settings behind the scenes.

---

## 🏗️ 2. Project Files & What They Do

Here is a simple breakdown of the main files in the project and what each one is responsible for:

| File | What It Does (Plain English) |
| :--- | :--- |
| [`bot.py`](bot.py) | **The Front Desk:** Listens to Discord messages, handles slash commands (`/imagine`, `/flux`, `/video`), and coordinates tasks. |
| [`characters.py`](characters.py) | **Character Wardrobe:** Stores character presets like **Valerie**, **Sully**, and **Ogarla**. Automatically applies their hair/glasses and protects real-person privacy. |
| [`parsers.py`](parsers.py) | **Prompt Translator:** Reads flags like `--ar 16:9` (widescreen), `--smart` (auto-lighting), `--sref` (style copy), and wildcards `{a\|b\|c}`. |
| [`views.py`](views.py) | **Interactive Buttons:** Creates all clickable buttons in Discord (U1–U4, V1–V4, `🛑 Cancel`, and `✏️ Remix` popup windows). |
| [`comfy_client.py`](comfy_client.py) | **The Messenger:** Talks to ComfyUI on your computer, tracks render progress, and automatically frees GPU memory when needed. |
| [`image_utils.py`](image_utils.py) | **Image Crafter:** Stitches the 4 pictures into a 2x2 grid, cuts out individual images for upscaling, and optimizes file sizes. |
| [`db.py`](db.py) | **Memory & Notebook:** An SQLite database (`cache.db`) that remembers your favorite prompts, style codes, and past creations. |
| [`config.py`](config.py) | **Settings & Guardrails:** Stores default models, safety limits, and admin permissions so only server owners can run sensitive controls. |
| [`suite_test.py`](suite_test.py) | **Safety Inspector:** An automated test runner that checks 54 different parts of the bot to make sure nothing is broken. |
| [`auto_changelog.py`](auto_changelog.py) | **Secretary:** Keeps the [CHANGELOG.md](CHANGELOG.md) updated so you always know what was added or changed. |
| [`workflows/`](workflows/) | **Recipe Book:** Pre-built ComfyUI recipes for SDXL, Flux.1, Wan 2.2 video, and high-resolution upscaling. |

---

## ⚡ 3. Current Features & Commands

### 🎨 Image Generation
* **`/imagine`**: Creates a 2x2 grid of 4 pictures using SDXL. Supports aspect ratios (`--ar`), style references (`--sref`), and character presets.
* **`/flux`**: Creates ultra-detailed, photographic pictures using the next-generation **Flux.1** AI model.
* **`/blend`**: Blends 2 to 5 different images together into a brand new creation.

### 🎭 Character Presets (Consistent Faces)
* **Valerie (`--valerie.85`)**: Keeps the Valerie character look consistent across prompts.
* **Sully (`--sully.85`)**: Automatically adds Sully's signature black hair and thin-rim glasses.
* **Ogarla (`--ogarla.85`)**: Fantasy character preset.
* 🛡️ **Privacy Shield:** All character presets automatically disguise private trigger names so real identities are never exposed in Discord.

### 🎬 Video & Animation
* **`/video`**: Turns any still picture into an animated video with matching sound effects (using **Wan 2.2** + **MMAudio**).
* **`/ltx`**: Creates a fast 35-second animation using **LTX-Video**.

### 🔍 Vision & Image Tools
* **`/describe`**: Upload any picture and AI will analyze it and write a prompt for you.
* **`/study`**: Upload an AI image found online to extract the secret prompt used to make it.
* **`/upscale`**: Makes any picture bigger and sharper with extra detail.

### 🎛️ Interactive Discord Buttons
Under every 4-image grid, you get 1-click buttons:
* **`[ U1 ] [ U2 ] [ U3 ] [ U4 ]`**: Isolate and upscale picture #1, #2, #3, or #4 to full size.
* **`[ V1 ] [ V2 ] [ V3 ] [ V4 ]`**: Make 4 new variations based on that specific picture.
* **`[ 🔄 ]`**: Re-roll the exact same prompt with fresh random seeds.
* **`[ ✏️ Remix ]`**: Opens a popup with your prompt pre-filled so you can easily change a few words and try again.
* **`[ 🛑 Cancel ]`**: Instantly stop a running generation if you change your mind.

### ⚙️ Server Controls
* **`/cui-start` & `/cui-stop`**: Turn your local ComfyUI engine on or off directly from Discord.
* **`/cui-status`**: Check if ComfyUI is online and see how much graphics card memory (VRAM) is free.
* **`/models` & `/scan_models`**: View all installed AI models and scan for newly downloaded ones with 1 click.
* **`/queue`**: Check what jobs are currently rendering.

---

## 🌟 4. What's New in v2.4.0

1. **New Character Presets:** Added **Valerie** and **Sully** with built-in privacy protection and automatic trait injection (e.g., glasses and hair).
2. **Live Cancel Button (`🛑 Cancel`):** You can now stop any running render without having to open the ComfyUI console.
3. **Grid Remix Modal (`✏️ Remix`):** One-click button to tweak prompts directly from Discord popups.
4. **Automatic Memory Cleaning (VRAM Auto-Purge):** The bot automatically frees graphics card memory when switching between SDXL, Flux, and video models so your computer never crashes from low memory.
5. **Simplified & Cleaner Menu:** Removed bloated, rarely-used tools (like the experimental LoRA builder) to keep the bot lean, fast, and easy to use.
6. **Double-Launch Warning:** The bot alerts you if you accidentally start two bot windows, preventing double generations.

---

## 🧪 5. Testing & Quality Assurance

Every time you run the bot using `run_bot.bat`, it performs an automatic safety check:
* **Automated Tests:** **54 / 54 tests passing** (`python suite_test.py`).
* **What is tested:**
  * Aspect ratio math and sizing.
  * Wildcard randomization (`{cat|dog|fox}`).
  * Character preset trigger substitution and privacy masking.
  * Interactive buttons and Remix popup modals.
  * Model compatibility and graphics card memory cleanup.
  * Documentation sync between Discord commands and README.

---

## 🚀 6. How to Start Everything

1. Make sure your `.env` file has your `DISCORD_TOKEN`.
2. Start ComfyUI on your computer (or run `/cui-start` in Discord).
3. Double-click **`run_bot.bat`**.
4. Head into Discord and type `/imagine` to start creating! 🎨
