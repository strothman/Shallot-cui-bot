# 🧅 Shallot-CUI Bot — Discord AI Creation Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord.py-v2.3%2B-5865F2.svg)](https://github.com/Rapptz/discord.py)
[![ComfyUI API](https://img.shields.io/badge/ComfyUI-REST%20%26%20WS-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/tests-49%20passed-success.svg)](suite_test.py)

Welcome! **Shallot-CUI Bot** is your personal AI art and video creation studio built directly into Discord, powered by **ComfyUI**.

Whether you want to create beautiful pictures, generate smooth videos, or copy styles from your favorite images, this guide explains how everything works in simple, easy-to-understand terms.

---

## 🚀 Quickstart: How to Run the Bot

1. **Configuration:** Copy [`.env.example`](.env.example) to `.env` and add your `DISCORD_TOKEN`.
2. **Start ComfyUI:** Make sure ComfyUI is running on your computer (or run `/cui-start` from Discord).
3. **Start the Bot:** Double-click `run_bot.bat` inside the `Shallot-cui-bot` folder.
4. **Start Creating:** Go to your Discord server and type `/` to see all the commands!

---

## 🎨 1. Creating Images

### 🌸 `/imagine` — Create 4 Pictures at Once (SDXL)
Type `/imagine` followed by your prompt description. The bot generates a 2x2 grid with 4 different variations!
* **Example:** `/imagine prompt: a cute ghost drinking boba tea in a neon city --ar 16:9`
* **Checkpoints:** Switch between Illustrious SDXL, RealVisXL, Animagine, and more.
* **Enhancements:**
  * `Powerhouse 2-Stage Refiner` (`--refine`): 2-stage generation pipeline adding rich skin textures and micro-details.
  * `Face Detailer` (`--face`): Automatic face-cleanup pass so character faces and eyes look extra sharp and symmetrical.
  * `Smart Art Director` (`--smart`): Automatically analyzes the mood/genre (Cyberpunk, Fantasy, Cozy, Retro) and adds matching lighting and details.
  * `Magic Prompt` (`--magic`): Auto-expands your prompt with studio lighting and high-definition details.
  * `FreeU V2`: Enhances contrast and high-frequency latents.
* **Prompt Modifiers:**
  * `{a|b|c}` $\rightarrow$ **Wildcards**: Give each of the 4 pictures a random choice! (e.g. `/imagine prompt: a cute {cat|fox|dragon|panda} wizard`).
  * `--ar 16:9` $\rightarrow$ Widescreen wallpaper shape.
  * `--ar 9:16` $\rightarrow$ Phone screen portrait shape.
  * `--ar 1920:1032` $\rightarrow$ Taskbar Fit (fits your monitor right above the Windows taskbar).
  * `--sref <code|url>` $\rightarrow$ Style Reference: Copies the color palette and artistic vibe from an image or style code.
  * `--cref <url>` $\rightarrow$ Character Reference: Copies a character's face/identity from an image.
  * `--sr.85` $\rightarrow$ Semi-Realism mode for rich shading and lighting.
  * `--raw` $\rightarrow$ Clean, natural photo look without extra artistic styling.

---

### ✨ `/flux` — Next-Gen Flow-Matching (Flux.1 GGUF)
Generate high-fidelity AI imagery using the 12-billion parameter **Flux.1** flow-matching model, quantized for smooth performance on 8GB VRAM!
* **Model Choices:**
  * `flux1-dev-Q4_K_S.gguf` *(Default)*: Maximum photorealism, accurate anatomy, sharp text, and artistic quality.
  * `flux1-schnell-Q4_K_S.gguf`: Ultra-fast turbo version (creates pictures in just ~8 seconds!).
* **Guidance:** Control prompt adherence and style fidelity (default: 3.5).
* **Example:** `/flux prompt: futuristic cyber warrior standing on a rooftop at sunset --ar 16:9`

---

## 🎬 2. Videos & Animation

Turn any still image into a smooth, animated video:

* **`/video` (Wan 2.2)**: High-quality AI video generator with smooth 60fps movement.
  * **Example:** `/video image: [upload picture] prompt: gentle wind blowing hair, blinking eyes, smiling duration: 5`
* **`/ltx` (LTX-Video)**: Super-fast video generator that renders animations in ~35 seconds.
  * **Example:** `/ltx image: [upload picture] prompt: camera slowly zooms in duration: 4`

---

## 🔍 3. Image Tools & Vision Studio

* **`/describe` (AI Vision)**: Upload any image and the AI will analyze it, write a prompt description for you, and give you 1-click buttons to remake it in different styles or shapes!
* **`/study` (Read Hidden Prompts)**: Upload any AI picture you found on the web. The bot inspects the hidden file data and extracts the exact prompt used to make it!
* **`/blend` (Image Blend Studio)**: Upload an image to mix it with new styles, checkpoints, and text ideas.
* **`/upscale` (1920px AI Upscaler)**: Upload any picture to make it sharp and high-resolution.

---

## 🖱️ 4. Right-Click Quick Apps

You can skip typing slash commands entirely! In Discord, **right-click any picture** (or hold down on mobile) and hover over **Apps**:

* **`🎬 Animate to Video`**: Opens a quick popup window to turn that picture into a video!
* **`🎨 Blend Image`**: Opens the Blend Studio to remix the picture with other styles.
* **`📥 Adopt Post / Image`**: Extracts the prompt and settings from an image post so you can tweak it.
* **`⛵ Adopt Midjourney Post`**: Turns Midjourney posts into Shallot-CUI bot commands.

---

## 🎛️ 5. Interactive Buttons Under Every 4-Image Grid

Whenever the bot generates a 4-image grid, you'll see these buttons below it:

```text
[ U1 ]  [ U2 ]  [ U3 ]  [ U4 ]
[ V1 ]  [ V2 ]  [ V3 ]  [ V4 ]  [ 🔄 ]
[ ⭐ Favorite Style ]  [ ⭐ Favorite Prompt ]  [ 📋 Copy Prompt ]
```

| Button | What It Does |
| :--- | :--- |
| **U1, U2, U3, U4** | **Isolate & AI Upscale**: Extracts that picture, makes it full size, and enhances all fine details. |
| **V1, V2, V3, V4** | **Variations**: Generates 4 new ideas based on that specific picture. |
| **🔄 (Re-roll)** | **Try Again**: Re-runs the exact same prompt with new random seeds for 4 fresh ideas. |
| **⭐ Favorite Style** | Saves the image's style code to your personal favorites library. |
| **⭐ Favorite Prompt** | Saves the prompt text to your personal favorites library. |
| **📋 Copy Prompt** | Pops up a private message with the full prompt text so you can easily copy and edit it. |

---

## 📌 6. Managing Prompts, Styles & Negatives

* **`/prompt list`**: Browse, copy, edit, or launch your saved prompts with interactive buttons.
* **`/prompt save name: Neon Dragon prompt:...`**: Save a prompt with a nickname.
* **`/prompt edit prompt_id: 1`**: Edit an existing saved prompt.
* **`/prompt delete prompt_id: 1`**: Remove a prompt from your favorites.
* **`/style list`**: Browse your saved `--sref` style codes.
* **`/style batch prompt:... count: 5`**: Queue 5, 10, or 15 generations at once, testing different random or favorite styles!
* **`/negative`**: View, edit, or reset your default negative prompt.

---

## ⚙️ 7. Server Controls & Management

* **`/cui-start`**: Start your local ComfyUI server from Discord.
* **`/cui-stop`**: Safely stop ComfyUI and free your graphics card memory.
* **`/cui-status`**: Check if ComfyUI is online, view active jobs, and check your GPU VRAM.
* **`/queue`**: View active rendering jobs with visual progress bars.
* **`/models`**: View all registered Checkpoints and LoRAs grouped by architecture (SDXL, Flux, Wan, LTX).
* **`/scan_models`**: One-click scanner that scans your ComfyUI models folder and adds newly downloaded checkpoints and LoRAs to the bot.
* **`/variation_mode`**: Toggle variation strength between **High** and **Very High**.
* **`/diagnostics`**: View render speeds, GPU stats, and recent generations.
* **❌ Delete Any Message**: React with the **❌** (red X) emoji on any bot message to instantly delete it.

---

## 📂 Where Your Files Are Saved

All high-resolution outputs are saved directly to your computer:
* 🖼️ **High-Res Images:** `C:\ComfyUI\ComfyUI\output\Discord Bot\highres`
* ✨ **Flux Images:** `C:\ComfyUI\ComfyUI\output\Discord Bot\flux`

---

## 💡 Quick Cheat Sheet

| What I want to do | Command Example |
| :--- | :--- |
| Generate 4 ideas for a wallpaper | `/imagine prompt: cyberpunk city in the rain --smart --ar 16:9` |
| Make an ultra-realistic picture | `/flux prompt: portrait of an astronaut on Mars --smart --ar 16:9` |
| Make a video from an image | `/video image: [upload] prompt: camera slowly zooms in duration: 5` |
| Fast 35-second animation | `/ltx image: [upload] prompt: camera slowly zooms in duration: 4` |
| Find out what prompt made a photo | `/study image: [upload]` |
| Try 4 outfits in 1 prompt | `/imagine prompt: a character wearing a {hoodie\|suit\|armor\|kimono}` |
| Copy style from another image | `/imagine prompt: knight on a hill style_reference: [upload]` |
| Scan for newly downloaded models | `/scan_models` |
| Start ComfyUI | `/cui-start` |

Have fun creating! 🎨✨
