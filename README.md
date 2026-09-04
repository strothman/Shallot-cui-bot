# 🧅 Shallot-CUI Bot — Discord AI Creation Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord.py-v2.3%2B-5865F2.svg)](https://github.com/Rapptz/discord.py)
[![ComfyUI API](https://img.shields.io/badge/ComfyUI-REST%20%26%20WS-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Tests](https://img.shields.io/badge/tests-54%20passed-success.svg)](suite_test.py)

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
Type `/imagine` followed by what you want to see. The bot will create a 2x2 grid with 4 different picture ideas!
* **Example:** `/imagine prompt: a cute ghost drinking boba tea in a neon city --ar 16:9`
* **Art Styles (Checkpoints):** Pick the overall look you want (like Anime, Photorealistic, or Semi-Realism).
* **Optional Magic Enhancements:**
  * `Powerhouse 2-Stage Refiner` (`--refine`): Runs a second cleanup pass to give characters smooth skin textures and rich micro-details.
  * `Face Detailer` (`--face`): Automatic face-cleanup pass so character eyes and faces are always sharp and symmetrical.
  * `Smart Art Director` (`--smart`): Analyzes your prompt's mood (Cyberpunk, Fantasy, Cozy, Retro) and automatically adds beautiful cinematic lighting.
  * `Magic Prompt` (`--magic`): Expands simple prompts with studio lighting and high-definition detail keywords.
  * `FreeU V2`: Enhances picture contrast and vivid colors.
* **Prompt Modifiers (Shortcuts):**
  * `{a|b|c}` $\rightarrow$ **Wildcards**: Give each of the 4 pictures a different choice! (e.g. `/imagine prompt: a cute {cat|fox|dragon|panda} wizard`).
  * `--ar 16:9` $\rightarrow$ Widescreen wallpaper shape.
  * `--ar 9:16` $\rightarrow$ Phone screen portrait shape.
  * `--ar 1920:1032` $\rightarrow$ Taskbar Fit (fits perfectly on your monitor right above the Windows taskbar).
  * `--sref <code|url>` $\rightarrow$ Style Reference: Copies the color palette and artistic vibe from an image or style code.
  * `--cref <url>` $\rightarrow$ Character Reference: Copies a character's face/identity from an uploaded photo.
  * `--valerie.85` $\rightarrow$ **Valerie**: Keeps the Valerie character look consistent across all your pictures.
  * `--sully.85` $\rightarrow$ **Sully**: Keeps the Sully character look consistent (black hair & thin-rim glasses).
  * `--ogarla.85` $\rightarrow$ **Ogarla**: Fantasy character preset.
  * `--sr.85` $\rightarrow$ Semi-Realism mode for rich 3D shading and lighting.
  * `--raw` $\rightarrow$ Clean, natural photo look without artistic filters.

---

### ✨ `/flux` — Ultra-Realistic Pictures (Flux.1)
Generate photorealistic pictures with incredible hands, natural skin, and clear, readable text on signs or clothes!
* **Model Choices:**
  * `flux1-dev` *(Default)*: Highest quality, realism, and accurate text.
  * `flux1-schnell`: Ultra-fast turbo version (creates pictures in just ~8 seconds!).
* **Characters:** Select `🌿 Ogarla Flux` directly from the `character` dropdown or use `--ogarla`.
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
[ ⭐ Favorite Style ]  [ ⭐ Favorite Prompt ]  [ 📋 Copy Prompt ]  [ ✏️ Remix ]
```

* 🛑 **Live Cancel Button**: Whenever a job is queued or running, click `🛑 Cancel` on the status message to immediately halt ComfyUI generation and clear pending tasks.

| Button | What It Does |
| :--- | :--- |
| **U1, U2, U3, U4** | **Isolate & AI Upscale**: Extracts that picture, makes it full size, and enhances all fine details. |
| **V1, V2, V3, V4** | **Variations**: Generates 4 new ideas based on that specific picture. |
| **🔄 (Re-roll)** | **Try Again**: Re-runs the exact same prompt with new random seeds for 4 fresh ideas. |
| **✏️ Remix** | **Tweak & Re-generate**: Opens a popup modal with your prompt and seed pre-filled so you can easily edit wording and try again. |
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
* **`/cui-status`**: Check if ComfyUI is online, view active jobs, and check your graphics card memory (VRAM).
* **🧹 Automatic Memory Cleaner**: The bot automatically frees up GPU memory whenever you switch between different model types (like SDXL and Flux).
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
| Make an image with Sully | `/imagine prompt: drinking coffee in a cafe --sully` |
| Make an image with Valerie | `/imagine prompt: walking in the park --valerie` |
| Stop a running render | Click the `🛑 Cancel` button on the status message |
| Tweak words from a picture grid | Click the `✏️ Remix` button below the pictures |
| Make a video from an image | `/video image: [upload] prompt: camera slowly zooms in duration: 5` |
| Fast 35-second animation | `/ltx image: [upload] prompt: camera slowly zooms in duration: 4` |
| Find out what prompt made a photo | `/study image: [upload]` |
| Try 4 outfits in 1 prompt | `/imagine prompt: a character wearing a {hoodie\|suit\|armor\|kimono}` |
| Copy style from another image | `/imagine prompt: knight on a hill style_reference: [upload]` |
| Scan for newly downloaded models | `/scan_models` |
| Start ComfyUI | `/cui-start` |

Have fun creating! 🎨✨
