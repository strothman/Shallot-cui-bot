"""
System & Administration Cog for Shallot-CUI Bot.
Houses slash commands for:
- ComfyUI server lifecycle: /cui-start, /cui-stop, /cui-status
- GPU memory & queue management: /free, /purge-vram, /queue
- Telemetry & performance benchmarks: /diagnostics
- Model registry and discovery: /models, /scan_models
- Personalization settings: /negative, /variation_mode, /prompt, /style
"""

import os
import sys
import logging
import asyncio
import subprocess
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from config import (
    COMFYUI_ADDRESS, 
    COMFYUI_BATCH_PATH, 
    VRAM_MIN_FREE_GB, 
    is_authorized_admin
)
from core_helpers import (
    safe_defer, 
    check_gpu_vram_caution, 
    send_followup_fallback,
    edit_message_fallback
)
import db
import model_architecture
from views import (
    PromptPaginationView, 
    EditPromptModal, 
    StylePaginationView, 
    EditStyleModal
)
from services.system_service import (
    settings,
    save_settings,
    terminate_existing_comfyui,
    fetch_comfyui_queue,
    fetch_comfyui_system_stats,
    build_queue_embed,
    build_diagnostics_embed,
    build_models_embed,
    purge_vram_core
)

logger = logging.getLogger("DiscordBot.SystemCog")


class NegativePromptView(discord.ui.View):
    """Interactive view for editing or resetting personal negative prompt."""

    def __init__(self, owner_id: int, current_neg: str):
        super().__init__(timeout=180)
        self.owner_id = owner_id
        self.current_neg = current_neg

    @discord.ui.button(label="✏️ Edit Negative Prompt", style=discord.ButtonStyle.primary)
    async def edit_button(self, btn_interaction: discord.Interaction, button: discord.ui.Button):
        if btn_interaction.user.id != self.owner_id:
            await btn_interaction.response.send_message("Only the command requester can edit this.", ephemeral=True)
            return

        current_val = self.current_neg

        class EditNegModal(discord.ui.Modal, title="Edit Negative Prompt"):
            neg_input = discord.ui.TextInput(
                label="Negative Prompt Text",
                style=discord.TextStyle.paragraph,
                default=current_val,
                max_length=1000
            )

            async def on_submit(self, modal_interaction: discord.Interaction):
                db.set_negative_prompt(modal_interaction.user.id, self.neg_input.value)
                new_embed = discord.Embed(
                    title="✅ Negative Prompt Updated",
                    description=f"```text\n{self.neg_input.value}\n```",
                    color=discord.Color.green()
                )
                await modal_interaction.response.send_message(embed=new_embed, ephemeral=True)

        await btn_interaction.response.send_modal(EditNegModal())

    @discord.ui.button(label="🔄 Reset to Default", style=discord.ButtonStyle.secondary)
    async def reset_button(self, btn_interaction: discord.Interaction, button: discord.ui.Button):
        if btn_interaction.user.id != self.owner_id:
            await btn_interaction.response.send_message("Only the command requester can reset this.", ephemeral=True)
            return
        db.reset_negative_prompt(btn_interaction.user.id)
        reset_embed = discord.Embed(
            title="🔄 Negative Prompt Reset to Default",
            description=f"```text\n{db.DEFAULT_NEGATIVE_PROMPT}\n```",
            color=discord.Color.blue()
        )
        await btn_interaction.response.send_message(embed=reset_embed, ephemeral=True)


class SystemCog(commands.Cog):
    """Cog handling administration, server lifecycle, memory cleanup, and system settings."""

    # Define /prompt command group
    prompt_group = app_commands.Group(name="prompt", description="Manage your saved favorite prompts")

    # Define /style command group
    style_group = app_commands.Group(name="style", description="Manage your favorite ComfyUI style references")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.comfy_process = None

    def _get_comfy_client(self):
        return getattr(self.bot, "comfy_client", None)

    async def _run_imagine(self, interaction: discord.Interaction, prompt: str):
        bot = interaction.client
        imagine_func = getattr(bot, "execute_imagine", None)
        if not imagine_func:
            import bot as bot_module
            imagine_func = getattr(bot_module, "execute_imagine", None)
        if imagine_func:
            await imagine_func(interaction, prompt=prompt)
        else:
            await interaction.followup.send("❌ Image generation handler unavailable.", ephemeral=True)

    async def _run_bertflow(self, interaction: discord.Interaction, prompt: str):
        await safe_defer(interaction, thinking=True)
        from services.krea_service import execute_bertflow
        await execute_bertflow(interaction, prompt=prompt)

    # =========================================================================
    # /prompt Group Commands
    # =========================================================================

    @prompt_group.command(name="list", description="View and manage your saved favorite prompts.")
    async def prompt_list(self, interaction: discord.Interaction):
        prompts = db.get_favorite_prompts(interaction.user.id)
        if not prompts:
            await interaction.response.send_message(
                "You have no saved favorite prompts yet! Click **⭐ Favorite Prompt** on any generation or use `/prompt save`.", 
                ephemeral=True
            )
            return

        async def prompt_imagine_callback(inter: discord.Interaction, prompt_text: str):
            await self._run_imagine(inter, prompt_text)

        async def prompt_bertflow_callback(inter: discord.Interaction, prompt_text: str):
            await self._run_bertflow(inter, prompt_text)

        view = PromptPaginationView(
            interaction.user.id, 
            prompts, 
            per_page=5, 
            imagine_callback=prompt_imagine_callback, 
            bertflow_callback=prompt_bertflow_callback
        )
        await interaction.response.send_message(embed=view.build_embed(), view=view, ephemeral=True)

    @prompt_group.command(name="save", description="Save a custom prompt to your favorite prompts.")
    @app_commands.describe(
        name="Short nickname/label for this prompt",
        prompt="The prompt text to save"
    )
    async def prompt_save(self, interaction: discord.Interaction, name: str, prompt: str):
        db.add_favorite_prompt(interaction.user.id, name, prompt)
        await interaction.response.send_message(f"⭐ Saved prompt **\"{name}\"** to your favorites (`/prompt list`)!", ephemeral=True)

    @prompt_group.command(name="edit", description="Edit a saved favorite prompt's name or text.")
    @app_commands.describe(prompt_id="The ID of the prompt to edit (check /prompt list)")
    async def prompt_edit(self, interaction: discord.Interaction, prompt_id: int):
        prompts = db.get_favorite_prompts(interaction.user.id)
        selected = next((p for p in prompts if p["id"] == prompt_id), None)
        if not selected:
            await interaction.response.send_message(f"Prompt ID **{prompt_id}** is not in your favorites list.", ephemeral=True)
            return
            
        modal = EditPromptModal(interaction.user.id, selected)
        await interaction.response.send_modal(modal)

    @prompt_edit.autocomplete('prompt_id')
    async def prompt_edit_autocomplete(self, interaction: discord.Interaction, current: str):
        prompts = db.get_favorite_prompts(interaction.user.id)
        choices = []
        for p in prompts:
            label = f"{p['prompt_name']} (ID: {p['id']})"
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=p['id']))
        return choices[:25]

    @prompt_group.command(name="delete", description="Delete a saved prompt from your favorites.")
    @app_commands.describe(prompt_id="The ID of the prompt to delete (check /prompt list)")
    async def prompt_delete(self, interaction: discord.Interaction, prompt_id: int):
        db.remove_favorite_prompt(interaction.user.id, prompt_id)
        await interaction.response.send_message(f"🗑️ Deleted prompt ID **{prompt_id}** from your favorites.", ephemeral=True)

    @prompt_delete.autocomplete('prompt_id')
    async def prompt_delete_autocomplete(self, interaction: discord.Interaction, current: str):
        prompts = db.get_favorite_prompts(interaction.user.id)
        choices = []
        for p in prompts:
            label = f"{p['prompt_name']} (ID: {p['id']})"
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=p['id']))
        return choices[:25]

    # =========================================================================
    # /style Group Commands
    # =========================================================================

    @style_group.command(name="list", description="List and manage your saved favorite style codes.")
    async def style_list(self, interaction: discord.Interaction):
        await safe_defer(interaction, ephemeral=True)
        favorites = db.get_favorite_styles(interaction.user.id)
        
        if not favorites:
            await interaction.followup.send("You have no saved style references yet. Click `⭐ Favorite Style` on any completed image grid with a style reference!", ephemeral=True)
            return
            
        view = StylePaginationView(interaction.user.id, favorites, per_page=8)
        await interaction.followup.send(embed=view.build_embed(), view=view, ephemeral=True)

    @style_group.command(name="edit", description="Edit a saved style's name or prompt.")
    @app_commands.describe(code="The style reference code to edit")
    async def style_edit(self, interaction: discord.Interaction, code: int):
        favorites = db.get_favorite_styles(interaction.user.id)
        selected = next((fav for fav in favorites if fav["style_code"] == code), None)
        
        if not selected:
            await interaction.response.send_message(f"Code `{code}` is not in your favorites list.", ephemeral=True)
            return
            
        modal = EditStyleModal(interaction.user.id, selected)
        await interaction.response.send_modal(modal)

    @style_edit.autocomplete('code')
    async def style_edit_autocomplete(self, interaction: discord.Interaction, current: str):
        favorites = db.get_favorite_styles(interaction.user.id)
        choices = []
        for fav in favorites:
            label = f"{fav['style_name']} ({fav['style_code']})"
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=fav['style_code']))
        return choices[:25]

    @style_group.command(name="remove", description="Remove a style code from your favorites.")
    @app_commands.describe(code="The 6-digit style reference code to remove")
    async def style_remove(self, interaction: discord.Interaction, code: int):
        await safe_defer(interaction, ephemeral=True)
        favorites = db.get_favorite_styles(interaction.user.id)
        codes = [fav["style_code"] for fav in favorites]
        
        if code not in codes:
            await interaction.followup.send(f"Code `{code}` is not in your favorites list.", ephemeral=True)
            return
            
        db.remove_favorite_style(interaction.user.id, code)
        await interaction.followup.send(f"❌ Removed style code `{code}` from your favorites.", ephemeral=True)

    @style_remove.autocomplete('code')
    async def style_remove_autocomplete(self, interaction: discord.Interaction, current: str):
        favorites = db.get_favorite_styles(interaction.user.id)
        choices = []
        for fav in favorites:
            label = f"{fav['style_name']} ({fav['style_code']})"
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=fav['style_code']))
        return choices[:25]

    @style_group.command(name="batch", description="Queue 5, 10, or 15 generations with style codes from your /styles list or random.")
    @app_commands.describe(
        prompt="The prompt to generate images from",
        count="Number of style generations to queue (5, 10, or 15)",
        aspect_ratio="Aspect ratio for generated images (--ar)"
    )
    @app_commands.choices(
        count=[
            app_commands.Choice(name="5 Styles", value=5),
            app_commands.Choice(name="10 Styles", value=10),
            app_commands.Choice(name="15 Styles", value=15),
        ],
        aspect_ratio=[
            app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
            app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
            app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
            app_commands.Choice(name="10:7 (iPad)", value="10:7"),
            app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
            app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
        ]
    )
    async def style_batch(self, interaction: discord.Interaction, prompt: str, count: int = 5, aspect_ratio: str = None):
        await safe_defer(interaction, thinking=True)
        full_prompt = f"{prompt} --sref batch:{count}"
        if aspect_ratio:
            full_prompt = f"{full_prompt} --ar {aspect_ratio}"
        await self._run_imagine(interaction, prompt=full_prompt)

    # =========================================================================
    # VRAM & Memory Management
    # =========================================================================

    @app_commands.command(name="free", description="🧹 Purge ComfyUI model weights and free GPU VRAM immediately.")
    async def free_vram(self, interaction: discord.Interaction):
        """Frees loaded models and purges PyTorch CUDA cache on the local ComfyUI instance."""
        await safe_defer(interaction, thinking=True)
        try:
            client = self._get_comfy_client()
            success = await purge_vram_core(client=client)
            if success:
                embed = discord.Embed(
                    title="🧹 ComfyUI VRAM Purged",
                    description="Successfully unloaded all models from GPU VRAM and cleared PyTorch memory caches.",
                    color=discord.Color.green()
                )
                embed.add_field(name="Status", value="✅ Ready for fresh generation or gaming / system tasks.", inline=False)
                embed.set_footer(text="Shallot ComfyUI VRAM Manager • 8GB Low-VRAM Hygiene")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("⚠️ ComfyUI `/free` endpoint did not respond with 200 OK. Verify ComfyUI is running.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error purging VRAM via /free: {e}")
            await interaction.followup.send(f"Failed to purge VRAM: {e}", ephemeral=True)

    @app_commands.command(name="purge-vram", description="🧹 Alias for /free - Purge ComfyUI models & GPU memory.")
    async def purge_vram(self, interaction: discord.Interaction):
        await self.free_vram(interaction)

    @app_commands.command(name="queue", description="Show the current ComfyUI processing queue and system status.")
    async def queue_status(self, interaction: discord.Interaction):
        """Display ComfyUI queue status, VRAM usage, and job details."""
        await safe_defer(interaction, ephemeral=True)
        queue = await fetch_comfyui_queue()
        stats = await fetch_comfyui_system_stats()
        embed = build_queue_embed(queue, stats)
        await interaction.followup.send(embed=embed, ephemeral=True)

    # =========================================================================
    # Telemetry & Diagnostics
    # =========================================================================

    @app_commands.command(name="diagnostics", description="View bot generation metrics, speed benchmarks, and troubleshooting data.")
    async def diagnostics(self, interaction: discord.Interaction):
        """Displays telemetry benchmarks, average render times, and recent error diagnostics."""
        await safe_defer(interaction, ephemeral=False)
        embed = build_diagnostics_embed(interaction.user.name)
        await send_followup_fallback(interaction, embed=embed)

    # =========================================================================
    # Model Architecture Registry & Discovery
    # =========================================================================

    @app_commands.command(name="models", description="📦 View all registered checkpoints, LoRAs, and their suited architectures.")
    @app_commands.describe(
        architecture="Filter models by architecture (e.g., sdxl, flux, wan, ltx)",
        model_type="Filter by model type (checkpoint, lora, unet)"
    )
    @app_commands.choices(
        architecture=[
            app_commands.Choice(name="All Architectures", value="all"),
            app_commands.Choice(name="🎨 SDXL", value="sdxl"),
            app_commands.Choice(name="⚡ FLUX", value="flux"),
            app_commands.Choice(name="🎬 WAN (Video)", value="wan"),
            app_commands.Choice(name="🎥 LTX (Video)", value="ltx"),
        ],
        model_type=[
            app_commands.Choice(name="All Types", value="all"),
            app_commands.Choice(name="Checkpoints", value="checkpoint"),
            app_commands.Choice(name="LoRAs", value="lora"),
        ]
    )
    async def models(self, interaction: discord.Interaction, architecture: str = "all", model_type: str = "all"):
        embed = build_models_embed(architecture=architecture, model_type=model_type)
        if not embed:
            await interaction.response.send_message(
                f"No registered models found matching filter `architecture={architecture}`, `type={model_type}`.", 
                ephemeral=True
            )
            return
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="scan_models", description="🔄 Scan local ComfyUI model directories and auto-register new models & LoRAs.")
    async def scan_models(self, interaction: discord.Interaction):
        if not is_authorized_admin(interaction):
            await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can scan and register models.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        stats = await asyncio.to_thread(model_architecture.scan_and_register_comfyui_models)
        
        if not stats.get("success"):
            await interaction.followup.send(f"❌ Scan failed: {stats.get('error')}", ephemeral=True)
            return
            
        embed = discord.Embed(
            title="🔍 Model & LoRA Auto-Discovery Complete",
            description="Scanned local ComfyUI model directories and updated SQLite architecture registry.",
            color=discord.Color.green()
        )
        embed.add_field(name="📁 Files Scanned", value=str(stats.get("scanned", 0)), inline=True)
        embed.add_field(name="🏛️ Checkpoints", value=str(stats.get("checkpoints", 0)), inline=True)
        embed.add_field(name="⚡ LoRAs", value=str(stats.get("loras", 0)), inline=True)
        embed.add_field(name="🟠 UNets / Diffusion", value=str(stats.get("unets", 0)), inline=True)
        embed.add_field(name="✅ Total Registered", value=str(stats.get("total_registered", 0)), inline=True)
        
        await interaction.followup.send(embed=embed, ephemeral=True)

    # =========================================================================
    # User Preferences & Settings
    # =========================================================================

    @app_commands.command(name="variation_mode", description="Toggle variation strength between 'High' (0.85 denoise) and 'Very High' (0.95 denoise).")
    async def variation_mode(self, interaction: discord.Interaction):
        """Toggle default variation mode (High vs Very High) persistently."""
        current = settings.get("variation_mode", "high")
        new_mode = "very_high" if current == "high" else "high"
        settings["variation_mode"] = new_mode
        save_settings()
        
        mode_label = "🔥 Very High (Denoise: 0.95)" if new_mode == "very_high" else "⚡ High (Denoise: 0.85)"
        embed = discord.Embed(
            title="Variation Mode Updated",
            description=f"Variation buttons will now use **{mode_label}**.",
            color=discord.Color.green()
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="negative", description="🚫 View and manage your active negative prompt for /imagine.")
    @app_commands.describe(prompt="Optional new negative prompt text to set immediately")
    async def negative(self, interaction: discord.Interaction, prompt: str = None):
        if prompt:
            db.set_negative_prompt(interaction.user.id, prompt)
            embed = discord.Embed(
                title="✅ Negative Prompt Updated",
                description=f"All future `/imagine` generations will use:\n```text\n{prompt}\n```",
                color=discord.Color.green()
            )
            await interaction.response.send_message(embed=embed)
            return

        current_neg = db.get_negative_prompt(interaction.user.id)
        is_default = (current_neg == db.DEFAULT_NEGATIVE_PROMPT)
        
        embed = discord.Embed(
            title="🚫 Active Negative Prompt (/imagine)",
            description=f"```text\n{current_neg}\n```",
            color=discord.Color.blue() if is_default else discord.Color.gold()
        )
        embed.add_field(name="Status", value="⚙️ Default System Prompt" if is_default else "🎨 Custom User Prompt", inline=True)
        embed.set_footer(text="Use /negative prompt: to customize or click Edit/Reset below.")

        view = NegativePromptView(owner_id=interaction.user.id, current_neg=current_neg)
        await interaction.response.send_message(embed=embed, view=view)

    # =========================================================================
    # ComfyUI Server Remote Control
    # =========================================================================

    @app_commands.command(name="cui-start", description="Start or restart a fresh instance of the local ComfyUI server.")
    @app_commands.describe(force="Force start even if high VRAM usage (Tdarr) is detected")
    async def cui_start(self, interaction: discord.Interaction, force: bool = False):
        if not is_authorized_admin(interaction):
            await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can start or manage the ComfyUI server.", ephemeral=True)
            return

        await safe_defer(interaction, thinking=True)

        client = self._get_comfy_client()
        was_running = False
        if client:
            was_running = await client.is_online()

        killed = terminate_existing_comfyui(self.comfy_process)
        if was_running or killed:
            await asyncio.sleep(2)

        is_caution, vram_info = await asyncio.to_thread(check_gpu_vram_caution)
        if is_caution and not force:
            gpu_name = vram_info.get("name", "NVIDIA GPU")
            pct = vram_info.get("percent_used", 0)
            used = vram_info.get("used_gb", 0)
            total = vram_info.get("total_gb", 0)
            free = vram_info.get("free_gb", 0)
            embed = discord.Embed(
                title="⚠️ CAUTION: High VRAM Load Detected",
                description=(
                    f"GPU **{gpu_name}** is under heavy load (likely Tdarr transcoding or external application).\n\n"
                    f"• **VRAM Usage:** `{pct:.1f}%` (`{used:.2f} GB` / `{total:.2f} GB`)\n"
                    f"• **Free VRAM:** `{free:.2f} GB` (Min threshold: `{VRAM_MIN_FREE_GB} GB`)\n\n"
                    "⛔ **Server launch was halted** to prevent Out-Of-Memory crashes.\n\n"
                    "💡 *To force launch anyway, run:* `/cui-start force:True`"
                ),
                color=discord.Color.gold()
            )
            await interaction.followup.send(embed=embed)
            return

        if not os.path.exists(COMFYUI_BATCH_PATH):
            await interaction.followup.send(f"❌ Could not find batch file at `{COMFYUI_BATCH_PATH}`. Please check configuration.")
            return

        comfy_dir = os.path.dirname(COMFYUI_BATCH_PATH)
        try:
            if os.name == 'nt':
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 6  # SW_MINIMIZE
                self.comfy_process = await asyncio.create_subprocess_exec(
                    "cmd.exe", "/c", COMFYUI_BATCH_PATH,
                    cwd=comfy_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    startupinfo=si
                )
            else:
                self.comfy_process = await asyncio.create_subprocess_exec(
                    COMFYUI_BATCH_PATH,
                    cwd=comfy_dir
                )

            self.bot.comfy_process = self.comfy_process
            logger.info(f"Launched fresh ComfyUI server batch file (PID: {self.comfy_process.pid})")
            action_msg = "🔄 **Restarting ComfyUI Server (Fresh Instance)...**" if (was_running or killed) else "⏳ **Starting ComfyUI Server...**"
            await interaction.followup.send(f"{action_msg} Please wait while models load.")
        except Exception as e:
            logger.error(f"Failed to start ComfyUI server: {e}")
            await interaction.followup.send(f"❌ Failed to start ComfyUI server: `{e}`")
            return

        # Poll server online status for up to 45 seconds
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 45:
            await asyncio.sleep(3)
            if client and await client.is_online():
                embed = discord.Embed(
                    title="🚀 ComfyUI Server Started (Fresh Instance)!",
                    description=f"Server is online, clean, and responding at `http://{COMFYUI_ADDRESS}`.",
                    color=discord.Color.green()
                )
                await interaction.followup.send(embed=embed)
                return

        embed = discord.Embed(
            title="⏳ ComfyUI Starting (Extended Load)",
            description=f"The process was launched (PID: {self.comfy_process.pid}), but the server is still initializing models. Check status again in a moment!",
            color=discord.Color.gold()
        )
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="cui-stop", description="Stop the local ComfyUI server remotely.")
    async def cui_stop(self, interaction: discord.Interaction):
        if not is_authorized_admin(interaction):
            await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can stop the ComfyUI server.", ephemeral=True)
            return

        await safe_defer(interaction, thinking=True)

        client = self._get_comfy_client()
        is_currently_online = await client.is_online() if client else False
        if not is_currently_online and (self.comfy_process is None or self.comfy_process.returncode is not None):
            killed = terminate_existing_comfyui(self.comfy_process)
            if not killed:
                await interaction.followup.send("🔴 **ComfyUI Server is already offline.**")
                return

        terminate_existing_comfyui(self.comfy_process)
        self.comfy_process = None
        self.bot.comfy_process = None

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < 10:
            await asyncio.sleep(1)
            if client and not await client.is_online():
                embed = discord.Embed(
                    title="🛑 ComfyUI Server Stopped",
                    description="The ComfyUI server has been successfully shut down.",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed)
                return

        if client and not await client.is_online():
            embed = discord.Embed(
                title="🛑 ComfyUI Server Stopped",
                description="The ComfyUI server process was terminated.",
                color=discord.Color.red()
            )
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("⚠️ Sent termination command, but server may still be shutting down. Check status again in a moment.")

    @app_commands.command(name="cui-status", description="Check the current status, GPU VRAM, and queue of the ComfyUI server.")
    async def cui_status(self, interaction: discord.Interaction):
        await safe_defer(interaction, thinking=True)

        client = self._get_comfy_client()
        is_online = await client.is_online() if client else False

        if not is_online:
            embed = discord.Embed(
                title="🔴 ComfyUI Server Status: OFFLINE",
                description=f"The server at `http://{COMFYUI_ADDRESS}` is currently offline.\n\nUse `/cui-start` to launch the server.",
                color=discord.Color.red()
            )
            if self.comfy_process and self.comfy_process.returncode is None:
                embed.add_field(name="Process State", value=f"Process launched (PID: `{self.comfy_process.pid}`), initializing models...", inline=False)
            await interaction.followup.send(embed=embed)
            return

        embed = discord.Embed(
            title="🟢 ComfyUI Server Status: ONLINE",
            description=f"Server is running and responsive at `http://{COMFYUI_ADDRESS}`.",
            color=discord.Color.green()
        )

        if self.comfy_process and self.comfy_process.returncode is None:
            embed.add_field(name="Process PID", value=f"`{self.comfy_process.pid}`", inline=True)

        queue_data = await client.get_queue() if client else None
        if queue_data:
            running = len(queue_data.get("queue_running", []))
            pending = len(queue_data.get("queue_pending", []))
            embed.add_field(name="Active Jobs", value=f"**{running}** running", inline=True)
            embed.add_field(name="Queue Depth", value=f"**{pending}** pending", inline=True)

        stats_data = await client.get_system_stats() if client else None
        if stats_data and "devices" in stats_data:
            for idx, dev in enumerate(stats_data["devices"]):
                dev_name = dev.get("name", f"GPU {idx}")
                vram_free = dev.get("vram_free", 0)
                vram_total = dev.get("vram_total", 0)

                if vram_total > 0:
                    vram_used = vram_total - vram_free
                    used_gb = vram_used / (1024 ** 3)
                    total_gb = vram_total / (1024 ** 3)
                    percent = (vram_used / vram_total) * 100
                    embed.add_field(
                        name=f"🎮 {dev_name}",
                        value=f"VRAM: `{used_gb:.2f} GB` / `{total_gb:.2f} GB` ({percent:.1f}% used)",
                        inline=False
                    )

        is_caution, vram_info = await asyncio.to_thread(check_gpu_vram_caution)
        if is_caution:
            pct = vram_info.get("percent_used", 0)
            free = vram_info.get("free_gb", 0)
            embed.add_field(
                name="⚠️ CAUTION: Heavy Background VRAM Load",
                value=f"GPU VRAM is at `{pct:.1f}%` capacity (`{free:.2f} GB` free). Tdarr or another process is heavily utilizing GPU memory.",
                inline=False
            )

        await interaction.followup.send(embed=embed)
