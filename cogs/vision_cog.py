"""
Vision Cog for Shallot-CUI Bot.
Handles /blend-sdxl and 'Blend Image (SDXL)' context menu.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

from services.vision_service import (
    execute_blend_core, 
    execute_blend_message,
    execute_describe_core
)
from core_helpers import safe_defer, edit_original_fallback

logger = logging.getLogger("DiscordBot.VisionCog")


class VisionCog(commands.Cog):
    """Cog handling Florence-2 vision interrogation and SDXL image blending."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Register Context Menu for right-click interaction on image messages
        self.ctx_blend = app_commands.ContextMenu(
            name="Blend Image (SDXL)",
            callback=self.blend_image_context,
        )
        self.bot.tree.add_command(self.ctx_blend)

    async def cog_unload(self):
        self.bot.tree.remove_command(self.ctx_blend.name, type=self.ctx_blend.type)

    async def blend_image_context(self, interaction: discord.Interaction, message: discord.Message):
        """Context menu command to run /blend-sdxl on any right-clicked message containing an image."""
        client = getattr(self.bot, "comfy_client", None)
        await execute_blend_message(interaction, message, client=client)

    @app_commands.command(
        name="blend-sdxl", 
        description="Interactively blend or remix an image using dedicated SDXL vision tags, LoRAs, and checkpoints."
    )
    @app_commands.describe(
        image="Upload an image to blend and remix with SDXL"
    )
    async def blend_sdxl(
        self,
        interaction: discord.Interaction, 
        image: discord.Attachment
    ):
        """Slash command to blend an uploaded image with SDXL checkpoints and LoRAs."""
        await safe_defer(interaction, thinking=False, ephemeral=False)
        await edit_original_fallback(interaction, content="Analyzing image for SDXL blending...")
        
        if not image.content_type or not image.content_type.startswith("image/"):
            await edit_original_fallback(interaction, content="❌ Please upload a valid image file (PNG/JPG).")
            return

        try:
            image_bytes = await image.read()
            client = getattr(self.bot, "comfy_client", None)
            await execute_blend_core(
                interaction=interaction,
                image_bytes=image_bytes,
                filename=image.filename,
                image_url=image.url,
                client=client
            )
        except Exception as e:
            logger.error(f"Error reading image for blend: {e}")
            await edit_original_fallback(interaction, content=f"❌ Failed to read uploaded image: {e}")

    @app_commands.command(
        name="describe", 
        description="Generate multi-architecture prompts for an image using Florence-2 or Qwen2.5-VL."
    )
    @app_commands.describe(
        image="The image file you want to describe",
        model="Preferred vision model (Florence-2 fast default or Qwen2.5-VL detailed scene)"
    )
    @app_commands.choices(
        model=[
            app_commands.Choice(name="Florence-2 (Fast • ~2s • Recommended Default)", value="florence2"),
            app_commands.Choice(name="Qwen2.5-VL (Detailed Scene • ~12s)", value="qwen2.5-vl"),
        ]
    )
    async def describe(self, interaction: discord.Interaction, image: discord.Attachment, model: str = None):
        """Slash command to interrogate an image and synthesize prompts for SDXL, Flux, and Krea 2."""
        client = getattr(self.bot, "comfy_client", None)
        await execute_describe_core(interaction, image, model=model, client=client)


async def setup(bot: commands.Bot):
    """Asynchronous setup hook to register VisionCog into the bot."""
    await bot.add_cog(VisionCog(bot))

