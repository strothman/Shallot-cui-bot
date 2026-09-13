"""
Video Cog for Shallot-CUI Bot.
Handles /video (Wan 2.2), /ltx (LTX-Video), and 'Animate to Video' context menu commands.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

from core_helpers import safe_defer, send_error_fallback
from services.video_service import (
    execute_video_core,
    execute_ltx_core,
    execute_animate_message,
)

logger = logging.getLogger("DiscordBot.VideoCog")


class VideoCog(commands.Cog):
    """Cog handling Wan 2.2 Image-to-Video (/video) and LTX-Video (/ltx)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.ctx_animate = app_commands.ContextMenu(
            name="Animate to Video",
            callback=self.animate_to_video_context,
        )
        self.bot.tree.add_command(self.ctx_animate)

    async def cog_unload(self):
        self.bot.tree.remove_command(self.ctx_animate.name, type=self.ctx_animate.type)

    async def animate_to_video_context(self, interaction: discord.Interaction, message: discord.Message):
        """Context menu command to animate any right-clicked image message into a Wan 2.2 video."""
        client = getattr(self.bot, "comfy_client", None)
        await execute_animate_message(interaction, message, client=client)

    @app_commands.command(name="video", description="Generate a 10s or 5s video from an image using Wan 2.2.")
    @app_commands.describe(
        image="The source image file you want to animate",
        prompt="Text prompt describing the desired video motion or action",
        duration="Video duration in seconds (10s default, or 5s for quick tests)",
        smoothness="Motion smoothing speed mode (Smooth 32 FPS vs Fast 16 FPS)",
        seed="Optional seed for generation reproducibility"
    )
    @app_commands.choices(
        duration=[
            app_commands.Choice(name="10 seconds (161 frames - Full Natural Motion)", value=10),
            app_commands.Choice(name="5 seconds (81 frames - Quick Test)", value=5),
        ],
        smoothness=[
            app_commands.Choice(name="🎬 Smooth (32 FPS - Accelerated RIFE)", value="smooth"),
            app_commands.Choice(name="⚡ Ultra Fast (16 FPS - Native / No RIFE)", value="fast"),
        ]
    )
    async def video(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        prompt: str,
        duration: int = 10,
        smoothness: str = "smooth",
        seed: int = None
    ):
        """Generate high-fidelity animated video from an uploaded image using Wan 2.2."""
        await safe_defer(interaction, thinking=True)

        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.followup.send("Please upload a valid image file (PNG/JPG/WEBP).")
            return

        try:
            image_bytes = await image.read()
            client = getattr(self.bot, "comfy_client", None)
            await execute_video_core(
                interaction=interaction,
                image_bytes=image_bytes,
                filename=image.filename,
                prompt=prompt,
                duration=duration,
                smoothness=smoothness,
                seed=seed,
                client=client
            )
        except Exception as e:
            logger.error(f"Error executing video command: {e}")
            await send_error_fallback(interaction, f"An error occurred during video generation: {e}")

    @app_commands.command(name="ltx", description="Generate high-speed video animation using LTX-Video (optimized for 8GB VRAM).")
    @app_commands.describe(
        image="The source image file you want to animate",
        prompt="Text prompt describing the desired video motion or action",
        duration="Video duration in seconds (4s, 6s, 8s, or 10s, default 4s)",
        motion_strength="Motion intensity (1 to 10, default 7)",
        seed="Optional seed for generation reproducibility"
    )
    @app_commands.choices(
        duration=[
            app_commands.Choice(name="4 seconds (97 frames - Fast ~35s)", value=4),
            app_commands.Choice(name="6 seconds (161 frames - ~55s)", value=6),
            app_commands.Choice(name="8 seconds (209 frames - ~75s)", value=8),
            app_commands.Choice(name="10 seconds (257 frames - ~95s)", value=10),
        ]
    )
    async def ltx(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        prompt: str,
        duration: int = 4,
        motion_strength: int = 7,
        seed: int = None
    ):
        """Generate rapid video animation using LTX-Video."""
        await safe_defer(interaction, thinking=True)

        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.followup.send("Please upload a valid image file (PNG/JPG/WEBP).")
            return

        try:
            image_bytes = await image.read()
            client = getattr(self.bot, "comfy_client", None)
            await execute_ltx_core(
                interaction=interaction,
                image_bytes=image_bytes,
                filename=image.filename,
                prompt=prompt,
                duration=duration,
                motion_strength=motion_strength,
                seed=seed,
                client=client
            )
        except Exception as e:
            logger.error(f"Error executing LTX video command: {e}")
            await send_error_fallback(interaction, f"An error occurred during LTX video generation: {e}")


async def setup(bot: commands.Bot):
    """Asynchronous setup hook to register VideoCog into the bot."""
    await bot.add_cog(VideoCog(bot))
