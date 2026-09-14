"""
Upscale Cog for Shallot-CUI Bot.
Handles /upscale slash command with multi-tier AI super-resolution.
"""

import io
import logging
import discord
from discord import app_commands
from discord.ext import commands

from services.upscale_service import (
    execute_upscale_core,
    DEFAULT_SCALE_FACTOR,
)
from core_helpers import safe_defer, edit_original_fallback, send_error_fallback

logger = logging.getLogger("DiscordBot.UpscaleCog")


class UpscaleCog(commands.Cog):
    """Cog handling modern multi-tier AI upscaling (/upscale)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="upscale",
        description="🔍 Upscale an image to 2K/4K with Fast Clean Super-Resolution or Generative Clarity."
    )
    @app_commands.describe(
        image="The image file you want to upscale (PNG/JPG/WEBP)",
        scale="Target scaling multiplier (2x Default, 4x Ultra 4K, 1.5x Subtle)",
        mode="Upscale engine mode (⚡ Fast Clean vs 💎 Generative Clarity)",
        style="Art style optimization (General/Photo vs Anime/Illustration)",
        prompt="Optional prompt guidance (used by Generative Clarity mode)"
    )
    @app_commands.choices(
        scale=[
            app_commands.Choice(name="2x (High Res - 2K Default)", value="2.0"),
            app_commands.Choice(name="4x (Ultra HD - 4K)", value="4.0"),
            app_commands.Choice(name="1.5x (Subtle Boost)", value="1.5"),
        ],
        mode=[
            app_commands.Choice(name="⚡ Fast Clean (Instant Model Super-Resolution)", value="fast"),
            app_commands.Choice(name="💎 Generative Clarity (Diffusion AI Micro-Detail Refiner)", value="generative"),
        ],
        style=[
            app_commands.Choice(name="🎨 General / Photorealistic (Remacri)", value="general"),
            app_commands.Choice(name="🌸 Anime / Illustration (AnimeSharp)", value="anime"),
        ]
    )
    async def upscale(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        scale: str = "2.0",
        mode: str = "fast",
        style: str = "general",
        prompt: str = ""
    ):
        """Modernized upscale command offering high-res scaling factors and engine modes."""
        await safe_defer(interaction, thinking=True)

        if not image.content_type or not image.content_type.startswith("image/"):
            await interaction.followup.send("❌ Please upload a valid image file (PNG/JPG/WEBP).", ephemeral=True)
            return

        try:
            scale_val = float(scale)
        except (ValueError, TypeError):
            scale_val = DEFAULT_SCALE_FACTOR

        try:
            image_bytes = await image.read()
            client = getattr(self.bot, "comfy_client", None)

            result = await execute_upscale_core(
                image_bytes=image_bytes,
                filename=image.filename,
                scale_factor=scale_val,
                mode=mode,
                style=style,
                prompt=prompt,
                client=client
            )

            orig_w, orig_h = result["orig_width"], result["orig_height"]
            target_w, target_h = result["target_width"], result["target_height"]
            out_bytes = result["image_bytes"]
            chosen_model = result["model"]

            engine_label = "💎 Generative Clarity (Diffusion Refiner)" if mode == "generative" else "⚡ Fast Clean (Model Super-Res)"
            embed = discord.Embed(
                title="🔍 Image Upscale Complete",
                color=discord.Color.from_rgb(52, 152, 219)
            )
            embed.add_field(name="📐 Resolution", value=f"`{orig_w}x{orig_h}` ➔ `{target_w}x{target_h}` (`{scale_val}x`)", inline=True)
            embed.add_field(name="⚙️ Engine", value=engine_label, inline=True)
            embed.add_field(name="🧠 Model", value=f"`{chosen_model}`", inline=True)

            if prompt and mode == "generative":
                embed.add_field(name="📝 Prompt Guidance", value=f"`{prompt[:250]}`", inline=False)

            embed.set_footer(text=f"Requested by {interaction.user.display_name} • Shallot-CUI Bot")
            
            clean_filename = f"upscaled_{scale_val}x_{image.filename}"
            embed.set_image(url=f"attachment://{clean_filename}")

            file = discord.File(fp=io.BytesIO(out_bytes), filename=clean_filename)
            await interaction.followup.send(embed=embed, file=file)

        except Exception as e:
            logger.error(f"Error executing upscale command: {e}", exc_info=True)
            await send_error_fallback(interaction, f"Failed to upscale image: {e}")


async def setup(bot: commands.Bot):
    """Asynchronous setup hook to register UpscaleCog into the bot."""
    await bot.add_cog(UpscaleCog(bot))
