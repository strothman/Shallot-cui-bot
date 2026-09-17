"""
Krea Cog for Shallot-CUI Bot.
Handles /bertflow and /blend-krea commands and autocompletes.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

from model_architecture import Architecture
from characters import get_character_autocomplete_choices
from celebrities import get_celebrity_autocomplete_choices
import db
from core_helpers import safe_defer, edit_original_fallback
from services.krea_service import (
    BERTFLOW_MODEL_CHOICES,
    execute_bertflow,
    execute_blend_krea_core,
)

logger = logging.getLogger("DiscordBot.KreaCog")


class KreaCog(commands.Cog):
    """Cog handling Bert's Krea 2 Photorealism workflow (/bertflow) and Krea 2 Blend Studio (/blend-krea)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="bertflow",
        description="📸 Generate ultra-photorealistic images using Bert's Krea 2 workflow!"
    )
    @app_commands.describe(
        prompt="Scene/subject description (supports natural language, --ar, --ogarla, --valerie)",
        aspect_ratio="Image aspect ratio (1:1, 16:9, 9:16, 21:9, 3:4, etc.)",
        character="Optional character LoRA preset (Ogarla / Valerie Krea 2)",
        celebrity="Optional favorite celebrity to inject into prompt (Audrey Hepburn, Zendaya, etc.)",
        favorite_prompt="Apply one of your saved favorite prompts",
        model="Select Krea 2 UNET Checkpoint (Auto-detects available model)",
        steps="Sampling steps (8 for Turbo, up to 20 for Extended)",
        seed="Optional fixed seed for reproducibility"
    )
    @app_commands.choices(
        aspect_ratio=[
            app_commands.Choice(name="1:1 (Square - 1224x1224 Native)", value="1:1"),
            app_commands.Choice(name="16:9 (Landscape - 1632x920)", value="16:9"),
            app_commands.Choice(name="9:16 (Portrait - 920x1632)", value="9:16"),
            app_commands.Choice(name="21:9 (Cinematic Ultrawide - 1872x800)", value="21:9"),
            app_commands.Choice(name="3:4 (Classic Portrait - 1056x1408)", value="3:4"),
            app_commands.Choice(name="4:3 (Classic Landscape - 1408x1056)", value="4:3"),
            app_commands.Choice(name="16:9.3 (Taskbar Fit - 1632x880)", value="16:9.3"),
        ],
        model=BERTFLOW_MODEL_CHOICES
    )
    async def bertflow(
        self,
        interaction: discord.Interaction,
        prompt: str,
        aspect_ratio: str = "1:1",
        character: str = None,
        celebrity: str = None,
        favorite_prompt: str = None,
        model: str = None,
        steps: int = 8,
        seed: int = None
    ):
        if favorite_prompt:
            clean_fav = favorite_prompt.replace("📌", "").strip()
            fav_text = None
            user_prompts = db.get_favorite_prompts(interaction.user.id)
            for item in user_prompts:
                p_id = str(item['id'])
                p_name = item['prompt_name'].strip()
                p_full = item['prompt_text'].strip()
                if clean_fav == p_id or clean_fav.startswith(p_id) or clean_fav == p_name or clean_fav in p_name:
                    fav_text = p_full
                    break
            if fav_text:
                if prompt and prompt.strip():
                    prompt = f"{prompt}, {fav_text}"
                else:
                    prompt = fav_text

        if not prompt or not prompt.strip():
            await interaction.response.send_message("Please provide a prompt or select a saved favorite prompt.", ephemeral=True)
            return

        await safe_defer(interaction, thinking=True)
        char_val = character.value if hasattr(character, "value") else character
        celeb_val = celebrity.value if hasattr(celebrity, "value") else celebrity
        client = getattr(self.bot, "comfy_client", None)
        await execute_bertflow(
            interaction=interaction,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=steps,
            model_name=model,
            character=char_val,
            celebrity=celeb_val,
            client=client
        )

    @bertflow.autocomplete('character')
    async def bertflow_character_autocomplete(self, interaction: discord.Interaction, current: str):
        return get_character_autocomplete_choices(current, Architecture.KREA2)

    @bertflow.autocomplete('celebrity')
    async def bertflow_celebrity_autocomplete(self, interaction: discord.Interaction, current: str):
        return get_celebrity_autocomplete_choices(current)

    @bertflow.autocomplete('favorite_prompt')
    async def bertflow_favorite_prompt_autocomplete(self, interaction: discord.Interaction, current: str):
        prompts = db.get_favorite_prompts(interaction.user.id)
        choices = []
        for item in prompts:
            label = f"📌 {item['prompt_name']}".strip()
            if not current or current.lower() in label.lower() or current.lower() in item['prompt_text'].lower():
                choices.append(app_commands.Choice(name=label[:100], value=str(item['id'])))
        return choices[:25]

    @app_commands.command(
        name="blend-krea",
        description="📸 Blend and remix an image using AI vision analysis and Bert's Krea 2 photorealism workflow!"
    )
    @app_commands.describe(
        image="The image file you want to analyze and blend with Krea 2",
        steps="Sampling steps (8 for Turbo, 10-16 for macro/close-up details)",
        prompt="Optional extra instructions or details to blend into vision prompt"
    )
    async def blend_krea(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
        steps: int = 8,
        prompt: str = None
    ):
        await safe_defer(interaction, thinking=False, ephemeral=False)
        await edit_original_fallback(interaction, content="Analyzing image with Qwen2.5-VL for Krea 2 photorealism blend...")

        if not image.content_type or not image.content_type.startswith("image/"):
            await edit_original_fallback(interaction, content="❌ Please upload a valid image file (PNG/JPG).")
            return

        try:
            image_bytes = await image.read()
            client = getattr(self.bot, "comfy_client", None)
            await execute_blend_krea_core(
                interaction=interaction,
                image_bytes=image_bytes,
                filename=image.filename,
                image_url=image.url,
                prompt=prompt,
                steps=steps,
                client=client
            )
        except Exception as e:
            logger.error(f"Error reading image for blend-krea: {e}")
            await edit_original_fallback(interaction, content=f"❌ Failed to read uploaded image: {e}")


async def setup(bot: commands.Bot):
    """Asynchronous setup hook to register KreaCog into the bot."""
    await bot.add_cog(KreaCog(bot))
