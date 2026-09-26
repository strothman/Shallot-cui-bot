"""
Anima Cog for Shallot-CUI Bot.
Handles /anima slash command and favorite prompt expansion for the Anima 2B DiT workflow.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

import db
from core_helpers import safe_defer
from services.anima_service import execute_anima

logger = logging.getLogger("DiscordBot.AnimaCog")


class AnimaCog(commands.Cog):
    """Cog handling Anima 2-Stage Anime Generation workflow (/anima)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="anima",
        description="🌸 Generate high-resolution anime art with the Anima 2B DiT 2-Stage pipeline!"
    )
    @app_commands.describe(
        prompt="Scene/subject description (supports natural language and dynamic wildcards)",
        aspect_ratio="Image aspect ratio (4:3, 3:4, 1:1, 16:9, 9:16, 21:9)",
        favorite_prompt="Apply one of your saved favorite prompts",
        steps="Sampling steps for Stage 1 base generation (default: 35)",
        cfg="Guidance scale / CFG (default: 5.0)",
        negative_prompt="Optional negative prompt filtering",
        seed="Optional fixed seed for reproducibility"
    )
    @app_commands.choices(
        aspect_ratio=[
            app_commands.Choice(name="4:3 (Classic Landscape - 1376x1072 ➔ 1928x1504)", value="4:3"),
            app_commands.Choice(name="3:4 (Classic Portrait - 1072x1376 ➔ 1504x1928)", value="3:4"),
            app_commands.Choice(name="1:1 (Square - 1216x1216 ➔ 1704x1704)", value="1:1"),
            app_commands.Choice(name="16:9 (Cinematic Landscape - 1600x896 ➔ 2240x1254)", value="16:9"),
            app_commands.Choice(name="9:16 (Story Portrait - 896x1600 ➔ 1254x2240)", value="9:16"),
            app_commands.Choice(name="21:9 (Ultrawide - 1824x768 ➔ 2552x1076)", value="21:9"),
        ]
    )
    async def anima(
        self,
        interaction: discord.Interaction,
        prompt: str,
        aspect_ratio: str = "4:3",
        favorite_prompt: str = None,
        steps: int = 35,
        cfg: float = 5.0,
        negative_prompt: str = None,
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

        # Bound check steps and CFG
        steps = max(10, min(steps, 60))
        cfg = max(1.0, min(cfg, 15.0))

        await safe_defer(interaction, thinking=True)
        client = getattr(self.bot, "comfy_client", None)
        await execute_anima(
            interaction=interaction,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=steps,
            cfg=cfg,
            negative_prompt=negative_prompt,
            client=client
        )

    @anima.autocomplete("favorite_prompt")
    async def favorite_prompt_autocomplete(self, interaction: discord.Interaction, current: str):
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        choices = []
        for item in user_prompts:
            name = item["prompt_name"]
            text = item["prompt_text"]
            label = f"📌 {name}: {text}"
            if len(label) > 100:
                label = label[:97] + "..."
            if current.lower() in name.lower() or current.lower() in text.lower():
                choices.append(app_commands.Choice(name=label, value=str(item["id"])))
        return choices[:25]


async def setup(bot: commands.Bot):
    await bot.add_cog(AnimaCog(bot))
