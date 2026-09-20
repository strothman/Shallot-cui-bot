"""
Poetic Gamble Cog for Shallot-CUI Bot.
Provides the /gamble slash command to spin evocative visual allegories
across SDXL Fine Art and Krea 2 Photorealism.
"""

import logging
import discord
from discord import app_commands
from discord.ext import commands

from core_helpers import safe_defer
from services.gamble_service import execute_gamble

logger = logging.getLogger("DiscordBot.GambleCog")


class GambleCog(commands.Cog):
    """Cog handling the /gamble command and poetic allegories."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(
        name="gamble",
        description="🎲 Spin an evocative visual allegory in SDXL or Krea 2 from a single word or feeling!"
    )
    @app_commands.describe(
        seed="Seed word, feeling, or concept (e.g. solitude, ember, whisper — or blank for a mystery spin)",
        mood="Aesthetic mood archetype",
        engine="Generation engine (SDXL Fine Art or Krea 2 Photorealism)",
        aspect_ratio="Image aspect ratio (1:1, 16:9, 9:16, 21:9, 3:4, etc.)"
    )
    @app_commands.choices(
        mood=[
            app_commands.Choice(name="🎲 Wild Mystery (Random)", value="wild"),
            app_commands.Choice(name="🌌 Ethereal & Dreamlike", value="ethereal"),
            app_commands.Choice(name="🕯️ Gothic Melancholy", value="gothic"),
            app_commands.Choice(name="⚡ Neon Noir & Solitude", value="noir"),
            app_commands.Choice(name="🌿 Mythic & Ancient", value="mythic"),
            app_commands.Choice(name="🌀 Surreal Impossible", value="surreal"),
        ],
        engine=[
            app_commands.Choice(name="🎨 SDXL (Fine Art Illustration)", value="sdxl"),
            app_commands.Choice(name="⚡ Krea 2 (Bertflow Photorealism)", value="krea2"),
        ],
        aspect_ratio=[
            app_commands.Choice(name="1:1 (Square - 1024x1024 / 1224x1224)", value="1:1"),
            app_commands.Choice(name="16:9 (Landscape - 1344x768 / 1632x920)", value="16:9"),
            app_commands.Choice(name="9:16 (Portrait - 768x1344 / 920x1632)", value="9:16"),
            app_commands.Choice(name="21:9 (Cinematic Ultrawide)", value="21:9"),
            app_commands.Choice(name="3:4 (Classic Portrait)", value="3:4"),
            app_commands.Choice(name="4:3 (Classic Landscape)", value="4:3"),
        ]
    )
    async def gamble(
        self,
        interaction: discord.Interaction,
        seed: str = None,
        mood: str = "wild",
        engine: str = "sdxl",
        aspect_ratio: str = "1:1"
    ):
        await safe_defer(interaction)
        await execute_gamble(
            interaction,
            seed=seed,
            mood=mood,
            engine=engine,
            aspect_ratio=aspect_ratio
        )
