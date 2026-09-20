"""
Poetic Gamble Views and Embed Builders for Shallot-CUI Bot.
Provides interactive Discord UI for the elevated Poetic Gamble Studio,
including engine switching (SDXL <-> Krea 2), mood shifting, and lyrical card embeds.
"""

import discord
from typing import Optional, Dict, Any

from parsers.poetic import MOOD_DISPLAY_NAMES, PoeticMood


class GambleButtons(discord.ui.View):
    """
    Interactive button view attached to Poetic Gamble generations.
    Supports quadrant isolation (for SDXL 4-grids), re-gambling with the same seed,
    shifting moods, and instantly switching between SDXL and Krea 2 engines.
    """
    def __init__(self, generation_id: str, engine: str = "sdxl", mood: str = "wild"):
        super().__init__(timeout=None) # Persistent UI
        self.generation_id = generation_id
        self.engine = "krea2" if engine.lower() in ["krea", "krea2", "bertflow"] else "sdxl"
        self.mood = mood.lower()

        is_sdxl = (self.engine == "sdxl")

        # Row 0: Quadrant Isolation U1-U4 (SDXL 4-image grid) or Quick Actions
        if is_sdxl:
            for i in range(1, 5):
                self.add_item(discord.ui.Button(
                    label=f"U{i}",
                    style=discord.ButtonStyle.primary,
                    custom_id=f"upscale:{self.generation_id}:{i}",
                    row=0
                ))
            # Row 0 button 5: Quick upscale U1
            self.add_item(discord.ui.Button(
                label="🔍 High-Res (2x)",
                style=discord.ButtonStyle.success,
                custom_id=f"upscale_run:{self.generation_id}:1:2.0",
                row=0
            ))

        # Row 1: Studio Gamble Controls (5 buttons max)
        self.add_item(discord.ui.Button(
            label="🎲 Re-Gamble",
            style=discord.ButtonStyle.success,
            custom_id=f"gamble_reroll:{self.generation_id}",
            row=1
        ))

        self.add_item(discord.ui.Button(
            label="🎭 Shift Mood",
            style=discord.ButtonStyle.secondary,
            custom_id=f"gamble_mood:{self.generation_id}",
            row=1
        ))

        # Engine switch toggle button
        if is_sdxl:
            self.add_item(discord.ui.Button(
                label="⚡ Switch to Krea 2",
                style=discord.ButtonStyle.primary,
                custom_id=f"gamble_switch:{self.generation_id}:krea2",
                row=1
            ))
        else:
            self.add_item(discord.ui.Button(
                label="🎨 Switch to SDXL",
                style=discord.ButtonStyle.primary,
                custom_id=f"gamble_switch:{self.generation_id}:sdxl",
                row=1
            ))

        self.add_item(discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.secondary,
            custom_id=f"remix:{self.generation_id}",
            row=1
        ))


def build_gamble_embed(
    gamble_info: Dict[str, Any],
    seed: int,
    width: int,
    height: int,
    model_name: str,
    user_name: str = "User",
    user_id: int = 0,
    timing_data: Optional[Dict[str, Any]] = None,
    engine: str = "sdxl"
) -> discord.Embed:
    """
    Builds the poetic presentation card embed featuring the lyrical stanza,
    mood and engine badges, and aesthetic palette specs.
    """
    seed_word = gamble_info.get("seed_word", "Mystery")
    mood = gamble_info.get("mood", "wild")
    stanza = gamble_info.get("stanza", "")
    palette = gamble_info.get("color_palette", "")
    
    mood_badge = MOOD_DISPLAY_NAMES.get(mood, "🎲 Mystery")
    is_krea = (engine.lower() in ["krea", "krea2", "bertflow"])
    engine_badge = "⚡ Krea 2 Turbo Photorealism" if is_krea else "🎨 SDXL Fine Art"
    color = discord.Color.from_rgb(235, 140, 52) if is_krea else discord.Color.purple()

    embed = discord.Embed(
        title=f"🎲 Poetic Gamble • \"{seed_word.title()}\"",
        color=color
    )

    # Main lyrical card in blockquote styling
    if stanza:
        formatted_stanza = "\n".join(f"> *{line.strip()}*" for line in stanza.split("\n") if line.strip())
        embed.description = f"{formatted_stanza}\n\n"
    else:
        embed.description = ""

    embed.add_field(name="🎭 Mood", value=f"`{mood_badge}`", inline=True)
    embed.add_field(name="⚙️ Engine", value=f"`{engine_badge}`", inline=True)
    if palette:
        embed.add_field(name="🎨 Palette", value=f"`{palette}`", inline=True)

    embed.add_field(name="📐 Specs", value=f"`{width}x{height}` • Seed: `{seed}`", inline=True)
    embed.add_field(name="🤖 Model", value=f"`{model_name.split('.')[0]}`", inline=True)

    timing_text = ""
    if timing_data:
        elapsed = timing_data.get("elapsed_time", 0.0)
        if elapsed > 0:
            timing_text = f" • Rendered in {elapsed:.1f}s"

    embed.set_footer(text=f"Requested by {user_name} (ID: {user_id}){timing_text}")
    return embed
