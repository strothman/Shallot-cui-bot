"""
Poetic Gamble Service for Shallot-CUI Bot.
Coordinates the execution of elevated poetic visual allegories across SDXL Fine Art
and Krea 2 Photorealism (Bertflow), managing rerolls, engine switches, and mood shifts.
"""

import logging
import discord
from typing import Optional, Dict, Any

from parsers.poetic import (
    PoeticMood,
    synthesize_poetic_prompt,
    PoeticPromptResult,
)
import db
from core_helpers import safe_defer, send_error_fallback

logger = logging.getLogger("DiscordBot.GambleService")


async def execute_gamble(
    interaction: discord.Interaction,
    seed: Optional[str] = None,
    mood: str = "wild",
    engine: str = "sdxl",
    aspect_ratio: str = "1:1"
):
    """
    Synthesizes a rich poetic allegory and dispatches generation to either
    SDXL Fine Art (execute_imagine) or Krea 2 Photorealism (execute_bertflow).
    """
    is_krea = (engine.lower() in ["krea", "krea2", "bertflow"])
    target_engine = "krea2" if is_krea else "sdxl"

    poetic_res = synthesize_poetic_prompt(
        seed=seed,
        mood=mood,
        engine=target_engine
    )

    gamble_info = {
        "seed_word": poetic_res.seed_word,
        "mood": poetic_res.mood,
        "stanza": poetic_res.stanza,
        "engine": target_engine,
        "metaphor": poetic_res.metaphor,
        "setting": poetic_res.setting,
        "color_palette": poetic_res.color_palette,
        "aspect_ratio": aspect_ratio or "1:1",
    }

    logger.info(
        f"[/gamble] Seed: '{poetic_res.seed_word}' | Mood: {poetic_res.mood} | "
        f"Engine: {target_engine} | Prompt: {poetic_res.prompt[:100]}..."
    )

    if is_krea:
        from services.krea_service import execute_bertflow
        await execute_bertflow(
            interaction,
            prompt=poetic_res.prompt,
            aspect_ratio=aspect_ratio or "1:1",
            gamble_info=gamble_info
        )
    else:
        from services.generation_service import execute_imagine
        sdxl_prompt = f"{poetic_res.prompt} --ar {aspect_ratio or '1:1'}"
        await execute_imagine(
            interaction,
            prompt=sdxl_prompt,
            aspect_ratio=aspect_ratio or "1:1",
            gamble_info=gamble_info
        )


async def handle_gamble_reroll(interaction: discord.Interaction, generation_id: str):
    """
    Re-rolls a fresh poetic metaphor, setting, and color harmony
    using the exact same seed word and active mood.
    """
    await safe_defer(interaction)
    gen_data = db.get_generation(generation_id) or {}
    gamble_info = gen_data.get("gamble_info") or {}

    seed_word = gamble_info.get("seed_word")
    mood = gamble_info.get("mood", "wild")
    engine = gamble_info.get("engine", "sdxl")
    ar = gamble_info.get("aspect_ratio", gen_data.get("aspect_ratio", "1:1"))

    await execute_gamble(
        interaction,
        seed=seed_word,
        mood=mood,
        engine=engine,
        aspect_ratio=ar
    )


async def handle_gamble_switch_engine(
    interaction: discord.Interaction,
    generation_id: str,
    target_engine: str
):
    """
    Switches the poetic generation from SDXL to Krea 2 (or vice versa),
    preserving the seed concept and active mood.
    """
    await safe_defer(interaction)
    gen_data = db.get_generation(generation_id) or {}
    gamble_info = gen_data.get("gamble_info") or {}

    seed_word = gamble_info.get("seed_word")
    mood = gamble_info.get("mood", "wild")
    ar = gamble_info.get("aspect_ratio", gen_data.get("aspect_ratio", "1:1"))

    await execute_gamble(
        interaction,
        seed=seed_word,
        mood=mood,
        engine=target_engine,
        aspect_ratio=ar
    )


async def handle_gamble_shift_mood(interaction: discord.Interaction, generation_id: str):
    """
    Cycles to the next aesthetic mood archetype in sequence:
    Ethereal -> Gothic -> Noir -> Mythic -> Surreal -> Wild -> Ethereal.
    """
    await safe_defer(interaction)
    gen_data = db.get_generation(generation_id) or {}
    gamble_info = gen_data.get("gamble_info") or {}

    seed_word = gamble_info.get("seed_word")
    current_mood = gamble_info.get("mood", "wild").lower()
    engine = gamble_info.get("engine", "sdxl")
    ar = gamble_info.get("aspect_ratio", gen_data.get("aspect_ratio", "1:1"))

    mood_cycle = [
        PoeticMood.ETHEREAL,
        PoeticMood.GOTHIC,
        PoeticMood.NOIR,
        PoeticMood.MYTHIC,
        PoeticMood.SURREAL,
        PoeticMood.WILD,
    ]

    try:
        idx = mood_cycle.index(current_mood)
        next_mood = mood_cycle[(idx + 1) % len(mood_cycle)]
    except ValueError:
        next_mood = PoeticMood.ETHEREAL

    await execute_gamble(
        interaction,
        seed=seed_word,
        mood=next_mood,
        engine=engine,
        aspect_ratio=ar
    )
