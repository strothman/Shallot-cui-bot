"""
Core Generation Lifecycle Views, 2x2 Grid Actions, Upscale, and Isolation Controls for Shallot-CUI Bot.
"""

import logging
import discord

logger = logging.getLogger("DiscordBot.Views.Grid")


class CancelGenerationView(discord.ui.View):
    """Temporary view attached to generation status messages allowing users to cancel execution."""
    def __init__(self, generation_id: str):
        super().__init__(timeout=600)
        self.generation_id = generation_id

        self.add_item(discord.ui.Button(
            label="🛑 Cancel",
            style=discord.ButtonStyle.danger,
            custom_id=f"cancel_gen:{self.generation_id}"
        ))


class IsolatedImageButtons(discord.ui.View):
    def __init__(self, generation_id: str, index: int, has_sref: bool = False, is_blend: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.index = index
        self.is_blend = is_blend

        # Row 0: Upscale & Variation Options (4 buttons max)
        self.add_item(discord.ui.Button(
            label="🔍 High-Res (2x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:2.0",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="💎 Ultra 4K (4x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:4.0",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Subtle)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_subtle:{self.generation_id}:{self.index}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Strong)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_strong:{self.generation_id}:{self.index}",
            row=0
        ))

        # Row 1: Directional Pan & Zoom Controls (5 buttons max)
        self.add_item(discord.ui.Button(
            label="⬅️ Pan Left",
            style=discord.ButtonStyle.secondary,
            custom_id=f"outpaint:{self.generation_id}:{self.index}:left",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="⬆️ Pan Up",
            style=discord.ButtonStyle.secondary,
            custom_id=f"outpaint:{self.generation_id}:{self.index}:up",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="⬇️ Pan Down",
            style=discord.ButtonStyle.secondary,
            custom_id=f"outpaint:{self.generation_id}:{self.index}:down",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="➡️ Pan Right",
            style=discord.ButtonStyle.secondary,
            custom_id=f"outpaint:{self.generation_id}:{self.index}:right",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="🔍 Zoom 1.5x",
            style=discord.ButtonStyle.secondary,
            custom_id=f"outpaint:{self.generation_id}:{self.index}:1.5x",
            row=1
        ))

        # Row 2: Actions, Favorites, Remix & Studio Loop (up to 5 buttons max)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=2
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.primary,
            custom_id=f"remix:{self.generation_id}",
            row=2
        ))
        if is_blend:
            self.add_item(discord.ui.Button(
                label="🎛️ Adjust Blend",
                style=discord.ButtonStyle.secondary,
                custom_id=f"reblend:{self.generation_id}",
                row=2
            ))

        # Row 3: Change Style Reference (--sref) (only if has_sref=True, 3 buttons max)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="🎨 Custom --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_custom:{self.generation_id}:{self.index}",
                row=3
            ))
            self.add_item(discord.ui.Button(
                label="🎲 Random --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_random:{self.generation_id}:{self.index}",
                row=3
            ))
            self.add_item(discord.ui.Button(
                label="⭐ Saved --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_saved:{self.generation_id}:{self.index}",
                row=3
            ))


class UpscaleButtons(discord.ui.View):
    def __init__(self, generation_id: str, index: int, upscale_scale: str = "2.0", has_sref: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.index = index
        self.upscale_scale = upscale_scale

        # Row 0: Re-roll Grid & Redo Upscale
        self.add_item(discord.ui.Button(
            label="🔄 Start Over (New Grid)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"reroll:{self.generation_id}",
            row=0
        ))

        self.add_item(discord.ui.Button(
            label="⚡ Vary Upscale Details",
            style=discord.ButtonStyle.secondary,
            custom_id=f"upscale_redo:{self.generation_id}:{self.index}:{self.upscale_scale}",
            row=0
        ))

        # Row 1: Variations (Subtle vs Strong)
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Subtle)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_subtle:{self.generation_id}:{self.index}",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Strong)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_strong:{self.generation_id}:{self.index}",
            row=1
        ))

        # Row 2: Favorite Buttons (Style & Prompt)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=2
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=2
        ))

        # Row 3: Change Style Reference (--sref)
        self.add_item(discord.ui.Button(
            label="🎨 Custom --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_custom:{self.generation_id}:{self.index}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="🎲 Random --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_random:{self.generation_id}:{self.index}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="⭐ Saved --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_saved:{self.generation_id}:{self.index}",
            row=3
        ))


class GridButtons(discord.ui.View):
    def __init__(self, generation_id, has_sref=False, is_blend=False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.is_blend = is_blend

        # Row 0: Upscale buttons U1-U4
        for i in range(1, 5):
            self.add_item(discord.ui.Button(
                label=f"U{i}", 
                style=discord.ButtonStyle.secondary, 
                custom_id=f"upscale:{self.generation_id}:{i}",
                row=0
            ))

        # Row 1: Variation buttons V1-V4 + Re-roll
        for i in range(1, 5):
            self.add_item(discord.ui.Button(
                label=f"V{i}", 
                style=discord.ButtonStyle.secondary, 
                custom_id=f"variation:{self.generation_id}:{i}",
                row=1
            ))
        self.add_item(discord.ui.Button(
            label="🔄",
            style=discord.ButtonStyle.secondary,
            custom_id=f"reroll:{self.generation_id}",
            row=1
        ))

        # Row 2: Actions & Favorites (up to 5 buttons, adhering to Discord limit)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=2
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.primary,
            custom_id=f"remix:{self.generation_id}",
            row=2
        ))
        if is_blend:
            self.add_item(discord.ui.Button(
                label="🎛️ Adjust Blend",
                style=discord.ButtonStyle.secondary,
                custom_id=f"reblend:{self.generation_id}",
                row=2
            ))


class StasisControlsView(discord.ui.View):
    def __init__(self, generation_id: str, user_id: int):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.user_id = user_id

        self.add_item(discord.ui.Button(
            label="⏸️ Pause / Stasis",
            style=discord.ButtonStyle.secondary,
            custom_id=f"stasis_pause:{self.generation_id}:{self.user_id}"
        ))


class StasisPausedView(discord.ui.View):
    def __init__(self, generation_id: str, user_id: int):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.user_id = user_id

        self.add_item(discord.ui.Button(
            label="▶️ Resume",
            style=discord.ButtonStyle.primary,
            custom_id=f"stasis_resume:{self.generation_id}:{self.user_id}"
        ))


class BertflowButtons(discord.ui.View):
    """Buttons attached to /bertflow photorealism generations.
    Actions are handled persistently via custom_id in bot.py on_interaction to prevent duplicate triggers.
    """
    def __init__(
        self,
        generation_id: str,
        on_reroll_cb=None,
        on_remix_cb=None,
        character: str = None,
        on_toggle_char_cb=None,
        on_upscale_cb=None
    ):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.on_reroll_cb = on_reroll_cb
        self.on_remix_cb = on_remix_cb
        self.character = character
        self.on_toggle_char_cb = on_toggle_char_cb
        self.on_upscale_cb = on_upscale_cb

        self.reroll_btn = discord.ui.Button(
            label="🔄 Re-roll",
            style=discord.ButtonStyle.primary,
            custom_id=f"bertflow_reroll:{generation_id}"
        )
        self.add_item(self.reroll_btn)

        self.remix_btn = discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.secondary,
            custom_id=f"bertflow_remix:{generation_id}"
        )
        self.add_item(self.remix_btn)

        has_char = character and str(character).lower() not in ["none", "nochar", "off", "false"]
        if not has_char:
            char_label = "🌿 Ogarla: OFF"
        elif "ogarla" in str(character).lower() or "oga" in str(character).lower():
            char_label = "🌿 Ogarla: ON"
        else:
            char_label = f"🎭 {character}: ON"
        char_style = discord.ButtonStyle.success if has_char else discord.ButtonStyle.secondary
        self.toggle_char_btn = discord.ui.Button(
            label=char_label,
            style=char_style,
            custom_id=f"bertflow_toggle_char:{generation_id}"
        )
        self.add_item(self.toggle_char_btn)

        self.upscale_btn = discord.ui.Button(
            label="🔍 Upscale (1.5x)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"bertflow_upscale:{generation_id}"
        )
        self.add_item(self.upscale_btn)
