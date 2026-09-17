"""
Native discord.ui.DynamicItem Persistent Components for Shallot-CUI Bot.

Provides typed, regex-matched persistent buttons that survive bot restarts
without manual string parsing.
"""

import re
import logging
import discord

logger = logging.getLogger("DiscordBot.Views.DynamicItems")


class CancelGenDynamicButton(
    discord.ui.DynamicItem[discord.ui.Button], 
    template=r"cancel_gen:(?P<gen_id>[a-zA-Z0-9_\-]+)"
):
    """Persistent 🛑 Cancel button for active generations."""
    def __init__(self, generation_id: str):
        super().__init__(
            discord.ui.Button(
                label="🛑 Cancel",
                style=discord.ButtonStyle.danger,
                custom_id=f"cancel_gen:{generation_id}"
            )
        )
        self.generation_id = generation_id

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        gen_id = match.group("gen_id")
        return cls(gen_id)

    async def callback(self, interaction: discord.Interaction):
        from services.grid_actions_service import handle_cancel_generation
        await handle_cancel_generation(interaction, self.generation_id)


class IsolateDynamicButton(
    discord.ui.DynamicItem[discord.ui.Button], 
    template=r"upscale:(?P<gen_id>[a-zA-Z0-9_\-]+):(?P<index>[1-4])"
):
    """Persistent U1–U4 Quadrant Isolation button."""
    def __init__(self, generation_id: str, index: int):
        super().__init__(
            discord.ui.Button(
                label=f"U{index}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"upscale:{generation_id}:{index}"
            )
        )
        self.generation_id = generation_id
        self.index = int(index)

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        gen_id = match.group("gen_id")
        index = int(match.group("index"))
        return cls(gen_id, index)

    async def callback(self, interaction: discord.Interaction):
        from services.grid_actions_service import handle_isolate
        await handle_isolate(interaction, self.generation_id, self.index)


class VariationDynamicButton(
    discord.ui.DynamicItem[discord.ui.Button], 
    template=r"variation:(?P<gen_id>[a-zA-Z0-9_\-]+):(?P<index>[1-4])"
):
    """Persistent V1–V4 Quadrant Variation button."""
    def __init__(self, generation_id: str, index: int):
        super().__init__(
            discord.ui.Button(
                label=f"V{index}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"variation:{generation_id}:{index}"
            )
        )
        self.generation_id = generation_id
        self.index = int(index)

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        gen_id = match.group("gen_id")
        index = int(match.group("index"))
        return cls(gen_id, index)

    async def callback(self, interaction: discord.Interaction):
        from services.grid_actions_service import handle_variation
        await handle_variation(interaction, self.generation_id, self.index)


class RerollDynamicButton(
    discord.ui.DynamicItem[discord.ui.Button], 
    template=r"reroll:(?P<gen_id>[a-zA-Z0-9_\-]+)"
):
    """Persistent 🔄 Re-roll button."""
    def __init__(self, generation_id: str):
        super().__init__(
            discord.ui.Button(
                label="🔄",
                style=discord.ButtonStyle.secondary,
                custom_id=f"reroll:{generation_id}"
            )
        )
        self.generation_id = generation_id

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        gen_id = match.group("gen_id")
        return cls(gen_id)

    async def callback(self, interaction: discord.Interaction):
        from services.grid_actions_service import handle_reroll
        await handle_reroll(interaction, self.generation_id)


class RemixDynamicButton(
    discord.ui.DynamicItem[discord.ui.Button], 
    template=r"remix:(?P<gen_id>[a-zA-Z0-9_\-]+)"
):
    """Persistent ✏️ Remix button."""
    def __init__(self, generation_id: str):
        super().__init__(
            discord.ui.Button(
                label="✏️ Remix",
                style=discord.ButtonStyle.primary,
                custom_id=f"remix:{generation_id}"
            )
        )
        self.generation_id = generation_id

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        gen_id = match.group("gen_id")
        return cls(gen_id)

    async def callback(self, interaction: discord.Interaction):
        from services.grid_actions_service import handle_remix
        await handle_remix(interaction, self.generation_id)
