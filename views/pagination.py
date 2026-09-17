"""
Pagination, Prompt History, Style Browsing, and Interrogation Views for Shallot-CUI Bot.
"""

import logging
import math
import discord
from views.modals import (
    StudyImagineModal, 
    EditStyleModal, 
    EditPromptModal, 
    EditAdoptPromptModal
)

logger = logging.getLogger("DiscordBot.Views.Pagination")


class DescribeButtons(discord.ui.View):
    def __init__(self, generation_id: str, ar: str = "16:9", sr = True, oga: bool = False, model_choice: str = "hyphoria"):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.ar = ar
        self.sr = sr
        self.oga = oga
        self.model_choice = model_choice

        # Resolve sr mode
        if isinstance(sr, str):
            sr_mode = sr
        elif sr is True:
            sr_mode = 'sr90' if model_choice == 'hyphoria' else 'sr75'
        else:
            sr_mode = 'nosr'
        sr_tag = sr_mode
        oga_tag = 'oga' if self.oga else 'nooga'

        # Row 0: Aspect Ratio Selection (21:9, 16:9, 10:7, 3:5, 9:16)
        ar_options = [("21:9", "21:9"), ("16:9", "16:9"), ("10:7", "10:7"), ("3:5", "3:5"), ("9:16", "9:16")]
        for label, val in ar_options:
            is_selected = (self.ar == val)
            style = discord.ButtonStyle.primary if is_selected else discord.ButtonStyle.secondary
            self.add_item(discord.ui.Button(
                label=f"📐 {label}",
                style=style,
                custom_id=f"set_desc_ar:{self.generation_id}:{val}:{sr_tag}:{oga_tag}:{self.model_choice}",
                row=0
            ))

        # Row 1: LoRA Toggles (Semi-Realism & Ogarla)
        if sr_mode == "nosr":
            sr_label = "✨ Semi-Realism (--sr): OFF"
            sr_style = discord.ButtonStyle.secondary
            toggle_sr_val = "sr60"
        elif sr_mode in ["sr60", "sr.60"]:
            sr_label = "✨ Semi-Realism (--sr.60): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr70"
        elif sr_mode in ["sr70", "sr.70"]:
            sr_label = "✨ Semi-Realism (--sr.70): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr80"
        elif sr_mode in ["sr80", "sr.80"]:
            sr_label = "✨ Semi-Realism (--sr.80): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr90"
        elif sr_mode in ["sr90", "sr.90", "sr"]:
            sr_label = "✨ Semi-Realism (--sr.90): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "nosr"
        else:
            sr_label = "✨ Semi-Realism (--sr): OFF"
            sr_style = discord.ButtonStyle.secondary
            toggle_sr_val = "sr60"

        self.add_item(discord.ui.Button(
            label=sr_label,
            style=sr_style,
            custom_id=f"toggle_desc_sr:{self.generation_id}:{self.ar}:{toggle_sr_val}:{oga_tag}:{self.model_choice}",
            row=1
        ))

        oga_style = discord.ButtonStyle.primary if self.oga else discord.ButtonStyle.secondary
        oga_label = "🌿 Ogarla (--ogarla.70): ON" if self.oga else "🌿 Ogarla (--ogarla.70): OFF"
        toggle_oga_val = 'nooga' if self.oga else 'oga'
        self.add_item(discord.ui.Button(
            label=oga_label,
            style=oga_style,
            custom_id=f"toggle_desc_oga:{self.generation_id}:{self.ar}:{sr_tag}:{toggle_oga_val}:{self.model_choice}",
            row=1
        ))

        # Row 2: Model Toggle (Hyphoria NAI)
        is_hyphoria = (self.model_choice == "hyphoria")
        hyp_style = discord.ButtonStyle.primary if is_hyphoria else discord.ButtonStyle.secondary
        hyp_label = "🤖 Model: Hyphoria NAI" if is_hyphoria else "🤖 Model: Default Checkpoint"
        next_model = "default" if is_hyphoria else "hyphoria"
        self.add_item(discord.ui.Button(
            label=hyp_label,
            style=hyp_style,
            custom_id=f"toggle_desc_model:{self.generation_id}:{self.ar}:{sr_tag}:{oga_tag}:{next_model}",
            row=2
        ))

        # Row 3: Generation Targets & Copy Actions (SDXL, Krea 2, Copy Prompt)
        self.add_item(discord.ui.Button(
            label="🎨 Generate SDXL",
            style=discord.ButtonStyle.primary,
            custom_id=f"gen_desc:{self.generation_id}:sdxl:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Generate Krea 2",
            style=discord.ButtonStyle.danger,
            custom_id=f"gen_desc:{self.generation_id}:krea2:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=3
        ))


class StudyButtons(discord.ui.View):
    def __init__(self, prompt: str, imagine_callback=None):
        super().__init__(timeout=None)
        self.prompt = prompt
        self.imagine_callback = imagine_callback

    @discord.ui.button(label="🎨 Imagine", style=discord.ButtonStyle.primary, custom_id="study_imagine_btn")
    async def imagine_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = StudyImagineModal(initial_prompt=self.prompt, on_submit_callback=self.imagine_callback)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="📋 Copy /imagine", style=discord.ButtonStyle.secondary, custom_id="study_copy_btn")
    async def copy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_text = f"/imagine prompt: {self.prompt}"
        if len(cmd_text) > 1950:
            cmd_text = cmd_text[:1950] + "..."
        await interaction.response.send_message(
            content=f"Copy this into your chat bar to toggle settings:\n```{cmd_text}```",
            ephemeral=True
        )

    @discord.ui.button(label="⭐ Save Prompt", style=discord.ButtonStyle.success, custom_id="study_fav_btn")
    async def fav_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        import db
        short_name = self.prompt[:30].strip() + ("..." if len(self.prompt) > 30 else "")
        db.add_favorite_prompt(interaction.user.id, short_name, self.prompt)
        await interaction.response.send_message(f"⭐ Saved prompt to your favorites (`/my_prompts`)!", ephemeral=True)


class StylePaginationView(discord.ui.View):
    def __init__(self, user_id: int, favorites: list[dict], per_page: int = 8):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.favorites = favorites
        self.per_page = per_page
        self.current_page = 0
        self.selected_code = None

        self._update_components()

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(len(self.favorites) / self.per_page))

    def _update_components(self):
        self.clear_items()
        
        if not self.favorites:
            return

        # Ensure page is within bounds
        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]

        # Check if selected_code is in page_items, else pick first
        code_in_page = any(f["style_code"] == self.selected_code for f in page_items)
        if not code_in_page and page_items:
            self.selected_code = page_items[0]["style_code"]

        # 1. Select menu for current page styles
        options = []
        for fav in page_items:
            code_str = str(fav["style_code"])
            is_default = (self.selected_code == fav["style_code"])
            options.append(discord.SelectOption(
                label=f"{fav['style_name']}"[:100],
                value=code_str,
                description=f"Code: {fav['style_code']}",
                default=is_default
            ))

        select = discord.ui.Select(
            placeholder="Select a style to edit or delete...",
            min_values=1,
            max_values=1,
            options=options,
            row=0
        )
        select.callback = self.on_select_style
        self.add_item(select)

        # 2. Navigation buttons
        prev_btn = discord.ui.Button(
            label="◀ Prev",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page == 0),
            row=1
        )
        prev_btn.callback = self.on_prev_page
        self.add_item(prev_btn)

        next_btn = discord.ui.Button(
            label="Next ▶",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page >= self.total_pages - 1),
            row=1
        )
        next_btn.callback = self.on_next_page
        self.add_item(next_btn)

        # 3. Action buttons (Edit, Delete)
        edit_btn = discord.ui.Button(
            label="✏️ Edit Style",
            style=discord.ButtonStyle.primary,
            row=1
        )
        edit_btn.callback = self.on_edit_style
        self.add_item(edit_btn)

        delete_btn = discord.ui.Button(
            label="🗑️ Delete Style",
            style=discord.ButtonStyle.danger,
            row=1
        )
        delete_btn.callback = self.on_delete_style
        self.add_item(delete_btn)

    def build_embed(self) -> discord.Embed:
        if not self.favorites:
            return discord.Embed(
                title="⭐ Your Favorite Styles",
                description="You have no saved style references yet. Click `⭐ Favorite Style` on any completed image grid to save styles!"
            )

        embed = discord.Embed(
            title="⭐ Your Favorite Styles",
            description="Select a style name or code using the `favorite_style` parameter when running `/imagine`."
        )

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        for fav in page_items:
            s_name = fav['style_name']
            if len(s_name) > 180:
                s_name = s_name[:177] + "..."
            s_prompt = fav['style_prompt']
            if len(s_prompt) > 950:
                s_prompt = s_prompt[:950] + "..."

            field_name = f"✨ {s_name} (Code: `{fav['style_code']}`)"
            if len(field_name) > 250:
                field_name = field_name[:245] + "...)"

            embed.add_field(
                name=field_name,
                value=f"**Prompt:** *{s_prompt}*",
                inline=False
            )

        embed.set_footer(text=f"Page {self.current_page + 1} of {self.total_pages} ({len(self.favorites)} total saved styles)")
        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This menu is for the command caller only.", ephemeral=True)
            return False
        return True

    async def on_select_style(self, interaction: discord.Interaction):
        selected_str = interaction.data["values"][0]
        self.selected_code = int(selected_str)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_prev_page(self, interaction: discord.Interaction):
        if self.current_page > 0:
            self.current_page -= 1
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_next_page(self, interaction: discord.Interaction):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_edit_style(self, interaction: discord.Interaction):
        selected = next((f for f in self.favorites if f["style_code"] == self.selected_code), None)
        if not selected:
            await interaction.response.send_message("Please select a style first.", ephemeral=True)
            return

        modal = EditStyleModal(self.user_id, selected, on_save_callback=self._handle_style_edited)
        await interaction.response.send_modal(modal)

    async def _handle_style_edited(self, interaction: discord.Interaction, style_code: int, new_name: str, new_prompt: str):
        import db
        self.favorites = db.get_favorite_styles(self.user_id)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_delete_style(self, interaction: discord.Interaction):
        import db
        if not self.selected_code:
            await interaction.response.send_message("Please select a style first.", ephemeral=True)
            return

        db.remove_favorite_style(self.user_id, self.selected_code)
        self.favorites = db.get_favorite_styles(self.user_id)
        
        if not self.favorites:
            self.clear_items()
            await interaction.response.edit_message(embed=self.build_embed(), view=None)
            return

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        if page_items:
            self.selected_code = page_items[0]["style_code"]
        else:
            self.current_page = max(0, self.total_pages - 1)
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]

        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)


class PromptPaginationView(discord.ui.View):
    def __init__(self, user_id: int, prompts: list[dict], per_page: int = 5, imagine_callback=None, bertflow_callback=None):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.prompts = prompts
        self.per_page = per_page
        self.imagine_callback = imagine_callback
        self.bertflow_callback = bertflow_callback
        self.current_page = 0
        self.selected_id = None

        self._update_components()

    @property
    def total_pages(self) -> int:
        return max(1, math.ceil(len(self.prompts) / self.per_page))

    def _update_components(self):
        self.clear_items()
        
        if not self.prompts:
            return

        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]

        id_in_page = any(p["id"] == self.selected_id for p in page_items)
        if not id_in_page and page_items:
            self.selected_id = page_items[0]["id"]

        # 1. Select Menu
        options = []
        for p in page_items:
            id_str = str(p["id"])
            is_default = (self.selected_id == p["id"])
            options.append(discord.SelectOption(
                label=f"{p['prompt_name']}"[:100],
                value=id_str,
                description=f"ID: {p['id']}",
                default=is_default
            ))

        select = discord.ui.Select(
            placeholder="Select a prompt to copy, imagine, edit, or delete...",
            min_values=1,
            max_values=1,
            options=options,
            row=0
        )
        select.callback = self.on_select_prompt
        self.add_item(select)

        # 2. Navigation buttons
        prev_btn = discord.ui.Button(
            label="◀ Prev",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page == 0),
            row=1
        )
        prev_btn.callback = self.on_prev_page
        self.add_item(prev_btn)

        next_btn = discord.ui.Button(
            label="Next ▶",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page >= self.total_pages - 1),
            row=1
        )
        next_btn.callback = self.on_next_page
        self.add_item(next_btn)

        # 3. Action buttons
        copy_btn = discord.ui.Button(
            label="📋 Copy Full",
            style=discord.ButtonStyle.secondary,
            row=1
        )
        copy_btn.callback = self.on_copy_prompt
        self.add_item(copy_btn)

        imagine_btn = discord.ui.Button(
            label="🎨 Imagine",
            style=discord.ButtonStyle.primary,
            row=1
        )
        imagine_btn.callback = self.on_imagine_prompt
        self.add_item(imagine_btn)

        if self.bertflow_callback:
            bertflow_btn = discord.ui.Button(
                label="⚡ Bertflow",
                style=discord.ButtonStyle.success,
                row=1
            )
            bertflow_btn.callback = self.on_bertflow_prompt
            self.add_item(bertflow_btn)

        edit_btn = discord.ui.Button(
            label="✏️ Edit",
            style=discord.ButtonStyle.primary,
            row=2
        )
        edit_btn.callback = self.on_edit_prompt
        self.add_item(edit_btn)

        delete_btn = discord.ui.Button(
            label="🗑️ Delete",
            style=discord.ButtonStyle.danger,
            row=2
        )
        delete_btn.callback = self.on_delete_prompt
        self.add_item(delete_btn)

    def build_embed(self) -> discord.Embed:
        if not self.prompts:
            return discord.Embed(
                title="⭐ Your Favorite Prompts",
                description="You have no saved favorite prompts yet! Click **⭐ Favorite Prompt** on any completed image grid to save prompts.",
                color=discord.Color.gold()
            )

        embed = discord.Embed(
            title="⭐ Your Favorite Prompts",
            description=f"You have **{len(self.prompts)}** saved prompt(s). Select a prompt to copy full text, generate, edit, or delete:",
            color=discord.Color.gold()
        )

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        for p in page_items:
            p_name = p['prompt_name']
            if len(p_name) > 180:
                p_name = p_name[:177] + "..."

            p_text = p['prompt_text']
            if len(p_text) > 950:
                display_text = p_text[:950] + "..."
            else:
                display_text = p_text

            field_name = f"📌 {p_name} (ID: {p['id']})"
            if len(field_name) > 250:
                field_name = field_name[:245] + "...)"

            embed.add_field(
                name=field_name,
                value=f"```\n{display_text}\n```",
                inline=False
            )

        embed.set_footer(text=f"Page {self.current_page + 1} of {self.total_pages} ({len(self.prompts)} total saved prompts)")
        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This menu is for the command caller only.", ephemeral=True)
            return False
        return True

    async def on_select_prompt(self, interaction: discord.Interaction):
        selected_str = interaction.data["values"][0]
        self.selected_id = int(selected_str)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_prev_page(self, interaction: discord.Interaction):
        if self.current_page > 0:
            self.current_page -= 1
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_next_page(self, interaction: discord.Interaction):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_copy_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        full_text = selected["prompt_text"]
        header = f"📋 **Full Prompt for \"{selected['prompt_name']}\" (ID: {selected['id']}):**\n"
        
        # If full text fits in a single message codeblock
        if len(header) + len(full_text) + 10 <= 1980:
            content = f"{header}```\n{full_text}\n```"
            await interaction.response.send_message(content=content, ephemeral=True)
        else:
            await interaction.response.send_message(content=f"{header}*(Prompt split across messages due to length)*:", ephemeral=True)
            # Chunk long prompts into 1900 character blocks
            chunk_size = 1900
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size]
                await interaction.followup.send(content=f"```\n{chunk}\n```", ephemeral=True)

    async def on_imagine_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = StudyImagineModal(initial_prompt=selected["prompt_text"], on_submit_callback=self.imagine_callback)
        await interaction.response.send_modal(modal)

    async def on_bertflow_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = StudyImagineModal(initial_prompt=selected["prompt_text"], on_submit_callback=self.bertflow_callback)
        await interaction.response.send_modal(modal)

    async def on_edit_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = EditPromptModal(self.user_id, selected, on_save_callback=self._handle_prompt_edited)
        await interaction.response.send_modal(modal)

    async def _handle_prompt_edited(self, interaction: discord.Interaction, prompt_id: int, new_name: str, new_text: str):
        import db
        self.prompts = db.get_favorite_prompts(self.user_id)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_delete_prompt(self, interaction: discord.Interaction):
        import db
        if not self.selected_id:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        db.remove_favorite_prompt(self.user_id, self.selected_id)
        self.prompts = db.get_favorite_prompts(self.user_id)
        
        if not self.prompts:
            self.clear_items()
            await interaction.response.edit_message(embed=self.build_embed(), view=None)
            return

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        if page_items:
            self.selected_id = page_items[0]["id"]
        else:
            self.current_page = max(0, self.total_pages - 1)
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]

        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)


class AdoptButtons(discord.ui.View):
    def __init__(self, adopt_id: str, ogarla_on: bool = False, cref_weight: float = 0.20, semi_realism_weight: float = 0.0, random_sref_on: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.adopt_id = adopt_id
        self.ogarla_on = ogarla_on
        self.cref_weight = cref_weight
        self.semi_realism_weight = float(semi_realism_weight) if isinstance(semi_realism_weight, (int, float, str)) else 0.0
        self.random_sref_on = random_sref_on

        # Row 0: Prompt Utilities
        self.add_item(discord.ui.Button(
            label="✏️ Edit Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_edit_prompt:{adopt_id}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_copy:{adopt_id}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="⭐ Save Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_save:{adopt_id}",
            row=0
        ))

        # Row 1: Character, Style & Reference Weight Controls
        oga_label = "👩 Ogarla Main: ON" if ogarla_on else "👩 Ogarla Main: OFF"
        oga_style = discord.ButtonStyle.primary if ogarla_on else discord.ButtonStyle.secondary
        self.add_item(discord.ui.Button(
            label=oga_label,
            style=oga_style,
            custom_id=f"adopt_toggle_oga:{adopt_id}",
            row=1
        ))

        if self.semi_realism_weight > 0.0:
            sr_label = f"🌟 Semi-Realism: {self.semi_realism_weight:.2f}"
            sr_style = discord.ButtonStyle.primary
        else:
            sr_label = "🌟 Semi-Realism: OFF"
            sr_style = discord.ButtonStyle.secondary

        self.add_item(discord.ui.Button(
            label=sr_label,
            style=sr_style,
            custom_id=f"adopt_toggle_sr:{adopt_id}",
            row=1
        ))

        rnd_label = "🎲 Random Style: ON" if random_sref_on else "🎲 Random Style: OFF"
        rnd_style = discord.ButtonStyle.primary if random_sref_on else discord.ButtonStyle.secondary
        self.add_item(discord.ui.Button(
            label=rnd_label,
            style=rnd_style,
            custom_id=f"adopt_toggle_sref:{adopt_id}",
            row=1
        ))

        cw_label = f"🎚️ Ref Weight: {cref_weight:.2f}"
        self.add_item(discord.ui.Button(
            label=cw_label,
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_cycle_cw:{adopt_id}",
            row=1
        ))

        # Row 2: Action Launchers (Green buttons at bottom)
        self.add_item(discord.ui.Button(
            label="🎨 Imagine Grid",
            style=discord.ButtonStyle.success,
            custom_id=f"adopt_imagine:{adopt_id}",
            row=2
        ))
