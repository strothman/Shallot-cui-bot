import logging
import discord
from core_helpers import send_error_fallback

logger = logging.getLogger("DiscordBot")

class CustomSrefModal(discord.ui.Modal, title="Change Style Reference (--sref)"):
    sref_input = discord.ui.TextInput(
        label="Enter --sref code, style name, or image URL",
        placeholder="e.g. 772382, --sref 492104, or https://...",
        required=True,
        max_length=200
    )

    def __init__(self, generation_id: str, index: int, on_submit_callback=None):
        super().__init__()
        self.generation_id = generation_id
        self.index = index
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = self.sref_input.value.strip()
            if val and self.on_submit_callback:
                await self.on_submit_callback(interaction, self.generation_id, self.index, val)
        except Exception as e:
            logger.error(f"Error in CustomSrefModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to apply style reference: {e}")



class SavedSrefSelectView(discord.ui.View):
    def __init__(self, generation_id: str, index: int, favorites: list[dict], select_callback=None):
        super().__init__(timeout=60)
        self.generation_id = generation_id
        self.index = index
        self.callback_fn = select_callback

        options = []
        for fav in favorites[:25]:
            code = fav.get("style_code")
            name = fav.get("style_name", f"Style {code}")
            options.append(discord.SelectOption(
                label=name[:100],
                value=str(code),
                description=f"--sref {code}"
            ))

        select = discord.ui.Select(
            placeholder="⭐ Choose a saved style from /my_prompts...",
            min_values=1,
            max_values=1,
            options=options
        )
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        selected_code = interaction.data["values"][0]
        if self.callback_fn:
            await self.callback_fn(interaction, self.generation_id, self.index, selected_code)


class IsolatedImageButtons(discord.ui.View):
    def __init__(self, generation_id: str, index: int, has_sref: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.index = index

        # Row 0: Upscaling Options
        self.add_item(discord.ui.Button(
            label="⚡ Detailed Upscale (1.25x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:1.25",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Creative Upscale (1.5x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:1.5",
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
    def __init__(self, generation_id: str, index: int, upscale_scale: str = "1.25", has_sref: bool = False):
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
    def __init__(self, generation_id, has_sref=False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id

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

        # Row 3: Generation Targets (Caption vs Detailed)
        self.add_item(discord.ui.Button(
            label="🎨 Generate Caption",
            style=discord.ButtonStyle.primary,
            custom_id=f"gen_desc:{self.generation_id}:caption:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Generate Detailed",
            style=discord.ButtonStyle.success,
            custom_id=f"gen_desc:{self.generation_id}:detailed:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))

def build_blend_embed(gen_data: dict, author_str: str = "User", image_url: str = None, is_edited: bool = False) -> discord.Embed:
    """Builds a streamlined, professional embed for /blend sessions."""
    caption = gen_data.get("caption", "No caption")
    detailed = gen_data.get("detailed_caption", "No detailed description")
    extra = gen_data.get("extra_details", "").strip()
    
    ar = gen_data.get("ar", "16:9")
    sr = gen_data.get("sr", True)
    oga = gen_data.get("oga", False)
    model_choice = gen_data.get("model_choice", "wai")
    comp_strength = gen_data.get("comp_strength", "style")
    sref_rand = gen_data.get("sref_rand", "nosref")

    model_names = {
        "wai": "Wai Illustrious SDXL v1.70",
        "illustrious_realism": "Illustrious Realism v1.0",
        "realvis": "RealVisXL V4.0",
        "juggernaut": "Juggernaut XL",
        "copax": "Copax Timeless XL",
        "ultra": "Ultra Realistic XL v2.5",
        "hyphoria": "Hyphoria NAI",
        "nova": "Nova Furry",
        "default": "Wai Illustrious SDXL v1.70"
    }
    model_display = model_names.get(model_choice, model_choice)

    comp_names = {
        "style": "🎨 Style Only (0.20)",
        "low": "🖼️ Low Comp (0.35)",
        "med": "🖼️ Med Comp (0.60)",
        "high": "🖼️ High Comp (0.85)"
    }
    comp_display = comp_names.get(comp_strength, comp_strength)

    if sr is False or sr == "nosr":
        sr_display = "OFF"
    elif isinstance(sr, str) and sr.startswith("sr"):
        sr_display = f"ON (--{sr})"
    else:
        sr_display = "ON"

    oga_display = "ON (--ogarla.70)" if oga else "OFF"
    
    if sref_rand == "nosref" or not sref_rand:
        sref_display = "OFF"
    elif sref_rand in ["sref", "sref1", True]:
        sref_display = "1 Style"
    elif sref_rand == "sref5":
        sref_display = "5 Styles"
    elif sref_rand == "sref10":
        sref_display = "10 Styles"
    else:
        sref_display = str(sref_rand)

    title_text = "🎨 Image Blend Studio (Customized)" if is_edited else "🎨 Image Blend Studio"
    embed = discord.Embed(
        title=title_text,
        description="*Select your aspect ratio, model, and reference composition below, then click **Blend**.*",
        color=discord.Color.from_rgb(0, 168, 252)
    )
    
    embed.add_field(
        name="📋 Short Caption / Tags",
        value=f"> {caption[:900]}",
        inline=False
    )
    
    embed.add_field(
        name="🔍 Detailed Vision Description",
        value=f"> {detailed[:1000]}",
        inline=False
    )

    if extra:
        embed.add_field(
            name="✨ Custom Extra Details",
            value=f"> `{extra}`",
            inline=False
        )

    config_badges = (
        f"**📐 Ratio:** `{ar}` • **🤖 Model:** `{model_display}`\n"
        f"**🖼️ Reference:** `{comp_display}`\n"
        f"**✨ Semi-Realism:** `{sr_display}` • **🌿 Ogarla:** `{oga_display}` • **🎲 Style Random:** `{sref_display}`"
    )
    embed.add_field(
        name="⚙️ Active Configuration",
        value=config_badges,
        inline=False
    )

    if image_url:
        embed.set_thumbnail(url=image_url)
    embed.set_footer(text=f"Florence-2 Vision AI • Requested by {author_str}")
    return embed


class BlendButtons(discord.ui.View):
    def __init__(self, generation_id: str, ar: str = "16:9", sr = True, oga: bool = False, model_choice: str = "wai", comp_strength: str = "style", sref_rand = "nosref"):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.ar = ar
        self.sr = sr
        self.oga = oga
        self.model_choice = model_choice
        self.comp_strength = comp_strength
        self.sref_rand = sref_rand

        # Resolve tags
        sr_tag = sr if isinstance(sr, str) else ('sr90' if sr else 'nosr')
        oga_tag = 'oga' if self.oga else 'nooga'
        sref_tag = sref_rand if isinstance(sref_rand, str) else ('sref' if sref_rand else 'nosref')

        # Row 0: Aspect Ratio Selection Dropdown
        ar_options = [
            discord.SelectOption(label="16:9 Landscape (1344x768)", value="16:9", emoji="📐", default=(self.ar == "16:9")),
            discord.SelectOption(label="21:9 Ultra-Wide (1536x640)", value="21:9", emoji="📐", default=(self.ar == "21:9")),
            discord.SelectOption(label="10:7 Standard / Tablet (1216x832)", value="10:7", emoji="📐", default=(self.ar == "10:7")),
            discord.SelectOption(label="1:1 Square (1024x1024)", value="1:1", emoji="📐", default=(self.ar == "1:1")),
            discord.SelectOption(label="3:5 Portrait (832x1216)", value="3:5", emoji="📐", default=(self.ar == "3:5")),
            discord.SelectOption(label="9:16 Tall Portrait (768x1344)", value="9:16", emoji="📐", default=(self.ar == "9:16")),
        ]
        ar_select = discord.ui.Select(
            placeholder="📐 Select Aspect Ratio...",
            options=ar_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_ar:{self.generation_id}",
            row=0
        )
        self.add_item(ar_select)

        # Row 1: Model Checkpoint Dropdown
        model_options = [
            discord.SelectOption(label="Wai Illustrious SDXL v1.70 (Anime / Illustration)", value="wai", emoji="🌸", default=(self.model_choice in ["wai", "default"])),
            discord.SelectOption(label="Illustrious Realism v1.0 (Anime Realism)", value="illustrious_realism", emoji="🎨", default=(self.model_choice == "illustrious_realism")),
            discord.SelectOption(label="RealVisXL V4.0 (Photorealistic)", value="realvis", emoji="📸", default=(self.model_choice == "realvis")),
            discord.SelectOption(label="Juggernaut XL (Balanced Realism)", value="juggernaut", emoji="⚔️", default=(self.model_choice == "juggernaut")),
            discord.SelectOption(label="Copax Timeless XL (Cinematic)", value="copax", emoji="🎬", default=(self.model_choice == "copax")),
            discord.SelectOption(label="Ultra Realistic XL v2.5 (Fine Details)", value="ultra", emoji="✨", default=(self.model_choice == "ultra")),
            discord.SelectOption(label="Hyphoria NAI (Illustrious NAI)", value="hyphoria", emoji="🤖", default=(self.model_choice == "hyphoria")),
            discord.SelectOption(label="Nova Furry (Stylized)", value="nova", emoji="🦊", default=(self.model_choice == "nova")),
        ]
        model_select = discord.ui.Select(
            placeholder="🤖 Select Model Checkpoint...",
            options=model_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_model:{self.generation_id}",
            row=1
        )
        self.add_item(model_select)

        # Row 2: Reference & Composition Strength Dropdown
        comp_options = [
            discord.SelectOption(label="Style Reference Only (0.20 weight)", value="style", emoji="🎨", description="Transfers style/colors without copying layout", default=(self.comp_strength == "style")),
            discord.SelectOption(label="Light Composition (0.35 weight)", value="low", emoji="🖼️", description="Soft pose/layout reference", default=(self.comp_strength == "low")),
            discord.SelectOption(label="Medium Composition (0.60 weight)", value="med", emoji="🖼️", description="Balanced character identity & layout", default=(self.comp_strength == "med")),
            discord.SelectOption(label="Strong Composition (0.85 weight)", value="high", emoji="🖼️", description="Strict pose, framing & structural locking", default=(self.comp_strength == "high")),
        ]
        comp_select = discord.ui.Select(
            placeholder="🖼️ Select Reference & Composition Strength...",
            options=comp_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_comp:{self.generation_id}",
            row=2
        )
        self.add_item(comp_select)

        # Resolve sr mode
        if isinstance(sr, str):
            sr_mode = sr
        elif sr is True:
            sr_mode = 'sr90' if model_choice == 'hyphoria' else 'sr75'
        else:
            sr_mode = 'nosr'
        sr_tag = sr_mode
        oga_tag = 'oga' if self.oga else 'nooga'

        # Resolve sref_mode string
        if isinstance(sref_rand, str):
            sref_mode = sref_rand
        elif sref_rand is True:
            sref_mode = 'sref'
        else:
            sref_mode = 'nosref'
        sref_tag = sref_mode

        # Row 3: Quick Toggles (Semi-Realism, Ogarla, Style Random)
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
            custom_id=f"toggle_blend_sr:{self.generation_id}:{self.ar}:{toggle_sr_val}:{oga_tag}:{self.model_choice}:{self.comp_strength}:{sref_tag}",
            row=3
        ))

        oga_style = discord.ButtonStyle.primary if self.oga else discord.ButtonStyle.secondary
        oga_label = "🌿 Ogarla (--ogarla.70): ON" if self.oga else "🌿 Ogarla (--ogarla.70): OFF"
        toggle_oga_val = 'nooga' if self.oga else 'oga'
        self.add_item(discord.ui.Button(
            label=oga_label,
            style=oga_style,
            custom_id=f"toggle_blend_oga:{self.generation_id}:{self.ar}:{sr_tag}:{toggle_oga_val}:{self.model_choice}:{self.comp_strength}:{sref_tag}",
            row=3
        ))

        # Define button properties & cycle order for Style Random / Style Batch
        if sref_mode == "nosref":
            sref_label = "🎲 Style Random (--sref random): OFF"
            sref_style = discord.ButtonStyle.secondary
            toggle_sref_val = "sref"
        elif sref_mode in ["sref", "sref1"]:
            sref_label = "🎲 Style Random (--sref random): 1 Style"
            sref_style = discord.ButtonStyle.primary
            toggle_sref_val = "sref5"
        elif sref_mode == "sref5":
            sref_label = "🎲 Style Batch (--sref random): 5 Styles"
            sref_style = discord.ButtonStyle.primary
            toggle_sref_val = "sref10"
        elif sref_mode == "sref10":
            sref_label = "🎲 Style Batch (--sref random): 10 Styles"
            sref_style = discord.ButtonStyle.primary
            toggle_sref_val = "sref15"
        elif sref_mode == "sref15":
            sref_label = "🎲 Style Batch (--sref random): 15 Styles"
            sref_style = discord.ButtonStyle.primary
            toggle_sref_val = "nosref"
        else:
            sref_label = "🎲 Style Random (--sref random): OFF"
            sref_style = discord.ButtonStyle.secondary
            toggle_sref_val = "sref"

        self.add_item(discord.ui.Button(
            label=sref_label,
            style=sref_style,
            custom_id=f"toggle_blend_sref:{self.generation_id}:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}:{self.comp_strength}:{toggle_sref_val}",
            row=3
        ))

        # Row 4: Action Launchers & Edit Prompt
        self.add_item(discord.ui.Button(
            label="✏️ Edit & Add Details",
            style=discord.ButtonStyle.secondary,
            custom_id=f"edit_blend_prompt:{self.generation_id}",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Blend with Caption",
            style=discord.ButtonStyle.success,
            custom_id=f"blend_desc:{self.generation_id}:caption",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Blend with Detailed",
            style=discord.ButtonStyle.success,
            custom_id=f"blend_desc:{self.generation_id}:detailed",
            row=4
        ))


class EditBlendPromptModal(discord.ui.Modal, title="✏️ Edit & Refine Blend Prompts"):
    def __init__(self, generation_id: str, current_caption: str, current_detailed: str, current_extra: str = "", on_submit_callback=None):
        super().__init__()
        self.generation_id = generation_id
        self.on_submit_callback = on_submit_callback

        self.caption_input = discord.ui.TextInput(
            label="Caption (Short / Tags)",
            style=discord.TextStyle.paragraph,
            default=current_caption[:400] if current_caption else "",
            max_length=1000,
            required=False,
            placeholder="e.g. 1girl, solo, palshallot, adventurer outfit, looking at viewer"
        )
        self.add_item(self.caption_input)

        self.detailed_input = discord.ui.TextInput(
            label="Detailed Vision Description",
            style=discord.TextStyle.paragraph,
            default=current_detailed[:1500] if current_detailed else "",
            max_length=2000,
            required=False,
            placeholder="Edit or expand the scene description generated by Florence-2"
        )
        self.add_item(self.detailed_input)

        self.extra_input = discord.ui.TextInput(
            label="✨ Extra Details / Add-ons (Optional)",
            style=discord.TextStyle.short,
            default=current_extra[:200] if current_extra else "",
            max_length=400,
            required=False,
            placeholder="e.g. glowing magic aura, sunset rim light, cinematic lens flare"
        )
        self.add_item(self.extra_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            if self.on_submit_callback:
                await self.on_submit_callback(
                    interaction,
                    self.generation_id,
                    self.caption_input.value.strip(),
                    self.detailed_input.value.strip(),
                    self.extra_input.value.strip()
                )
        except Exception as e:
            logger.error(f"Error in EditBlendPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to update blend prompt: {e}")



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


class StudyImagineModal(discord.ui.Modal, title="🎨 Imagine with Extracted Prompt"):
    prompt_input = discord.ui.TextInput(
        label="Prompt",
        style=discord.TextStyle.paragraph,
        placeholder="Enter or edit your prompt...",
        required=True,
        max_length=4000
    )
    flags_input = discord.ui.TextInput(
        label="Optional Flags (--ar, --sr, --sref, etc.)",
        style=discord.TextStyle.short,
        placeholder="e.g. --ar 16:9 --sr.60 --sref 772382",
        required=False,
        max_length=500
    )

    def __init__(self, initial_prompt: str, on_submit_callback=None):
        super().__init__()
        self.prompt_input.default = initial_prompt[:4000]
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            full_prompt = self.prompt_input.value.strip()
            flags = self.flags_input.value.strip()
            if flags:
                full_prompt = f"{full_prompt} {flags}"
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, full_prompt)
        except Exception as e:
            logger.error(f"Error in StudyImagineModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to queue generation: {e}")



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


class EditStyleModal(discord.ui.Modal, title="✏️ Edit Style Reference"):
    def __init__(self, user_id: int, style: dict, on_save_callback=None):
        super().__init__()
        self.user_id = user_id
        self.style_code = style["style_code"]
        self.on_save_callback = on_save_callback

        self.name_input = discord.ui.TextInput(
            label="Style Name",
            placeholder="e.g. Gothic Watercolor Wash",
            default=style.get("style_name", "")[:100],
            required=True,
            max_length=100
        )
        self.prompt_input = discord.ui.TextInput(
            label="Style Prompt",
            style=discord.TextStyle.paragraph,
            placeholder="Enter style prompt description...",
            default=style.get("style_prompt", "")[:2000],
            required=True,
            max_length=2000
        )
        self.add_item(self.name_input)
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            import db
            new_name = self.name_input.value.strip()
            new_prompt = self.prompt_input.value.strip()
            
            updated = db.update_favorite_style(self.user_id, self.style_code, new_name, new_prompt)
            if updated:
                if self.on_save_callback:
                    await self.on_save_callback(interaction, self.style_code, new_name, new_prompt)
                else:
                    await interaction.response.send_message(f"✅ Updated style `{new_name}` (Code: `{self.style_code}`).", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Failed to update style.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in EditStyleModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit style: {e}")



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
        import math
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


class EditPromptModal(discord.ui.Modal, title="✏️ Edit Saved Prompt"):
    def __init__(self, user_id: int, prompt_data: dict, on_save_callback=None):
        super().__init__()
        self.user_id = user_id
        self.prompt_id = prompt_data["id"]
        self.on_save_callback = on_save_callback

        self.name_input = discord.ui.TextInput(
            label="Prompt Label / Nickname",
            placeholder="e.g. Semi-realism Girl",
            default=prompt_data.get("prompt_name", "")[:100],
            required=True,
            max_length=100
        )
        self.prompt_input = discord.ui.TextInput(
            label="Full Prompt Text",
            style=discord.TextStyle.paragraph,
            placeholder="Enter prompt text...",
            default=prompt_data.get("prompt_text", "")[:4000],
            required=True,
            max_length=4000
        )
        self.add_item(self.name_input)
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            import db
            new_name = self.name_input.value.strip()
            new_text = self.prompt_input.value.strip()
            
            updated = db.update_favorite_prompt(self.user_id, self.prompt_id, new_name, new_text)
            if updated:
                if self.on_save_callback:
                    await self.on_save_callback(interaction, self.prompt_id, new_name, new_text)
                else:
                    await interaction.response.send_message(f"✅ Updated prompt **\"{new_name}\"** (ID: `{self.prompt_id}`).", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Failed to update prompt.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in EditPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit prompt: {e}")



class PromptPaginationView(discord.ui.View):
    def __init__(self, user_id: int, prompts: list[dict], per_page: int = 5, imagine_callback=None):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.prompts = prompts
        self.per_page = per_page
        self.imagine_callback = imagine_callback
        self.current_page = 0
        self.selected_id = None

        self._update_components()

    @property
    def total_pages(self) -> int:
        import math
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


class EditAdoptPromptModal(discord.ui.Modal, title="✏️ Edit Adopted Prompt"):
    prompt_input = discord.ui.TextInput(
        label="Prompt Text",
        style=discord.TextStyle.paragraph,
        placeholder="Edit your prompt...",
        required=True,
        max_length=4000
    )

    def __init__(self, adopt_id: str, current_prompt: str, on_submit_callback=None):
        super().__init__()
        self.adopt_id = adopt_id
        self.prompt_input.default = current_prompt[:4000]
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            new_prompt = self.prompt_input.value.strip()
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, self.adopt_id, new_prompt)
        except Exception as e:
            logger.error(f"Error in EditAdoptPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit adopted prompt: {e}")



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
        self.add_item(discord.ui.Button(
            label="✨ Flux HD",
            style=discord.ButtonStyle.success,
            custom_id=f"adopt_flux:{adopt_id}",
            row=2
        ))



class VideoPromptModal(discord.ui.Modal, title="🎬 Animate Image to Video"):
    """Modal allowing the user to configure motion prompt and Wan 2.2 settings before queuing video generation."""
    def __init__(self, default_prompt: str = "", on_submit_callback=None):
        super().__init__()
        self.on_submit_callback = on_submit_callback

        self.prompt_input = discord.ui.TextInput(
            label="Motion Prompt",
            style=discord.TextStyle.paragraph,
            placeholder="Describe the desired video motion (e.g. hair flowing in wind, subtle smile, slow zoom)",
            default=default_prompt[:800] if default_prompt else "",
            max_length=1000,
            required=True
        )
        self.add_item(self.prompt_input)

        self.duration_input = discord.ui.TextInput(
            label="Duration in Seconds (5 or 10)",
            style=discord.TextStyle.short,
            placeholder="5 or 10 (default: 5)",
            default="5",
            max_length=2,
            required=False
        )
        self.add_item(self.duration_input)

        self.smoothness_input = discord.ui.TextInput(
            label="Smoothness Mode (smooth / fast)",
            style=discord.TextStyle.short,
            placeholder="smooth (32 FPS) or fast (16 FPS)",
            default="smooth",
            max_length=10,
            required=False
        )
        self.add_item(self.smoothness_input)

        self.seed_input = discord.ui.TextInput(
            label="Seed (Optional)",
            style=discord.TextStyle.short,
            placeholder="Leave empty for random seed",
            default="",
            max_length=20,
            required=False
        )
        self.add_item(self.seed_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            if self.on_submit_callback:
                await self.on_submit_callback(
                    interaction,
                    self.prompt_input.value.strip(),
                    self.duration_input.value.strip(),
                    self.smoothness_input.value.strip(),
                    self.seed_input.value.strip()
                )
        except Exception as e:
            logger.error(f"Error in VideoPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to queue video animation: {e}")







