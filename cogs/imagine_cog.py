"""
Imagine Cog for Shallot-CUI Bot.
Houses slash commands and context menus for core image generation:
- /imagine (SDXL, Face Detailer, Powerhouse refiner, LoRAs, Character Presets)
- /study (Prompt extraction from image metadata)
- 'Adopt Post / Image' message context menu (Midjourney & Discord post adoption)
"""

import re
import random
import logging
import discord
from discord import app_commands
from discord.ext import commands

from config import SDXL_CHECKPOINT_CHOICES, SDXL_ENHANCEMENT_CHOICES
from model_architecture import Architecture
from characters import get_character_autocomplete_choices
from parsers import extract_positive_prompt, parse_cref
from core_helpers import safe_defer, send_followup_fallback
from views import StudyButtons, AdoptButtons
import db

logger = logging.getLogger("DiscordBot.ImagineCog")


class ImagineCog(commands.Cog):
    """Cog handling /imagine, /study, and post adoption commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Register context menu explicitly on cog initialization
        self.adopt_context_menu = app_commands.ContextMenu(
            name="Adopt Post / Image",
            callback=self.adopt_post_context
        )
        self.bot.tree.add_command(self.adopt_context_menu)

    def cog_unload(self):
        self.bot.tree.remove_command(self.adopt_context_menu.name, type=self.adopt_context_menu.type)

    def _get_imagine_executor(self):
        """Resolves execute_imagine function dynamically to support testing mocks."""
        return getattr(self.bot, "execute_imagine", None)

    # =========================================================================
    # /imagine
    # =========================================================================

    @app_commands.command(
        name="imagine",
        description="🌸 Create 4 pictures at once with SDXL, Face Detailer, Powerhouse refiner, or LoRAs!"
    )
    @app_commands.describe(
        prompt="The prompt to generate images from (supports wildcards {a|b|c}, --smart, --magic, etc.)", 
        checkpoint="The checkpoint model to use",
        enhancements="⚡ Studio & pipeline presets (Ultimate Quality, Smart Director, Magic, Powerhouse)",
        aspect_ratio="Aspect ratio for generated images (--ar)",
        semi_realism="Select Semi-realism LoRA strength (--sr weight)",
        character="Select Character LoRA preset (Ogarla or Valerie)",
        favorite_style="Apply one of your saved favorite styles",
        favorite_prompt="Apply one of your saved favorite prompts",
        style_reference="An image to use as style reference (--sref)"
    )
    @app_commands.choices(
        checkpoint=SDXL_CHECKPOINT_CHOICES,
        enhancements=SDXL_ENHANCEMENT_CHOICES,
        semi_realism=[
            app_commands.Choice(name="✨ Semi-Realism (.60 - Light)", value="sr.60"),
            app_commands.Choice(name="✨ Semi-Realism (.70 - Medium)", value="sr.70"),
            app_commands.Choice(name="✨ Semi-Realism (.80 - High)", value="sr.80"),
            app_commands.Choice(name="✨ Semi-Realism (.90 - Maximum)", value="sr.90"),
        ],
        aspect_ratio=[
            app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
            app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
            app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
            app_commands.Choice(name="10:7 (iPad)", value="10:7"),
            app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
            app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
        ]
    )
    async def imagine(
        self,
        interaction: discord.Interaction, 
        prompt: str, 
        checkpoint: str = None, 
        enhancements: str = None,
        aspect_ratio: str = None,
        semi_realism: str = None,
        character: str = None,
        favorite_style: str = None,
        favorite_prompt: str = None,
        style_reference: discord.Attachment = None
    ):
        if favorite_prompt:
            clean_fav = favorite_prompt.replace("📌", "").strip()
            fav_text = None
            user_prompts = db.get_favorite_prompts(interaction.user.id)
            for item in user_prompts:
                p_id = str(item['id'])
                p_name = item['prompt_name'].strip()
                p_full = item['prompt_text'].strip()
                
                if (p_id == favorite_prompt or p_id == clean_fav or 
                    p_name == favorite_prompt or p_name == clean_fav or
                    p_name.lower() == clean_fav.lower() or
                    p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                    (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                    (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                    fav_text = item['prompt_text']
                    break

            if not fav_text:
                fav_text = clean_fav

            prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

        # Defer response since generation takes time
        await safe_defer(interaction, thinking=True)
        imagine_func = self._get_imagine_executor()
        if imagine_func:
            await imagine_func(
                interaction, prompt, None, checkpoint, style_reference, 
                favorite_style=favorite_style, semi_realism=semi_realism, 
                aspect_ratio=aspect_ratio, character=character, enhancements=enhancements
            )
        else:
            await send_followup_fallback(interaction, content="❌ Generation service is unavailable.", ephemeral=True)

    @imagine.autocomplete('character')
    async def imagine_character_autocomplete(self, interaction: discord.Interaction, current: str):
        return get_character_autocomplete_choices(current, Architecture.SDXL)

    @imagine.autocomplete('favorite_style')
    async def imagine_favorite_style_autocomplete(self, interaction: discord.Interaction, current: str):
        choices = []
        random_label = "🎲 Random (--sref random)"
        if not current or current.lower() in "random" or current.lower() in random_label.lower():
            choices.append(app_commands.Choice(name=random_label, value="random"))

        favorites = db.get_favorite_styles(interaction.user.id)
        for fav in favorites:
            label = f"{fav['style_name']} ({fav['style_code']})"
            if current.lower() in label.lower():
                choices.append(app_commands.Choice(name=label[:100], value=str(fav['style_code'])))
        return choices[:25]

    @imagine.autocomplete('favorite_prompt')
    async def imagine_favorite_prompt_autocomplete(self, interaction: discord.Interaction, current: str):
        prompts = db.get_favorite_prompts(interaction.user.id)
        choices = []
        for item in prompts:
            label = f"📌 {item['prompt_name']}".strip()
            if not current or current.lower() in label.lower() or current.lower() in item['prompt_text'].lower():
                choices.append(app_commands.Choice(name=label[:100], value=str(item['id'])))
        return choices[:25]

    # =========================================================================
    # /study
    # =========================================================================

    async def run_study_imagine_callback(self, interaction: discord.Interaction, prompt: str):
        await safe_defer(interaction, thinking=True)
        imagine_func = self._get_imagine_executor()
        if imagine_func:
            await imagine_func(interaction, prompt)
        else:
            await send_followup_fallback(interaction, content="❌ Generation service is unavailable.", ephemeral=True)

    @app_commands.command(name="study", description="Extract the positive prompt embedded in an uploaded image.")
    @app_commands.describe(
        image="The image file (PNG/JPG) you want to extract the prompt from"
    )
    async def study(self, interaction: discord.Interaction, image: discord.Attachment):
        """Extracts the positive prompt used to make a previous image from PNG metadata."""
        await safe_defer(interaction, thinking=True)

        if not image.content_type or not image.content_type.startswith("image/"):
            await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)
            return

        try:
            image_bytes = await image.read()
            prompt = extract_positive_prompt(image_bytes)

            if not prompt or prompt == "NOT FOUND":
                await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)
                return

            if len(prompt) > 3900:
                formatted_prompt = prompt[:3900] + "..."
            else:
                formatted_prompt = prompt

            embed = discord.Embed(
                title="🔍 Extracted Positive Prompt",
                description=f"```\n{formatted_prompt}\n```",
                color=discord.Color.blue()
            )
            embed.set_footer(text=f"Extracted from {image.filename}")

            view = StudyButtons(prompt=prompt, imagine_callback=self.run_study_imagine_callback)
            await send_followup_fallback(interaction, embed=embed, view=view, ephemeral=False)
        except Exception as e:
            logger.error(f"Error in /study command: {e}")
            await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)

    # =========================================================================
    # Adopt Post / Image Context Menu
    # =========================================================================

    async def adopt_post_context(self, interaction: discord.Interaction, message: discord.Message):
        """Context menu command to adopt any Discord image or post (ComfyUI, User Upload, Midjourney)."""
        await self.execute_adopt_post(interaction, message)

    async def execute_adopt_post(self, interaction: discord.Interaction, message: discord.Message):
        """Core logic to adopt any Discord post or image (Midjourney, ComfyUI, user upload)."""
        await safe_defer(interaction, thinking=True)
        try:
            from parsers import parse_adopted_post
            parsed = parse_adopted_post(message)
            adopt_id = f"adopt_{message.id}_{random.randint(1000, 9999)}"

            parsed_prompt = parsed["clean_prompt"]
            image_url = parsed["image_url"]

            florence_prompt = None
            if image_url:
                try:
                    from services.vision_service import run_florence_interrogate
                    logger.info("Running Florence-2 vision model to generate SDXL prompt for adopted image...")
                    florence_prompt = await run_florence_interrogate(image_url)
                except Exception as flor_err:
                    logger.debug(f"Florence interrogate note during adopt post: {flor_err}")

            if florence_prompt:
                ar_match = re.search(r'--ar\s+\d+:\d+', parsed_prompt, flags=re.IGNORECASE)
                ar_flag = f" {ar_match.group(0)}" if ar_match else ""
                final_prompt = f"{florence_prompt}{ar_flag}"
                logger.info(f"Florence-2 generated SDXL prompt: {final_prompt}")
            else:
                final_prompt = parsed_prompt

            _, _, parsed_cw = parse_cref(parsed_prompt)
            initial_cw = parsed_cw if parsed_cw is not None else 0.20
            initial_oga = "ogarla" in parsed_prompt.lower() or "oga" in parsed_prompt.lower()
            initial_sr_has = "semi-realism" in parsed_prompt.lower() or "hyper-realistic" in parsed_prompt.lower() or "octane render" in parsed_prompt.lower()
            initial_sr_w = 0.85 if initial_sr_has else 0.0
            initial_rnd = "--sref" in parsed_prompt.lower()

            db.save_generation(adopt_id, {
                "prompt": final_prompt,
                "original_prompt": parsed_prompt,
                "image_url": image_url,
                "author_str": parsed["author_str"],
                "jump_url": parsed["jump_url"],
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "cref_weight": initial_cw,
                "ogarla": initial_oga,
                "semi_realism": initial_sr_has,
                "semi_realism_weight": initial_sr_w,
                "random_sref": initial_rnd
            })

            is_mj = "midjourney" in str(parsed["author_str"]).lower() or "midjourney" in str(message.author.name).lower()
            title_str = "⛵ Adopted Midjourney Post" if is_mj else "⛵ Adopted Post / Image"

            embed = discord.Embed(
                title=title_str,
                description=f"```\n{final_prompt}\n```\n"
                            f"**Original Author:** {parsed['author_str']}\n"
                            f"**Source Message:** [Jump to Message]({parsed['jump_url']})",
                color=discord.Color.from_rgb(0, 168, 252)
            )

            has_image = bool(image_url)
            if has_image:
                embed.set_image(url=image_url)
                embed.set_footer(text="Florence-2 SDXL description generated! Use controls below to customize or generate.")
            else:
                embed.set_footer(text="Prompt extracted! Use the buttons below to generate or save.")

            view = AdoptButtons(adopt_id=adopt_id, ogarla_on=initial_oga, cref_weight=initial_cw, semi_realism_weight=initial_sr_w, random_sref_on=initial_rnd)
            await send_followup_fallback(interaction, embed=embed, view=view, ephemeral=False)
        except Exception as e:
            logger.error(f"Error adopting post: {e}")
            await send_followup_fallback(interaction, content=f"⚠️ Failed to parse message: {e}", ephemeral=True)

    async def handle_submit_edit_adopt_prompt(self, interaction: discord.Interaction, adopt_id: str, new_prompt: str):
        """Updates the prompt of an adopted post and edits the embed description in place."""
        data = db.get_generation(adopt_id)
        if not data:
            await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
            return

        data["prompt"] = new_prompt
        db.save_generation(adopt_id, data)

        author_str = data.get("author_str", "@Midjourney Bot")
        jump_url = data.get("jump_url", "")
        image_url = data.get("image_url")
        oga_on = data.get("ogarla", False)
        cw = data.get("cref_weight", 0.20)
        sr_w = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
        rnd_on = data.get("random_sref", False)

        embed = discord.Embed(
            title="⛵ Adopted Midjourney Post (Edited)",
            description=f"```\n{new_prompt}\n```\n"
                        f"**Original Author:** {author_str}\n"
                        f"**Source Message:** [Jump to Message]({jump_url})",
            color=discord.Color.from_rgb(0, 168, 252)
        )
        if image_url:
            embed.set_image(url=image_url)
            embed.set_footer(text="Prompt updated! Use controls below to customize or generate.")
        else:
            embed.set_footer(text="Prompt updated! Use controls below to customize or generate.")

        view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga_on, cref_weight=cw, semi_realism_weight=sr_w, random_sref_on=rnd_on)
        await interaction.response.edit_message(embed=embed, view=view)
