"""
Interaction Dispatcher for Shallot-CUI Bot.
Decouples component button and modal interactions from bot.py into a structured routing service.
"""

import sys
import random
import logging
import discord

import db
from core_helpers import safe_defer
from views import (
    SavedSrefSelectView,
    CustomSrefModal,
    EditBlendKreaModal,
    EditBlendPromptModal,
    EditAdoptPromptModal,
    AdoptButtons,
)

# Default service handlers
from services.generation_service import (
    handle_stasis_pause,
    handle_stasis_resume,
    handle_cancel_generation,
    handle_remix,
    handle_isolate,
    handle_variation,
    handle_reroll,
    handle_favorite_style,
    handle_favorite_prompt,
    handle_copy_prompt,
    handle_outpaint,
    handle_change_sref,
    handle_upscale,
    handle_update_blend_view,
    handle_generate_blended,
    handle_reblend,
    handle_submit_edit_blend_prompts,
    execute_imagine,
    get_generation,
)
from services.krea_service import (
    handle_bertflow_reroll,
    handle_bertflow_remix,
    handle_bertflow_toggle_char,
    handle_bertflow_upscale,
    handle_update_blend_krea_view,
    handle_submit_edit_blend_krea_prompt,
    handle_generate_blend_krea,
)
from services.vision_service import (
    handle_generate_described,
    handle_update_describe_view,
)

logger = logging.getLogger("DiscordBot.InteractionDispatcher")


def _resolve_handler(name: str, fallback_func):
    """
    Resolves a handler dynamically from the 'bot' module if present/patched,
    falling back to the native service implementation.
    """
    bot_mod = sys.modules.get("bot")
    if bot_mod and hasattr(bot_mod, name):
        return getattr(bot_mod, name)
    return fallback_func


async def dispatch_interaction(interaction: discord.Interaction) -> bool:
    """
    Dispatches a component interaction based on its custom_id prefix.
    Returns True if an interaction was handled, False otherwise.
    """
    if interaction.type != discord.InteractionType.component:
        return False

    custom_id = interaction.data.get("custom_id", "")
    if not custom_id:
        return False

    try:
        # 1. Stasis Pause / Resume
        if custom_id.startswith("stasis_pause:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    user_id = int(parts[2])
                    handler = _resolve_handler("handle_stasis_pause", handle_stasis_pause)
                    await handler(interaction, gen_id, user_id)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("stasis_resume:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    user_id = int(parts[2])
                    handler = _resolve_handler("handle_stasis_resume", handle_stasis_resume)
                    await handler(interaction, gen_id, user_id)
                    return True
                except ValueError:
                    pass

        # 2. Cancel / Remix / Reroll
        elif custom_id.startswith("cancel_gen:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_cancel_generation", handle_cancel_generation)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("remix:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_remix", handle_remix)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("reroll:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                handler = _resolve_handler("handle_reroll", handle_reroll)
                await handler(interaction, parts[1])
                return True

        # 3. Quadrant Isolation (U1-U4)
        elif custom_id.startswith("upscale:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    handler = _resolve_handler("handle_isolate", handle_isolate)
                    await handler(interaction, gen_id, index)
                    return True
                except ValueError:
                    pass

        # 4. Variations (V1-V4, subtle, strong)
        elif custom_id.startswith("variation:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[-2]
                try:
                    index = int(parts[-1])
                    handler = _resolve_handler("handle_variation", handle_variation)
                    await handler(interaction, gen_id, index)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("vary_subtle:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    handler = _resolve_handler("handle_variation", handle_variation)
                    await handler(interaction, gen_id, index, denoise_override=0.70, variation_type="Subtle")
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("vary_strong:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    handler = _resolve_handler("handle_variation", handle_variation)
                    await handler(interaction, gen_id, index, denoise_override=0.95, variation_type="Strong")
                    return True
                except ValueError:
                    pass

        # 5. Bertflow Buttons
        elif custom_id.startswith("bertflow_reroll:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_bertflow_reroll", handle_bertflow_reroll)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("bertflow_remix:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_bertflow_remix", handle_bertflow_remix)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("bertflow_toggle_char:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_bertflow_toggle_char", handle_bertflow_toggle_char)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("bertflow_upscale:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_bertflow_upscale", handle_bertflow_upscale)
                await handler(interaction, parts[1])
                return True

        # 6. Favorites & Prompts
        elif custom_id.startswith("fav_style:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                handler = _resolve_handler("handle_favorite_style", handle_favorite_style)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("fav_prompt:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                handler = _resolve_handler("handle_favorite_prompt", handle_favorite_prompt)
                await handler(interaction, parts[1])
                return True
        elif custom_id.startswith("copy_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_copy_prompt", handle_copy_prompt)
                await handler(interaction, parts[1])
                return True

        # 7. Upscale & Outpaint
        elif custom_id.startswith("upscale_run:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    scale = parts[3]
                    handler = _resolve_handler("handle_upscale", handle_upscale)
                    await handler(interaction, gen_id, index, upscale_scale=scale)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("upscale_redo:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    scale = parts[3] if len(parts) >= 4 else "2.0"
                    handler = _resolve_handler("handle_upscale", handle_upscale)
                    await handler(interaction, gen_id, index, force_new_seed=True, upscale_scale=scale)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("outpaint:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    target_ratio = parts[3]
                    if len(parts) == 5:
                        target_ratio = f"{parts[3]}:{parts[4]}"
                    handler = _resolve_handler("handle_outpaint", handle_outpaint)
                    await handler(interaction, gen_id, index, target_ratio)
                    return True
                except ValueError:
                    pass

        # 8. SREF Management
        elif custom_id.startswith("sref_change_random:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    new_code = str(random.randint(100000, 999999))
                    handler = _resolve_handler("handle_change_sref", handle_change_sref)
                    await handler(interaction, gen_id, index, new_code)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("sref_change_saved:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    favorites = db.get_favorite_styles(interaction.user.id)
                    if not favorites:
                        await interaction.response.send_message(
                            "⭐ You don't have any saved favorite styles yet! Save some using the **Favorite Style** button on generations, or using `/my_prompts`.", 
                            ephemeral=True
                        )
                    else:
                        handler = _resolve_handler("handle_change_sref", handle_change_sref)
                        view = SavedSrefSelectView(gen_id, index, favorites, select_callback=handler)
                        await interaction.response.send_message("⭐ **Select a saved style to apply to this image (same prompt & seed):**", view=view, ephemeral=True)
                    return True
                except ValueError:
                    pass
        elif custom_id.startswith("sref_change_custom:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                try:
                    index = int(parts[2])
                    handler = _resolve_handler("handle_change_sref", handle_change_sref)
                    await interaction.response.send_modal(CustomSrefModal(gen_id, index, on_submit_callback=handler))
                    return True
                except ValueError:
                    pass

        # 9. Describe Controls
        elif custom_id.startswith("gen_desc:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                desc_type = parts[2]
                if len(parts) == 8:
                    ar = f"{parts[3]}:{parts[4]}"
                    use_sr = parts[5]
                    use_oga = (parts[6] == "oga")
                    model_choice = parts[7]
                elif len(parts) == 7:
                    ar = parts[3]
                    use_sr = parts[4]
                    use_oga = (parts[5] == "oga")
                    model_choice = parts[6]
                else:
                    ar = "16:9"
                    use_sr = "nosr"
                    use_oga = False
                    model_choice = "hyphoria"

                await safe_defer(interaction)
                handler = _resolve_handler("handle_generate_described", handle_generate_described)
                await handler(interaction, gen_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice)
                return True
        elif custom_id.startswith("set_desc_ar:") or custom_id.startswith("toggle_desc_sr:") or custom_id.startswith("toggle_desc_oga:") or custom_id.startswith("toggle_desc_model:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                gen_id = parts[1]
                if len(parts) == 7:
                    new_ar = f"{parts[2]}:{parts[3]}"
                    new_sr = parts[4]
                    new_oga = (parts[5] == "oga")
                    new_model = parts[6]
                elif len(parts) == 6:
                    new_ar = parts[2]
                    new_sr = parts[3]
                    new_oga = (parts[4] == "oga")
                    new_model = parts[5]
                else:
                    new_ar = parts[2]
                    new_sr = "nosr"
                    new_oga = False
                    new_model = "hyphoria"
                handler = _resolve_handler("handle_update_describe_view", handle_update_describe_view)
                await handler(interaction, gen_id, new_ar, new_sr, new_oga, new_model)
                return True

        # 10. Blend Krea Controls
        elif custom_id.startswith("set_blend_krea_ar:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                val = parts[2]
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_ar=val)
                return True
        elif custom_id.startswith("toggle_blend_krea_model:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = db.get_generation(gen_id) or get_generation(gen_id) or {}
                cur_model = gen_data.get("model_choice", "muse")
                next_model = "pornmaster" if "muse" in cur_model.lower() else "muse"
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_model=next_model)
                return True
        elif custom_id.startswith("toggle_blend_krea_steps:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = db.get_generation(gen_id) or get_generation(gen_id) or {}
                cur_steps = int(gen_data.get("steps", 8))
                # Cycle 8 -> 10 -> 12 -> 16 -> 8
                step_cycle = {8: 10, 10: 12, 12: 16}
                next_steps = step_cycle.get(cur_steps, 8)
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_steps=next_steps)
                return True
        elif custom_id.startswith("toggle_blend_krea_wetness:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = db.get_generation(gen_id) or get_generation(gen_id) or {}
                cur_wet = float(gen_data.get("wetness", -2.0))
                if cur_wet <= -1.0:
                    next_wet = 0.0
                elif cur_wet < 0.5:
                    next_wet = 1.0
                else:
                    next_wet = -2.0
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_wetness=next_wet)
                return True
        elif custom_id.startswith("set_blend_krea_comp:"):
            parts = custom_id.split(":")
            if len(parts) >= 2 and interaction.data and "values" in interaction.data:
                gen_id = parts[1]
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_comp=val)
                return True
        elif custom_id.startswith("toggle_blend_krea_composition:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = db.get_generation(gen_id) or get_generation(gen_id) or {}
                cur_comp = gen_data.get("composition", "off")
                next_comp = "subtle" if cur_comp == "off" else ("medium" if cur_comp == "subtle" else ("strong" if cur_comp == "medium" else "off"))
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_comp=next_comp)
                return True
        elif custom_id.startswith("set_blend_krea_char:"):
            parts = custom_id.split(":")
            if len(parts) >= 2 and interaction.data and "values" in interaction.data:
                gen_id = parts[1]
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_char=val)
                return True
        elif custom_id.startswith("set_blend_krea_celeb:"):
            parts = custom_id.split(":")
            if len(parts) >= 2 and interaction.data and "values" in interaction.data:
                gen_id = parts[1]
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_krea_view", handle_update_blend_krea_view)
                await handler(interaction, gen_id, new_celeb=val)
                return True
        elif custom_id.startswith("edit_blend_krea_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id) or {}
                cur_prompt = gen_data.get("fused_prompt") or gen_data.get("krea2_prompt") or gen_data.get("user_prompt", "")
                submit_cb = _resolve_handler("handle_submit_edit_blend_krea_prompt", handle_submit_edit_blend_krea_prompt)
                modal = EditBlendKreaModal(
                    generation_id=gen_id,
                    current_prompt=cur_prompt,
                    on_submit_callback=submit_cb
                )
                try:
                    if not interaction.response.is_done():
                        await interaction.response.send_modal(modal)
                except discord.HTTPException as e:
                    if e.code != 40060:
                        logger.warning(f"Failed to send EditBlendKreaModal: {e}")
                return True
        elif custom_id.startswith("gen_blend_krea:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_generate_blend_krea", handle_generate_blend_krea)
                await handler(interaction, parts[1])
                return True

        # 11. Blend SDXL Studio Controls
        elif custom_id.startswith("edit_blend_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id)
                if not gen_data:
                    await interaction.response.send_message("⚠️ Blend session data expired.", ephemeral=True)
                    return True
                current_cap = gen_data.get("caption", "")
                current_det = gen_data.get("detailed_caption", "")
                current_extra = gen_data.get("extra_details", "")
                submit_cb = _resolve_handler("handle_submit_edit_blend_prompts", handle_submit_edit_blend_prompts)
                modal = EditBlendPromptModal(
                    generation_id=gen_id,
                    current_caption=current_cap,
                    current_detailed=current_det,
                    current_extra=current_extra,
                    on_submit_callback=submit_cb
                )
                await interaction.response.send_modal(modal)
                return True
        elif custom_id.startswith("cycle_blend_ar:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                ar_list = ["16:9", "21:9", "10:7", "1:1", "3:5", "9:16"]
                cur_ar = gen_data.get("ar", "16:9")
                try:
                    idx = ar_list.index(cur_ar)
                    next_ar = ar_list[(idx + 1) % len(ar_list)]
                except ValueError:
                    next_ar = "16:9"
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_ar=next_ar)
            return True
        elif custom_id.startswith("cycle_blend_comp:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                comp_list = ["style", "low", "med", "high"]
                cur_comp = gen_data.get("comp_strength", "style")
                try:
                    idx = comp_list.index(cur_comp)
                    next_comp = comp_list[(idx + 1) % len(comp_list)]
                except ValueError:
                    next_comp = "low"
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_comp=next_comp)
            return True
        elif custom_id.startswith("toggle_blend_sref:") or custom_id.startswith("cycle_blend_style:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                cur_sref = gen_data.get("sref_rand", "nosref")
                is_on = cur_sref in ["sref", "sref1", True] or (isinstance(cur_sref, str) and cur_sref.lower() in ["true", "1", "on"])
                next_sref = "nosref" if is_on else "sref"
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_sref=next_sref)
            return True
        elif custom_id.startswith("switch_blend_tab:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                tab = parts[2]
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, tab=tab)
            return True
        elif custom_id.startswith("set_blend_char:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_char=val)
            return True
        elif custom_id.startswith("set_blend_sr:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_sr=val)
            return True
        elif custom_id.startswith("set_blend_style:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_sref=val)
            return True
        elif custom_id.startswith("set_blend_ar:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
            if len(parts) == 2 and interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handler(interaction, gen_id, new_ar=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handler(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
            return True
        elif custom_id.startswith("set_blend_model:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
                await handler(interaction, gen_id, new_model=val)
            return True
        elif custom_id.startswith("set_blend_comp:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
            if len(parts) == 2 and interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handler(interaction, gen_id, new_comp=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handler(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
            return True
        elif custom_id.startswith("toggle_blend_sr:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
            if len(parts) == 3:
                val = parts[2]
                await handler(interaction, gen_id, new_sr=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handler(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
            return True
        elif custom_id.startswith("toggle_blend_oga:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
            if len(parts) == 3:
                val = (parts[2] == "oga")
                await handler(interaction, gen_id, new_oga=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handler(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
            return True
        elif custom_id.startswith("toggle_blend_model:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            handler = _resolve_handler("handle_update_blend_view", handle_update_blend_view)
            if len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handler(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
            return True
        elif custom_id.startswith("blend_desc:"):
            parts = custom_id.split(":")
            handler = _resolve_handler("handle_generate_blended", handle_generate_blended)
            if len(parts) == 3:
                gen_id = parts[1]
                desc_type = parts[2]
                gen_data = get_generation(gen_id) or {}
                ar = gen_data.get("ar", "16:9")
                use_sr = gen_data.get("sr", True)
                use_oga = gen_data.get("oga", False)
                char_choice = gen_data.get("char_choice", "ogarla" if use_oga else "none")
                model_choice = gen_data.get("model_choice", "wai")
                comp_strength = gen_data.get("comp_strength", "style")
                use_sref = gen_data.get("sref_rand", "nosref")
                await safe_defer(interaction)
                await handler(interaction, gen_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice, comp_strength=comp_strength, use_sref_rand=use_sref, char_choice=char_choice)
                return True
            elif len(parts) >= 9:
                gen_id = parts[1]
                desc_type = parts[2]
                if len(parts) == 10:
                    ar = f"{parts[3]}:{parts[4]}"
                    use_sr = parts[5]
                    use_oga = (parts[6] == "oga")
                    model_choice = parts[7]
                    comp_strength = parts[8]
                    use_sref = parts[9]
                else:
                    ar = parts[3]
                    use_sr = parts[4]
                    use_oga = (parts[5] == "oga")
                    model_choice = parts[6]
                    comp_strength = parts[7]
                    use_sref = parts[8] if len(parts) > 8 else "nosref"
                await safe_defer(interaction)
                await handler(interaction, gen_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice, comp_strength=comp_strength, use_sref_rand=use_sref)
                return True
        elif custom_id.startswith("reblend:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                handler = _resolve_handler("handle_reblend", handle_reblend)
                await handler(interaction, parts[1])
                return True

        # 12. Adopt Post Actions
        elif custom_id.startswith("adopt_imagine:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data and "prompt" in data:
                    await safe_defer(interaction, thinking=True)
                    ref_url = data.get("image_url")
                    ref_weight = data.get("cref_weight", 0.20)
                    jump_url = data.get("jump_url")
                    prompt_str = data["prompt"]
                    if data.get("ogarla"):
                        if "ogarla" not in prompt_str.lower() and "oga" not in prompt_str.lower():
                            prompt_str = f"ogarla, {prompt_str} --ogarla.75"
                    if data.get("random_sref"):
                        if "--sref" not in prompt_str.lower():
                            prompt_str = f"{prompt_str} --sref random"
                    sr_w = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
                    sr_val = f"--sr {sr_w:.2f}" if sr_w > 0.0 else None
                    handler = _resolve_handler("execute_imagine", execute_imagine)
                    await handler(
                        interaction,
                        prompt=prompt_str,
                        semi_realism=sr_val,
                        reference_image_url=ref_url,
                        reference_image_weight=ref_weight,
                        original_post_url=jump_url
                    )
                else:
                    await interaction.response.send_message("⚠️ Adopted post data expired or not found.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_toggle_oga:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    new_oga = not data.get("ogarla", False)
                    data["ogarla"] = new_oga
                    db.save_generation(adopt_id, data)
                    cw = data.get("cref_weight", 0.20)
                    sr = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    rnd = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=new_oga, cref_weight=cw, semi_realism_weight=sr, random_sref_on=rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_toggle_sr:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    curr_sr = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
                    sr_weights = [0.00, 0.40, 0.60, 0.80, 1.00]
                    idx = 0
                    min_diff = 999
                    for i, w in enumerate(sr_weights):
                        if abs(w - curr_sr) < min_diff:
                            min_diff = abs(w - curr_sr)
                            idx = i
                    next_idx = (idx + 1) % len(sr_weights)
                    new_sr = sr_weights[next_idx]
                    data["semi_realism_weight"] = new_sr
                    data["semi_realism"] = (new_sr > 0.0)
                    db.save_generation(adopt_id, data)
                    cw = data.get("cref_weight", 0.20)
                    oga = data.get("ogarla", False)
                    rnd = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga, cref_weight=cw, semi_realism_weight=new_sr, random_sref_on=rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_toggle_sref:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    new_rnd = not data.get("random_sref", False)
                    data["random_sref"] = new_rnd
                    db.save_generation(adopt_id, data)
                    cw = data.get("cref_weight", 0.20)
                    oga = data.get("ogarla", False)
                    sr = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga, cref_weight=cw, semi_realism_weight=sr, random_sref_on=new_rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_cycle_cw:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    curr_cw = data.get("cref_weight", 0.20)
                    weights = [0.20, 0.40, 0.60, 0.80, 1.00]
                    idx = 0
                    min_diff = 999
                    for i, w in enumerate(weights):
                        if abs(w - curr_cw) < min_diff:
                            min_diff = abs(w - curr_cw)
                            idx = i
                    next_idx = (idx + 1) % len(weights)
                    new_cw = weights[next_idx]
                    data["cref_weight"] = new_cw
                    db.save_generation(adopt_id, data)
                    oga_on = data.get("ogarla", False)
                    sr_w = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    rnd_on = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga_on, cref_weight=new_cw, semi_realism_weight=sr_w, random_sref_on=rnd_on)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_edit_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data and "prompt" in data:
                    from cogs.imagine_cog import handle_submit_edit_adopt_prompt
                    submit_cb = _resolve_handler("handle_submit_edit_adopt_prompt", handle_submit_edit_adopt_prompt)
                    modal = EditAdoptPromptModal(adopt_id, current_prompt=data["prompt"], on_submit_callback=submit_cb)
                    await interaction.response.send_modal(modal)
                else:
                    await interaction.response.send_message("⚠️ Adopted post data expired or not found.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_copy:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                prompt_text = data.get("prompt", "") if data else ""
                if prompt_text:
                    await interaction.response.send_message(
                        content=f"📋 **Adopted Prompt:**\n```{prompt_text}```",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message("⚠️ Prompt not found.", ephemeral=True)
                return True
        elif custom_id.startswith("adopt_save:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                prompt_text = data.get("prompt", "") if data else ""
                if prompt_text:
                    short_name = prompt_text[:30].strip() + ("..." if len(prompt_text) > 30 else "")
                    db.add_favorite_prompt(interaction.user.id, short_name, prompt_text)
                    await interaction.response.send_message(f"⭐ Saved prompt to your favorites (`/my_prompts`)!", ephemeral=True)
                else:
                    await interaction.response.send_message("⚠️ Prompt not found.", ephemeral=True)
                return True

    except Exception as e:
        logger.error(f"Error dispatching interaction '{custom_id}': {e}", exc_info=e)
        if not interaction.response.is_done():
            try:
                await interaction.response.send_message("❌ An error occurred while processing this action.", ephemeral=True)
            except Exception:
                pass
        return True

    return False
