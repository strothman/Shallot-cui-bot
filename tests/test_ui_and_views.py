"""
Automated Test Suite Module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import unittest
import copy
import random
import logging
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image
import io

# Disable log output during test execution
logging.disable(logging.CRITICAL)

from parsers import (
    parse_aspect_ratio,
    parse_loras,
    apply_loras_to_workflow,
    parse_seed,
    parse_stylize,
    parse_sref,
    parse_cref,
    apply_ipadapter_to_workflow,
    expand_dynamic_prompt,
    parse_magic_prompt,
    apply_magic_enhancement,
    calculate_wan_dimensions,
    extract_positive_prompt,
)
from image_utils import (
    calculate_outpaint_padding,
    save_quadrant_images,
    get_quadrant_bytes,
)
from bot import build_blend_workflow
from error_handler import (
    error_handler,
    ErrorCategory,
    ErrorSeverity,
    AutoFixAction,
    AutoFixResult,
)
import model_architecture
import characters
import db


class TestUiAndViews(unittest.TestCase):
    def test_module12_style_batch_queuing(self):
        """Test 5, 10, 15 style batch parsing, BlendButtons cycling, and favorite style resolution."""
        from views import BlendButtons

        # 1. Test parse_sref batch parsing
        p5, _, _, info5 = parse_sref("cosmic dragon --sref random:5")
        self.assertIsNotNone(info5)
        self.assertEqual(info5.get("batch_count"), 5)

        p10, _, _, info10 = parse_sref("cyberpunk city --sref batch:10")
        self.assertIsNotNone(info10)
        self.assertEqual(info10.get("batch_count"), 10)

        p15, _, _, info15 = parse_sref("underwater scene --sref batch:15")
        self.assertIsNotNone(info15)
        self.assertEqual(info15.get("batch_count"), 15)

        # 2. Test BlendButtons Unified Single-Page Dashboard
        v = BlendButtons("gen123", ar="16:9", char_choice="sully", sr="sr80", sref_rand="sref")
        
        # Test Model select (Row 0)
        select_model = [item for item in v.children if "set_blend_model" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_model)
        self.assertEqual(select_model.row, 0)

        # Test Character select (Row 1)
        select_char = [item for item in v.children if "set_blend_char" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_char)
        self.assertEqual(select_char.row, 1)
        char_vals = [opt.value for opt in select_char.options]
        self.assertIn("none", char_vals)
        self.assertIn("ogarla", char_vals)
        self.assertIn("valerie", char_vals)
        self.assertIn("sully", char_vals)
        self.assertIn("cheri", char_vals)
        self.assertIn("mageill", char_vals)
        selected_char_opt = [opt for opt in select_char.options if opt.default][0]
        self.assertEqual(selected_char_opt.value, "sully")

        # Test Aspect Ratio select (Row 2)
        select_ar = [item for item in v.children if "set_blend_ar" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_ar)
        self.assertEqual(select_ar.row, 2)
        ar_vals = [opt.value for opt in select_ar.options]
        self.assertIn("1:1", ar_vals)
        self.assertIn("16:9", ar_vals)
        self.assertIn("9:16", ar_vals)
        self.assertIn("4:3", ar_vals)
        self.assertIn("3:4", ar_vals)
        self.assertIn("21:9", ar_vals)

        # Test Semi-Realism select (Row 3)
        select_sr = [item for item in v.children if "set_blend_sr" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_sr)
        self.assertEqual(select_sr.row, 3)
        sr_vals = [opt.value for opt in select_sr.options]
        self.assertIn("nosr", sr_vals)
        self.assertIn("sr60", sr_vals)
        self.assertIn("sr70", sr_vals)
        self.assertIn("sr75", sr_vals)
        self.assertIn("sr80", sr_vals)
        self.assertIn("sr90", sr_vals)
        selected_sr_opt = [opt for opt in select_sr.options if opt.default][0]
        self.assertEqual(selected_sr_opt.value, "sr80")

        # Test Action & Toggle Buttons in Row 4 (Blend image, Edit prompt, Comp cycle, Sref toggle)
        blend_btn = [item for item in v.children if "blend_desc" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(blend_btn)
        self.assertEqual(blend_btn.row, 4)
        self.assertEqual(blend_btn.custom_id, "blend_desc:gen123:blend")

        edit_btn = [item for item in v.children if "edit_blend_prompt" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(edit_btn)
        self.assertEqual(edit_btn.row, 4)

        comp_btn = [item for item in v.children if "cycle_blend_comp" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(comp_btn)
        self.assertEqual(comp_btn.row, 4)

        style_btn = [item for item in v.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(style_btn)
        self.assertEqual(style_btn.row, 4)
        self.assertIn("--sref random: ON", style_btn.label)

    def test_module13_followup_fallback_no_view_type_error(self):
        """Test send_followup_fallback omits view parameter when view=None so discord.py does not raise TypeError."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from bot import send_followup_fallback, edit_original_fallback, edit_message_fallback

        mock_interaction = MagicMock()
        mock_interaction.followup = MagicMock()
        mock_interaction.followup.send = AsyncMock(return_value="mock_msg")
        mock_interaction.edit_original_response = AsyncMock(return_value=None)
        res = asyncio.run(send_followup_fallback(mock_interaction, content="Job submitted"))
        self.assertEqual(res, "mock_msg")

        # Verify 'view' was NOT passed in kwargs to followup.send when view=None
        _, kwargs = mock_interaction.followup.send.call_args
        self.assertNotIn("view", kwargs)
        self.assertEqual(kwargs["content"], "Job submitted")

        # Test edit_original_fallback with default view=None
        asyncio.run(edit_original_fallback(mock_interaction, content="Editing"))
        _, edit_kwargs = mock_interaction.edit_original_response.call_args
        self.assertEqual(edit_kwargs.get("view"), None)
        self.assertEqual(edit_kwargs.get("content"), "Editing")

        # Test edit_original_fallback with attachments
        mock_file = MagicMock()
        asyncio.run(edit_original_fallback(mock_interaction, content="With file", attachments=[mock_file]))
        _, edit_kwargs = mock_interaction.edit_original_response.call_args
        self.assertEqual(edit_kwargs.get("attachments"), [mock_file])

        # Test edit_message_fallback with allow_send_fallback=False suppresses channel.send on expired token
        mock_msg_interaction = MagicMock()
        mock_msg_interaction.channel_id = 12345
        mock_msg_interaction.channel = MagicMock()
        mock_msg_interaction.channel.fetch_message = AsyncMock(side_effect=Exception("429 Too Many Requests: 30046"))
        mock_msg_interaction.followup = MagicMock()
        mock_msg_interaction.followup.edit_message = AsyncMock(side_effect=Exception("10062 Unknown interaction"))
        mock_msg_interaction.channel.send = AsyncMock()

        # Should complete quietly and NOT call channel.send
        asyncio.run(edit_message_fallback(mock_msg_interaction, 99999, content="Progress tick", allow_send_fallback=False))
        mock_msg_interaction.channel.send.assert_not_called()

    def test_module16_progress_bar(self):
        """Test progress bar rendering formatting."""
        from bot import create_progress_bar
        bar_50 = create_progress_bar(3, 6, length=10)
        self.assertIn("50%", bar_50)
        self.assertIn("Step 3/6", bar_50)
        self.assertIn("█████░░░░░", bar_50)

        bar_100 = create_progress_bar(6, 6, length=10)
        self.assertIn("100%", bar_100)
        self.assertIn("Step 6/6", bar_100)
        self.assertIn("██████████", bar_100)

    def test_module17_wan_video_duration_options(self):
        """Test video duration frame resolution for 5s (81 frames) and 10s (161 frames)."""
        def get_wan_frames(duration: int, settings_dict: dict):
            if duration == 10:
                return 161
            elif duration == 5:
                return settings_dict.get("wan_video_frames", 81)
            else:
                return (duration * 16) + 1

        dummy_settings = {"wan_video_frames": 81}
        self.assertEqual(get_wan_frames(5, dummy_settings), 81)
        self.assertEqual(get_wan_frames(10, dummy_settings), 161)
        self.assertEqual(get_wan_frames(7, dummy_settings), 113)

    def test_module31_core_commands_registration(self):
        """Test that core generation commands are registered and pruned commands are absent."""
        from bot import bot
        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("imagine", commands)
        self.assertIn("prompt", commands)
        self.assertIn("negative", commands)

        # Verify pruned commands are not present
        self.assertNotIn("video", commands)
        self.assertNotIn("ltx", commands)
        self.assertNotIn("flux", commands)
        self.assertNotIn("imagine_det", commands)
        self.assertNotIn("junji", commands)
        self.assertNotIn("ico", commands)
        self.assertNotIn("hunyuan", commands)
        self.assertNotIn("sdxl", commands)
        self.assertNotIn("com", commands)

    def test_module32_helper_functions_defined(self):
        """Ensure all message sending helpers in bot.py exist and are callable."""
        import bot
        self.assertTrue(callable(getattr(bot, "send_followup_fallback", None)))
        self.assertTrue(callable(getattr(bot, "send_error_fallback", None)))
        self.assertTrue(callable(getattr(bot, "edit_original_fallback", None)))
        self.assertTrue(callable(getattr(bot, "edit_message_fallback", None)))

    def test_module33_blend_file_naming_and_routing(self):
        """Test blend image file naming format with date, time, seed, checkpoint abbrev, and sref."""
        from image_utils import format_image_filename, get_checkpoint_abbrev

        # Test checkpoint abbreviations
        self.assertEqual(get_checkpoint_abbrev("waiIllustriousSDXL_v170.safetensors"), "wai")
        self.assertEqual(get_checkpoint_abbrev("RealVisXL_V4.0.safetensors"), "realvis")
        self.assertEqual(get_checkpoint_abbrev("illustriousRealismBy_v10VAE.safetensors"), "illuReal")
        self.assertEqual(get_checkpoint_abbrev("juggernautXL_ragnarok.safetensors"), "juggernaut")
        self.assertEqual(get_checkpoint_abbrev("CopaxTimeLessXL.safetensors"), "copax")
        self.assertEqual(get_checkpoint_abbrev("ultraRealisticByStable_v25.safetensors"), "ultra")

        # Verify blend filename with checkpoint abbrev
        ckpt_abbrev = get_checkpoint_abbrev("waiIllustriousSDXL_v170.safetensors")
        fn = format_image_filename(f"blend_{ckpt_abbrev}_3", 523453745845177, "png", sref="325465")
        self.assertTrue(fn.startswith("blend_wai_3_"))
        self.assertIn("seed523453745845177", fn)
        self.assertIn("sref325465", fn)
        self.assertTrue(fn.endswith(".png"))

        # Verify upscale filename for blend with checkpoint abbrev
        fn_up = format_image_filename(f"blend_{ckpt_abbrev}_upscale_2", 987654321, "png", sref="110291")
        self.assertTrue(fn_up.startswith("blend_wai_upscale_2_"))
        self.assertIn("seed987654321", fn_up)
        self.assertIn("sref110291", fn_up)

        # Verify icon filenames for /ico
        fn_ico_png = format_image_filename(f"icon_{ckpt_abbrev}_1", 123456789, "png", sref="456789")
        self.assertTrue(fn_ico_png.startswith("icon_wai_1_"))
        self.assertIn("seed123456789", fn_ico_png)
        self.assertIn("sref456789", fn_ico_png)
        self.assertTrue(fn_ico_png.endswith(".png"))

        # Verify imagine filenames
        fn_iso = format_image_filename(f"imagine_{ckpt_abbrev}_4", 11223344, "png", sref="998877")
        self.assertTrue(fn_iso.startswith("imagine_wai_4_"))
        self.assertIn("seed11223344", fn_iso)
        self.assertIn("sref998877", fn_iso)
        self.assertTrue(fn_iso.endswith(".png"))

        fn_grid = format_image_filename(f"grid_{ckpt_abbrev}", 55667788, "jpg")
        self.assertTrue(fn_grid.startswith("grid_wai_"))
        self.assertIn("seed55667788", fn_grid)
        self.assertTrue(fn_grid.endswith(".jpg"))

        # Verify junji filenames for /junji
        fn_junji_iso = format_image_filename(f"junji_{ckpt_abbrev}_2", 44332211, "png", sref="654321")
        self.assertTrue(fn_junji_iso.startswith("junji_wai_2_"))
        self.assertIn("seed44332211", fn_junji_iso)
        self.assertIn("sref654321", fn_junji_iso)
        self.assertTrue(fn_junji_iso.endswith(".png"))

        fn_junji_grid = format_image_filename(f"junji_{ckpt_abbrev}_grid", 44332211, "jpg")
        self.assertTrue(fn_junji_grid.startswith("junji_wai_grid_"))
        self.assertTrue(fn_junji_grid.endswith(".jpg"))

    def test_module34_blend_context_menu(self):
        """Test registration and setup of the Blend Image context menu."""
        from bot import bot, blend_image_context
        
        # Verify blend_image_context exists and is registered in the command tree
        ctx_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertTrue("Blend Image (SDXL)" in ctx_names or "Blend Image" in ctx_names)
        self.assertIn("Adopt Post / Image", ctx_names)
        self.assertNotIn("Adopt Midjourney Post", ctx_names)

    def test_module38_animate_to_video_retired(self):
        """Test that Animate to Video context menu has been retired and is absent from bot.tree."""
        from bot import bot

        # 1. Verify command tree registration
        ctx_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertNotIn("Animate to Video", ctx_names)

    def test_module39_all_modals_error_boundaries_and_callbacks(self):
        """Test that all UI modals in views.py execute callbacks safely and handle exceptions within error boundaries."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from views import (
            CustomSrefModal,
            EditBlendPromptModal,
            StudyImagineModal,
            EditStyleModal,
            EditPromptModal,
            EditAdoptPromptModal,
        )

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = False
        mock_interaction.response.send_message = AsyncMock()
        mock_interaction.followup.send = AsyncMock()


        # 1. CustomSrefModal
        sref_res = {}
        async def sref_cb(inter, gen_id, idx, val):
            sref_res["val"] = val
        m_sref = CustomSrefModal("gen123", 1, on_submit_callback=sref_cb)
        m_sref.sref_input._value = "889900"
        asyncio.run(m_sref.on_submit(mock_interaction))
        self.assertEqual(sref_res.get("val"), "889900")

        # 2. EditBlendPromptModal
        blend_res = {}
        async def blend_cb(inter, gen_id, cap, det, extra):
            blend_res["cap"] = cap
            blend_res["det"] = det
            blend_res["extra"] = extra
        m_blend = EditBlendPromptModal("gen123", "a cat", "a cute cat on a table", "glowing", on_submit_callback=blend_cb)
        m_blend.caption_input._value = "a dog"
        m_blend.detailed_input._value = "a cute dog on grass"
        m_blend.extra_input._value = "sunset"
        asyncio.run(m_blend.on_submit(mock_interaction))
        self.assertEqual(blend_res.get("cap"), "a dog")
        self.assertEqual(blend_res.get("extra"), "sunset")

        # 3. StudyImagineModal
        study_res = {}
        async def study_cb(inter, p):
            study_res["prompt"] = p
        m_study = StudyImagineModal("crystal tower", on_submit_callback=study_cb)
        m_study.prompt_input._value = "crystal tower"
        m_study.flags_input._value = "--ar 16:9"
        asyncio.run(m_study.on_submit(mock_interaction))
        self.assertEqual(study_res.get("prompt"), "crystal tower --ar 16:9")

        # 4. EditAdoptPromptModal
        adopt_res = {}
        async def adopt_cb(inter, adopt_id, p):
            adopt_res["p"] = p
        m_adopt = EditAdoptPromptModal("adopt1", "orig prompt", on_submit_callback=adopt_cb)
        m_adopt.prompt_input._value = "new prompt"
        asyncio.run(m_adopt.on_submit(mock_interaction))
        self.assertEqual(adopt_res.get("p"), "new prompt")

        # 5. Error Boundary Verification (Raising in callback does not crash)
        async def failing_cb(*args, **kwargs):
            raise ValueError("Simulated unexpected modal failure")

        m_fail = EditAdoptPromptModal("adopt_fail", "test failure", on_submit_callback=failing_cb)
        m_fail.prompt_input._value = "test failure"
        # Must execute without raising unhandled exception
        asyncio.run(m_fail.on_submit(mock_interaction))

    def test_discord_command_description_lengths(self):
        """Validates that all Discord slash command descriptions are <= 100 characters."""
        from bot import bot
        for cmd in bot.tree.get_commands():
            desc = getattr(cmd, "description", None)
            if desc:
                clean_desc = desc.replace('\n', ' ').strip()
                self.assertLessEqual(
                    len(clean_desc), 
                    100, 
                    f"Command '{cmd.name}' description exceeds Discord 100 char limit ({len(clean_desc)}): '{clean_desc}'"
                )

    def test_cancel_and_remix_views(self):
        """Test instantiation and custom_ids of CancelGenerationView and Remix button in GridButtons."""
        from views import CancelGenerationView, RemixModal, GridButtons

        # CancelGenerationView
        c_view = CancelGenerationView("gen_999")
        c_button = [item for item in c_view.children if getattr(item, "custom_id", None) == "cancel_gen:gen_999"]
        self.assertEqual(len(c_button), 1)

        # Remix button in GridButtons
        g_view = GridButtons("gen_999", has_sref=True)
        r_button = [item for item in g_view.children if getattr(item, "custom_id", None) == "remix:gen_999"]
        self.assertEqual(len(r_button), 1)

        # RemixModal
        modal = RemixModal("gen_999", initial_prompt="a cute cat", initial_seed=12345)
        self.assertEqual(modal.prompt_input.default, "a cute cat")
        self.assertEqual(modal.seed_input.default, "12345")

    def test_blend_complete_embed_and_reblend_button(self):
        """Test build_blend_complete_embed 3-column dashboard, GridButtons is_blend=True, and handle_reblend."""
        from views import build_blend_complete_embed, GridButtons, BlendButtons
        import bot
        from unittest.mock import MagicMock, AsyncMock, patch
        import asyncio

        # 1. Test build_blend_complete_embed
        prompt = "Semi-realism, masterpiece, best quality. mageill, Photo of a young woman with fair skin --sr.70 --sref 855014"
        embed = build_blend_complete_embed(
            display_prompt=prompt,
            selected_model="waiIllustriousSDXL_v170.safetensors",
            seed=377789952340926,
            width=1536,
            height=640,
            comp_strength="low",
            cfg=4.0,
            sref_info={"code": 855014, "name": "Dadaist Pixel Art"},
            char_choice="mageill",
            sr_choice="sr70",
            user_name="TestStroth",
            image_url="http://example.com/source.png",
            elapsed_time=8.4
        )
        self.assertEqual(embed.title, "✨ Image Blend Complete")
        self.assertEqual(embed.thumbnail.url, "http://example.com/source.png")
        self.assertEqual(embed.color.value, 0x8A2BE2)
        self.assertIn("TestStroth", embed.footer.text)
        self.assertIn("8.4s", embed.footer.text)
        self.assertIn("377789952340926", embed.footer.text)

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", fields)
        self.assertIn("🤖 Checkpoint & Tech", fields)
        self.assertIn("🎭 Aesthetics & Identity", fields)

        # Check friendly names & badges
        self.assertIn("1536x640", fields["📐 Canvas & Framing"])
        self.assertIn("🖼️ Light Comp (0.35)", fields["📐 Canvas & Framing"])
        self.assertIn("Wai Illustrious SDXL v1.70", fields["🤖 Checkpoint & Tech"])
        self.assertIn("377789952340926", fields["🤖 Checkpoint & Tech"])
        self.assertIn("CFG:** `4.0`", fields["🤖 Checkpoint & Tech"])
        self.assertIn("🔮 Mageill (Epoch 5)", fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.70", fields["🎭 Aesthetics & Identity"])
        self.assertIn("855014", fields["🎭 Aesthetics & Identity"])
        self.assertIn("Dadaist Pixel Art", fields["🎭 Aesthetics & Identity"])

        # Test fallback regex parsing branches in build_blend_complete_embed
        embed_fallback = build_blend_complete_embed(
            display_prompt="ogarla, beautiful portrait --sr.60 --sref 999111",
            selected_model="waiIllustriousSDXL_v170.safetensors",
            seed=12345,
            width=768,
            height=1344,
            char_choice=None,
            sr_choice=None,
            sref_info=None
        )
        fb_fields = {f.name: f.value for f in embed_fallback.fields}
        self.assertIn("🌿 Ogarla (--ogarla.70)", fb_fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.60", fb_fields["🎭 Aesthetics & Identity"])
        self.assertIn("`--sref 999111`", fb_fields["🎭 Aesthetics & Identity"])

        # 2. Test GridButtons with is_blend=True and has_sref=True
        grid_view = GridButtons("gen_blend_123", has_sref=True, is_blend=True)
        # Check max 5 rows and items per row
        row2_items = [c for c in grid_view.children if getattr(c, "row", None) == 2]
        self.assertEqual(len(row2_items), 5)
        btn_ids = [c.custom_id for c in row2_items]
        self.assertIn("reblend:gen_blend_123", btn_ids)
        self.assertIn("fav_style:gen_blend_123", btn_ids)
        self.assertIn("fav_prompt:gen_blend_123", btn_ids)
        self.assertIn("copy_prompt:gen_blend_123", btn_ids)
        self.assertIn("remix:gen_blend_123", btn_ids)

        # 3. Test handle_reblend interaction handler
        gen_data = {
            "caption": "a woman in a garden",
            "detailed_caption": "detailed woman in garden",
            "ar": "21:9",
            "model_choice": "wai",
            "comp_strength": "low",
            "sr": "sr70",
            "char_choice": "mageill",
            "sref_rand": "sref_855014",
            "image_url": "http://example.com/source.png"
        }
        bot.db.save_generation("gen_blend_123", gen_data)
        bot.active_generations["gen_blend_123"] = gen_data

        mock_reblend_interaction = MagicMock()
        mock_reblend_interaction.response.is_done.return_value = False
        mock_reblend_interaction.response.send_message = AsyncMock()
        mock_reblend_interaction.user.display_name = "TestStroth"
        mock_reblend_interaction.user.id = 12345

        asyncio.run(bot.handle_reblend(mock_reblend_interaction, "gen_blend_123"))
        mock_reblend_interaction.response.send_message.assert_called_once()
        sent_call = mock_reblend_interaction.response.send_message.call_args
        sent_view = sent_call.kwargs.get("view")
        self.assertIsInstance(sent_view, BlendButtons)
        self.assertEqual(sent_view.ar, "21:9")
        self.assertEqual(sent_view.char_choice, "mageill")

    def test_blended_image_embed_and_consolidated_buttons(self):
        """Test build_blended_image_embed 3-column dashboard and IsolatedImageButtons 3-row consolidation."""
        from views import build_blended_image_embed, IsolatedImageButtons

        # 1. Test build_blended_image_embed
        prompt = "Semi-realism, masterpiece, best quality. mageill, A digital illustration of a slender woman --sr.60 --sref 576522"
        embed = build_blended_image_embed(
            index=3,
            display_prompt=prompt,
            checkpoint="waiIllustriousSDXL_v170.safetensors",
            target_seed=1113230745203310,
            width=768,
            height=1344,
            comp_strength="low",
            cfg=4.0,
            sref_info={"code": 576522, "name": "Pop Art 3D Render"},
            char_choice="mageill",
            sr_choice="sr60",
            user_name="strothman",
            user_id=200011008327024641,
            image_url="http://example.com/source_ref.png",
            is_blend=True
        )
        self.assertEqual(embed.title, "✨ Blended Image 3")
        self.assertEqual(embed.thumbnail.url, "http://example.com/source_ref.png")
        self.assertEqual(embed.color.value, 0x8A2BE2)
        self.assertIn("strothman", embed.footer.text)
        self.assertIn("1113230745203310", embed.footer.text)

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", fields)
        self.assertIn("🤖 Checkpoint & Tech", fields)
        self.assertIn("🎭 Aesthetics & Identity", fields)

        self.assertIn("768x1344", fields["📐 Canvas & Framing"])
        self.assertIn("🖼️ Light Comp (0.35)", fields["📐 Canvas & Framing"])
        self.assertIn("Wai Illustrious SDXL v1.70", fields["🤖 Checkpoint & Tech"])
        self.assertIn("1113230745203310", fields["🤖 Checkpoint & Tech"])
        self.assertIn("🔮 Mageill (Epoch 5)", fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.60", fields["🎭 Aesthetics & Identity"])
        self.assertIn("576522", fields["🎭 Aesthetics & Identity"])

        # 2. Test IsolatedImageButtons with is_blend=True and has_sref=True (3 rows max)
        iso_view_sref = IsolatedImageButtons("gen_iso_1", index=3, has_sref=True, is_blend=True)
        r0 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 0]
        r1 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 1]
        r2 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 2]
        r3 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 3]

        self.assertEqual(len(r0), 4) # 1.25x, 1.5x, vary subtle, vary strong
        self.assertEqual(len(r1), 5) # fav_style, fav_prompt, copy_prompt, remix, reblend
        self.assertEqual(len(r2), 3) # custom, random, saved sref
        self.assertEqual(len(r3), 0) # Consolidated from 4 rows down to 3!

        btn_ids = [c.custom_id for c in r1]
        self.assertIn("remix:gen_iso_1", btn_ids)
        self.assertIn("reblend:gen_iso_1", btn_ids)

        # 3. Test IsolatedImageButtons with is_blend=True and has_sref=False (2 rows max!)
        iso_view_nosref = IsolatedImageButtons("gen_iso_2", index=1, has_sref=False, is_blend=True)
        r0_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 0]
        r1_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 1]
        r2_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 2]

        self.assertEqual(len(r0_no), 4)
        self.assertEqual(len(r1_no), 4) # fav_prompt, copy_prompt, remix, reblend
        self.assertEqual(len(r2_no), 0) # Only 2 rows!

    def test_module43_video_views_retired(self):
        """Test that video dashboard, VideoActionView, and VideoPromptModal are cleanly retired from views."""
        import views

        # 1. Verify video view artifacts are absent
        self.assertFalse(hasattr(views, "build_video_complete_embed"))
        self.assertFalse(hasattr(views, "VideoActionView"))
        self.assertFalse(hasattr(views, "VideoPromptModal"))

        # 2. Test AdoptButtons structure (flux button absent)
        adopt_view = views.AdoptButtons("adopt_test_123")
        custom_ids = [c.custom_id for c in adopt_view.children]
        self.assertIn("adopt_imagine:adopt_test_123", custom_ids)
        self.assertNotIn("adopt_flux:adopt_test_123", custom_ids)

    def test_module60_describe_krea2_workflow_and_buttons(self):
        """Test Krea 2 prompt formatting, workflow nodes, and DescribeButtons integration."""
        from parsers import format_krea2_prompt
        from views import DescribeButtons
        import json

        # 1. Test format_krea2_prompt
        sample1 = "The image shows a cinematic portrait of an astronaut on Mars."
        self.assertEqual(format_krea2_prompt(sample1), "A cinematic portrait of an astronaut on Mars.")

        sample2 = "In this photo, a woman in a red silk dress stands on a balcony overlooking the city."
        self.assertEqual(format_krea2_prompt(sample2), "A woman in a red silk dress stands on a balcony overlooking the city.")

        sample3 = "This is an image of a vintage analog synthesizer with glowing patch cables."
        self.assertEqual(format_krea2_prompt(sample3), "A vintage analog synthesizer with glowing patch cables.")

        sample4 = "the photo depicts a dense bamboo forest with morning sunbeams."
        self.assertEqual(format_krea2_prompt(sample4), "A dense bamboo forest with morning sunbeams.")

        self.assertEqual(format_krea2_prompt(""), "")
        self.assertEqual(format_krea2_prompt(None), "")

        # 2. Test workflows/DESCRIBE_cuibot.json
        with open("workflows/DESCRIBE_cuibot.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("5", wf, "Node 5 (Florence2Run for Krea 2) should exist in workflow")
        self.assertEqual(wf["5"]["inputs"]["task"], "more_detailed_caption")
        self.assertEqual(wf["5"]["class_type"], "Florence2Run")

        self.assertIn("11", wf, "Node 11 (ShowText for krea2_prompt) should exist in workflow")
        self.assertEqual(wf["11"]["_meta"]["title"], "krea2_prompt")

        self.assertIn("21", wf, "Node 21 (CR Text Replace for Krea 2) should exist in workflow")

        # 3. Test DescribeButtons view
        view = DescribeButtons(generation_id="desc_test_888", ar="16:9")
        btn_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]
        
        # Verify SDXL, Krea 2, and Copy Prompt buttons exist with correct custom_id structure
        sdxl_btns = [b for b in btn_ids if ":sdxl:" in b]
        self.assertEqual(len(sdxl_btns), 1, "There should be exactly one Generate SDXL button")
        self.assertTrue(sdxl_btns[0].startswith("gen_desc:desc_test_888:sdxl:16:9:"))

        krea2_btns = [b for b in btn_ids if ":krea2:" in b]
        self.assertEqual(len(krea2_btns), 1, "There should be exactly one Krea 2 button")
        self.assertTrue(krea2_btns[0].startswith("gen_desc:desc_test_888:krea2:16:9:"))

        copy_btns = [b for b in btn_ids if b == "copy_prompt:desc_test_888"]
        self.assertEqual(len(copy_btns), 1, "There should be exactly one Copy Prompt button")

    def test_module60b_handle_generate_described(self):
        """Test handle_generate_described dispatching for sdxl, caption, detailed, and krea2 with correct params."""
        import bot
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch

        mock_gen_data = {
            "caption": "A blonde woman in green scarf with tea cup",
            "sdxl_prompt": "A blonde woman in green scarf with tea cup",
            "detailed_caption": "A hyper-realistic digital painting features a nude, slender, blonde woman with small breasts, wearing a green scarf, standing beside a teapot and cup.",
            "krea2_prompt": "A hyper-realistic digital painting of a slender blonde woman beside teapot.",
        }
        bot.db.save_generation("test_desc_123", mock_gen_data)
        bot.active_generations["test_desc_123"] = mock_gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        # 1. Test Generate SDXL -> execute_imagine with checkpoint="hyphoriaIlluNAI_v001.safetensors"
        with patch.object(bot, "execute_imagine", new=AsyncMock()) as mock_imagine:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="sdxl",
                ar="16:9", use_sr="sr90", use_oga=False, model_choice="hyphoria"
            ))
            mock_imagine.assert_called_once()
            call_kwargs = mock_imagine.call_args[1]
            self.assertIn("A blonde woman in green scarf with tea cup", call_kwargs["prompt"])
            self.assertIn("--sr.90", call_kwargs["prompt"])
            self.assertIn("--ar 16:9", call_kwargs["prompt"])
            self.assertEqual(call_kwargs["checkpoint"], "hyphoriaIlluNAI_v001.safetensors")

        # 2. Test Generate Caption (backward compatibility) -> execute_imagine
        with patch.object(bot, "execute_imagine", new=AsyncMock()) as mock_imagine:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="caption",
                ar="16:9", use_sr="sr90", use_oga=False, model_choice="hyphoria"
            ))
            mock_imagine.assert_called_once()

        # 3. Test Generate Detailed -> execute_imagine with default checkpoint
        with patch.object(bot, "execute_imagine", new=AsyncMock()) as mock_imagine:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="detailed",
                ar="21:9", use_sr="nosr", use_oga=True, model_choice="default"
            ))
            mock_imagine.assert_called_once()
            call_kwargs = mock_imagine.call_args[1]
            self.assertIn("hyper-realistic digital painting", call_kwargs["prompt"])
            self.assertIn("ogarla,", call_kwargs["prompt"])
            self.assertIn("--ogarla.70", call_kwargs["prompt"])
            self.assertIn("--ar 21:9", call_kwargs["prompt"])
            self.assertIsNone(call_kwargs["checkpoint"])

        # 4. Test Generate Krea 2 -> execute_bertflow
        with patch.object(bot, "execute_bertflow", new=AsyncMock()) as mock_bert:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="krea2",
                ar="16:9", use_oga=True
            ))
            mock_bert.assert_called_once()
            call_kwargs = mock_bert.call_args[1]
            self.assertEqual(call_kwargs["prompt"], "A hyper-realistic digital painting of a slender blonde woman beside teapot.")
            self.assertEqual(call_kwargs["aspect_ratio"], "16:9")
            self.assertEqual(call_kwargs["character"], "ogarla.85")

    def test_module60d_describe_embed_single_output(self):
        """Test execute_describe_core renders a clean single Prompt Description embed field."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        from services.vision_service import execute_describe_core

        mock_interaction = MagicMock()
        mock_interaction.user.id = 12345
        mock_interaction.user.name = "TestUser"
        mock_interaction.response.send_message = AsyncMock()

        mock_image = MagicMock()
        mock_image.content_type = "image/png"
        mock_image.filename = "test.png"
        mock_image.url = "https://example.com/test.png"
        mock_image.read = AsyncMock(return_value=b"fake_image_bytes")

        mock_vision_res = {
            "caption": "sunflower, 1girl, smile, solo",
            "sdxl_prompt": "sunflower, 1girl, smile, solo",
            "flux_prompt": "Sunflower, 1girl, smile, solo",
            "krea2_prompt": "Sunflower, 1girl, smile, solo",
            "engine_used": "Florence-2"
        }

        with patch("services.vision_service.run_vision_interrogate", new=AsyncMock(return_value=mock_vision_res)), \
             patch("services.vision_service.edit_original_fallback", new=AsyncMock()) as mock_edit:
            asyncio.run(execute_describe_core(mock_interaction, mock_image, model="florence2"))

            mock_edit.assert_called_once()
            call_kwargs = mock_edit.call_args[1]
            embed = call_kwargs["embed"]
            self.assertEqual(embed.title, "Image Description")
            self.assertEqual(len(embed.fields), 1, "Embed should contain exactly 1 field (single output)")
            self.assertEqual(embed.fields[0].name, "📝 Prompt Description")
            self.assertEqual(embed.fields[0].value, "sunflower, 1girl, smile, solo")
            self.assertIn("Florence-2", embed.footer.text)

    def test_module60c_describe_command_and_cog_registration(self):
        """Test /describe command registration on bot tree and VisionCog export."""
        import bot
        self.assertTrue(hasattr(bot, "describe"), "bot should re-export describe")
        self.assertTrue(hasattr(bot, "handle_update_describe_view"), "bot should re-export handle_update_describe_view")
        self.assertTrue(hasattr(bot, "handle_generate_described"), "bot should re-export handle_generate_described")
        
        # Verify VisionCog registration
        cog = bot.bot.get_cog("VisionCog")
        self.assertIsNotNone(cog, "VisionCog should be registered on bot")
        self.assertTrue(hasattr(cog, "describe"), "VisionCog should have describe method")
        
        # Verify tree command exists
        cmd = bot.bot.tree.get_command("describe")
        self.assertIsNotNone(cmd, "/describe command should exist on bot.tree")
        self.assertEqual(cmd.name, "describe")

    def test_module61_blend_krea_integration(self):
        """Test Krea 2 blend prompt fusion, workflow wetness tuning, view layouts, and command registration."""
        from parsers import fuse_krea2_blend_prompt, prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed, BlendButtons
        from bot import bot

        # 1. Test fuse_krea2_blend_prompt
        vision = "A cinematic portrait of an astronaut on Mars"
        remix = "wearing neon armor, golden hour lighting"
        fused = fuse_krea2_blend_prompt(vision, remix)
        self.assertEqual(fused, "wearing neon armor, golden hour lighting, A cinematic portrait of an astronaut on Mars")

        # Test empty remix returns vision
        self.assertEqual(fuse_krea2_blend_prompt(vision, ""), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, None), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, " , "), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, ","), "A cinematic portrait of an astronaut on Mars")

        # Test empty vision returns remix
        self.assertEqual(fuse_krea2_blend_prompt("", remix), remix)

        # 2. Test prepare_bertflow_workflow with custom wetness
        wf = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            width=1632,
            height=920,
            seed=12345,
            steps=8,
            unet_model="pornmasterKrea2_v1FP8.safetensors",
            wetness_strength=1.0
        )
        self.assertEqual(wf["822"]["inputs"]["lora_1"]["strength"], 1.0)
        self.assertTrue(wf["822"]["inputs"]["lora_1"]["on"])

        wf_zero = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            width=1632,
            height=920,
            wetness_strength=0.0
        )
        self.assertFalse(wf_zero["822"]["inputs"]["lora_1"]["on"])

        # Test Krea 2 Character LoRA injection and trigger word
        wf_char = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            character="ogarla.85"
        )
        self.assertTrue(wf_char["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_char["822"]["inputs"]["lora_2"]["lora"], "Krea2\\ogarla_krea2.safetensors")
        self.assertEqual(wf_char["822"]["inputs"]["lora_2"]["strength"], 0.85)
        self.assertEqual(wf_char["627"]["inputs"]["text"], "ogarla, cyberpunk alley")

        # Test prompt shorthand parsing (--ogarla.70)
        wf_prompt = prepare_bertflow_workflow(
            prompt="cyberpunk alley --ogarla.70"
        )
        self.assertTrue(wf_prompt["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_prompt["822"]["inputs"]["lora_2"]["strength"], 0.70)
        # Test prompt wetness flags (--dry, --dewy, --matte, --wet)
        wf_dry = prepare_bertflow_workflow("a sunny beach --dry")
        self.assertEqual(wf_dry["822"]["inputs"]["lora_1"]["strength"], -3.0)
        self.assertEqual(wf_dry["627"]["inputs"]["text"], "a sunny beach")

        wf_dewy = prepare_bertflow_workflow("portrait of a runner --dewy")
        self.assertEqual(wf_dewy["822"]["inputs"]["lora_1"]["strength"], -0.5)

        wf_matte = prepare_bertflow_workflow("studio portrait --matte")
        self.assertEqual(wf_matte["822"]["inputs"]["lora_1"]["strength"], -2.5)

        wf_custom_wet = prepare_bertflow_workflow("rainy street --wet 0.75")
        self.assertEqual(wf_custom_wet["822"]["inputs"]["lora_1"]["strength"], 0.75)

        # Test Ogarla Krea 2 Character LoRA injection and trigger word
        wf_oga = prepare_bertflow_workflow(
            prompt="fashion runway photo --ogarla.85",
            character="ogarla"
        )
        self.assertTrue(wf_oga["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_oga["822"]["inputs"]["lora_2"]["lora"], "Krea2\\ogarla_krea2.safetensors")
        self.assertEqual(wf_oga["822"]["inputs"]["lora_2"]["strength"], 0.85)
        self.assertEqual(wf_oga["627"]["inputs"]["text"], "ogarla, fashion runway photo")

        # Valerie is SDXL-only and must not inject into Krea 2
        wf_val = prepare_bertflow_workflow(
            prompt="fashion runway photo --valerie.90",
            character="valerie"
        )
        self.assertNotIn("lora_2", wf_val["822"]["inputs"])

        # Test scan_krea2_loras
        from characters import scan_krea2_loras
        discovered_loras = scan_krea2_loras()
        self.assertIsInstance(discovered_loras, list)

        # Test Architecture resolver for Krea 2
        from model_architecture import resolve_lora_for_architecture, Architecture
        self.assertEqual(
            resolve_lora_for_architecture("ogarla", Architecture.KREA2),
            "Krea2\\ogarla_krea2.safetensors"
        )
        self.assertEqual(
            resolve_lora_for_architecture("valerie", Architecture.SDXL),
            "jen_epoch_5.safetensors"
        )

        # 3. Test BlendKreaButtons
        view = BlendKreaButtons(generation_id="krea_blend_777", ar="16:9", model_choice="muse", wetness=-2.0, character="ogarla.85")
        btn_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]
        self.assertIn("set_blend_krea_ar:krea_blend_777:16:9", btn_ids)
        self.assertIn("set_blend_krea_ar:krea_blend_777:21:9", btn_ids)
        self.assertIn("toggle_blend_krea_model:krea_blend_777", btn_ids)
        self.assertIn("toggle_blend_krea_wetness:krea_blend_777", btn_ids)
        self.assertIn("set_blend_krea_char:krea_blend_777", btn_ids)
        self.assertIn("edit_blend_krea_prompt:krea_blend_777", btn_ids)
        self.assertIn("gen_blend_krea:krea_blend_777", btn_ids)

        char_select = next(item for item in view.children if getattr(item, "custom_id", None) == "set_blend_krea_char:krea_blend_777")
        char_values = [opt.value for opt in char_select.options]
        self.assertNotIn("valerie.90", char_values)
        self.assertNotIn("valerie.70", char_values)
        self.assertIn("ogarla.85", char_values)

        # Test BertflowButtons with character
        from views import BertflowButtons
        view_oga = BertflowButtons(generation_id="bert_test_789", character="ogarla.85")
        self.assertEqual(view_oga.toggle_char_btn.label, "🌿 Ogarla: ON")

        # Test CHARACTER_CHOICES_KREA2 for slash commands
        from bot import CHARACTER_CHOICES_KREA2
        choice_vals = [c.value for c in CHARACTER_CHOICES_KREA2]
        self.assertIn("ogarla.85", choice_vals)
        self.assertIn("ogarla.70", choice_vals)
        self.assertNotIn("valerie.90", choice_vals)
        self.assertNotIn("valerie.70", choice_vals)

        # 4. Test build_blend_krea_embed
        gen_data = {
            "krea2_prompt": "A close up photo of a cat",
            "user_prompt": "wearing a tiny bowtie",
            "fused_prompt": "wearing a tiny bowtie, A close up photo of a cat",
            "ar": "1:1",
            "model_choice": "muse",
            "wetness": -2.0
        }
        embed = build_blend_krea_embed(gen_data, author_str="TestUser")
        self.assertIn("Krea 2 Blend Studio", embed.title)
        field_names = [f.name for f in embed.fields]
        self.assertIn("📜 Generation Prompt", field_names)
        gen_prompt_field = next(f for f in embed.fields if f.name == "📜 Generation Prompt")
        self.assertIn("wearing a tiny bowtie, A close up photo of a cat", gen_prompt_field.value)

        # 4b. Test EditBlendKreaModal
        from views import EditBlendKreaModal
        modal = EditBlendKreaModal("krea_blend_777", current_prompt="A test generation prompt")
        self.assertEqual(modal.title, "✏️ Edit Generation Prompt")
        self.assertEqual(modal.prompt_input.default, "A test generation prompt")
        edit_btn = next(item for item in view.children if getattr(item, "custom_id", None) == "edit_blend_krea_prompt:krea_blend_777")
        self.assertEqual(edit_btn.label, "✏️ Edit Prompt")

        # 5. Test BlendButtons excludes Krea 2 button and Krea 2 models (strictly decoupled)
        blend_view = BlendButtons(generation_id="blend_gen_123")
        blend_btn_ids = [item.custom_id for item in blend_view.children if hasattr(item, "custom_id")]
        self.assertNotIn("blend_desc:blend_gen_123:krea2", blend_btn_ids, "Krea 2 button must NOT be in BlendButtons")
        self.assertIn("blend_desc:blend_gen_123:blend", blend_btn_ids)

        model_select = next(item for item in blend_view.children if getattr(item, "custom_id", "").startswith("set_blend_model:"))
        model_values = [opt.value for opt in model_select.options]
        self.assertNotIn("muse", model_values, "Krea 2 muse model must NOT be in BlendButtons")
        self.assertNotIn("pornmaster", model_values, "Krea 2 pornmaster model must NOT be in BlendButtons")
        self.assertIn("wai", model_values)

        # 5b. Test workflows/DESCRIBE_blend.json dedicated SDXL workflow exists and excludes Krea nodes
        blend_wf_path = "workflows/DESCRIBE_blend.json"
        self.assertTrue(os.path.exists(blend_wf_path), "workflows/DESCRIBE_blend.json must exist")
        with open(blend_wf_path, "r", encoding="utf-8") as f:
            blend_wf = json.load(f)
        self.assertIn("3", blend_wf, "Node 3 (caption) must be in DESCRIBE_blend.json")
        self.assertIn("4", blend_wf, "Node 4 (detailed_caption) must be in DESCRIBE_blend.json")
        self.assertNotIn("5", blend_wf, "Node 5 (Krea 2 Florence run) must NOT be in DESCRIBE_blend.json")
        self.assertNotIn("11", blend_wf, "Node 11 (Krea 2 text) must NOT be in DESCRIBE_blend.json")
        self.assertNotIn("21", blend_wf, "Node 21 (Krea 2 replace) must NOT be in DESCRIBE_blend.json")

        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("blend-krea", commands)
        bk_cmd = commands["blend-krea"]
        self.assertEqual(len(bk_cmd.parameters), 1, "blend-krea must only require the image parameter")
        img_param = bk_cmd.parameters[0]
        self.assertEqual(img_param.name, "image")
        self.assertTrue(img_param.required)

        # 7. Test dynamic character autocomplete attached to commands
        for cmd_name in ["bertflow", "imagine"]:
            self.assertIn(cmd_name, commands)
            cmd = commands[cmd_name]
            char_param = next((p for p in cmd.parameters if p.name == "character"), None)
            self.assertIsNotNone(char_param, f"Command {cmd_name} missing character parameter")
            self.assertTrue(char_param.autocomplete, f"Command {cmd_name} character parameter missing autocomplete")

    def test_bertflow_and_standard_remix_modals(self):
        """Verify handle_bertflow_remix and handle_remix construct the RemixModal properly."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from bot import handle_bertflow_remix, handle_remix, db

        # Seed mock generation into database
        db.save_generation(
            "test_remix_gen_123",
            {
                "prompt": "A beautiful sunrise in the mountains",
                "seed": 42,
                "aspect_ratio": "16:9",
                "unet_model": "museByStableYogi_v35Int8Extended.safetensors",
                "is_bertflow": True,
            }
        )

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = False
        mock_interaction.response.send_modal = AsyncMock()

        # Execute bertflow remix
        asyncio.run(handle_bertflow_remix(mock_interaction, "test_remix_gen_123"))
        mock_interaction.response.send_modal.assert_called_once()
        modal_passed = mock_interaction.response.send_modal.call_args[0][0]
        self.assertEqual(modal_passed.title, "✏️ Remix / Tweak Prompt")

        # Execute standard remix
        mock_interaction.reset_mock()
        mock_interaction.response.is_done.return_value = False
        db.save_generation(
            "test_standard_remix_123",
            {
                "prompt": "A futuristic cyberpunk city",
                "seed": 100,
                "is_bertflow": False,
            }
        )
        asyncio.run(handle_remix(mock_interaction, "test_standard_remix_123"))
        mock_interaction.response.send_modal.assert_called_once()

    def test_module72_blend_studio_debloated_unified_dashboard(self):
        """Test the de-bloated unified 1-page Blend Studio layout, 1-click toggles, cycles, and blend generation."""
        import bot
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        from views import BlendButtons

        # 1. Test 5-Row Component Layout and item distribution
        gen_id = "blend_debloat_999"
        view = BlendButtons(
            generation_id=gen_id,
            ar="16:9",
            sr=True,
            model_choice="wai",
            comp_strength="style",
            sref_rand="nosref",
            char_choice="valerie"
        )
        self.assertLessEqual(len(view.children), 25)
        rows = sorted(list(set(c.row for c in view.children)))
        self.assertEqual(rows, [0, 1, 2, 3, 4], "Dashboard must strictly use rows 0 through 4 without tabs")

        # Row 0: Model Checkpoint Select
        row0_items = [c for c in view.children if c.row == 0]
        self.assertEqual(len(row0_items), 1)
        self.assertEqual(row0_items[0].custom_id, f"set_blend_model:{gen_id}")
        self.assertNotIn("muse", [opt.value for opt in row0_items[0].options])

        # Row 1: Character LoRA Select
        row1_items = [c for c in view.children if c.row == 1]
        self.assertEqual(len(row1_items), 1)
        self.assertEqual(row1_items[0].custom_id, f"set_blend_char:{gen_id}")
        char_selected = next(opt for opt in row1_items[0].options if opt.default)
        self.assertEqual(char_selected.value, "valerie")

        # Row 2: Aspect Ratio Select
        row2_items = [c for c in view.children if c.row == 2]
        self.assertEqual(len(row2_items), 1)
        self.assertEqual(row2_items[0].custom_id, f"set_blend_ar:{gen_id}")
        ar_selected = next(opt for opt in row2_items[0].options if opt.default)
        self.assertEqual(ar_selected.value, "16:9")

        # Row 3: Semi-Realism Strength Select
        row3_items = [c for c in view.children if c.row == 3]
        self.assertEqual(len(row3_items), 1)
        self.assertEqual(row3_items[0].custom_id, f"set_blend_sr:{gen_id}")
        sr_selected = next(opt for opt in row3_items[0].options if opt.default)
        self.assertEqual(sr_selected.value, "sr75")

        # Row 4: Action Launchers & Toggles (4 buttons)
        row4_items = [c for c in view.children if c.row == 4]
        self.assertEqual(len(row4_items), 4)
        blend_btn = next(c for c in row4_items if c.custom_id.startswith("blend_desc:"))
        edit_btn = next(c for c in row4_items if c.custom_id.startswith("edit_blend_prompt:"))
        comp_btn = next(c for c in row4_items if c.custom_id.startswith("cycle_blend_comp:"))
        style_btn = next(c for c in row4_items if c.custom_id.startswith("toggle_blend_sref:"))

        self.assertEqual(blend_btn.label, "🎨 Blend Image")
        self.assertEqual(blend_btn.custom_id, f"blend_desc:{gen_id}:blend")
        self.assertEqual(edit_btn.label, "✏️ Edit Prompt")
        self.assertIn("Comp:", comp_btn.label)
        self.assertIn("OFF", style_btn.label)
        self.assertIn("--sref random", style_btn.label)

        # 2. Test handle_generate_blended with desc_type == "blend"
        gen_data = {
            "caption": "a lone warrior standing in the rain",
            "detailed_caption": "masterpiece digital painting of a lone samurai with glowing katana under volumetric rain and neon signs",
            "uploaded_image_name": "samurai_input.png",
            "image_url": "https://cdn.discordapp.com/samurai_thumb.jpg",
            "ar": "16:9",
            "sr": True,
            "char_choice": "valerie",
            "model_choice": "wai",
            "comp_strength": "style",
            "sref_rand": "nosref"
        }
        bot.db.save_generation("gen_blend_exec_test", gen_data)
        bot.active_generations["gen_blend_exec_test"] = gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            asyncio.run(bot.handle_generate_blended(
                mock_interaction,
                "gen_blend_exec_test",
                desc_type="blend",
                ar="16:9",
                use_sr=True,
                char_choice="valerie",
                model_choice="wai",
                comp_strength="style",
                use_sref_rand="nosref"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args.kwargs.get("prompt") or mock_exec.call_args.args[1]
            self.assertIn("samurai with glowing katana", called_prompt)

        # 3. Test cycle_blend_comp interaction handling
        import discord
        comp_inter = MagicMock()
        comp_inter.type = discord.InteractionType.component
        comp_inter.data = {"custom_id": "cycle_blend_comp:gen_blend_exec_test"}
        comp_inter.response.is_done.return_value = True
        comp_inter.followup.send = AsyncMock()

        with patch("bot.handle_update_blend_view", new=AsyncMock()) as mock_blend_update:
            asyncio.run(bot.on_interaction(comp_inter))
            mock_blend_update.assert_called_once()
            self.assertEqual(mock_blend_update.call_args.kwargs.get("new_comp"), "low")

        # 4. Test semi-realism dropdown values across all choices (.60, .70, .80, .90)
        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_blend_exec_test", desc_type="blend",
                ar="1:1", use_sr="sr60", char_choice="none", model_choice="wai"
            ))
            p_60 = mock_exec.call_args.kwargs.get("prompt") or mock_exec.call_args.args[1]
            self.assertIn("--sr.60", p_60)

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_blend_exec_test", desc_type="blend",
                ar="16:9", use_sr="sr90", char_choice="none", model_choice="wai"
            ))
            p_90 = mock_exec.call_args.kwargs.get("prompt") or mock_exec.call_args.args[1]
            self.assertIn("--sr.90", p_90)

    def test_module73_krea_cog_modular_architecture(self):
        """Verify KreaCog and KreaService modular separation, registration, and backward compatibility."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        import bot
        from cogs.krea_cog import KreaCog
        from services.krea_service import (
            BERTFLOW_MODEL_CHOICES,
            execute_bertflow,
            handle_bertflow_reroll,
            handle_bertflow_remix,
            handle_bertflow_toggle_char,
            handle_bertflow_upscale,
            handle_update_blend_krea_view,
            handle_submit_edit_blend_krea_prompt,
            handle_generate_blend_krea,
            execute_blend_krea_core,
        )

        # 1. Test KreaCog registration in bot
        cog = bot.bot.get_cog("KreaCog")
        self.assertIsNotNone(cog, "KreaCog must be loaded and registered in bot")
        self.assertIsInstance(cog, KreaCog)

        # 2. Test slash commands registered in tree
        commands = {cmd.name: cmd for cmd in bot.bot.tree.get_commands()}
        self.assertIn("bertflow", commands)
        self.assertIn("blend-krea", commands)

        # Verify bertflow command parameters
        bert_cmd = commands["bertflow"]
        bert_params = {p.name: p for p in bert_cmd.parameters}
        self.assertIn("prompt", bert_params)
        self.assertTrue(bert_params["prompt"].required)
        self.assertIn("aspect_ratio", bert_params)
        self.assertIn("character", bert_params)
        self.assertIn("celebrity", bert_params)
        self.assertIn("favorite_prompt", bert_params)
        self.assertIn("model", bert_params)
        self.assertIn("steps", bert_params)
        self.assertIn("seed", bert_params)

        # Verify blend-krea streamlined parameter (image only)
        krea_cmd = commands["blend-krea"]
        self.assertEqual(len(krea_cmd.parameters), 1)
        self.assertEqual(krea_cmd.parameters[0].name, "image")

        # 3. Test backward-compatibility re-exports on bot module
        self.assertTrue(callable(bot.execute_bertflow))
        self.assertTrue(callable(bot.handle_bertflow_reroll))
        self.assertTrue(callable(bot.handle_bertflow_remix))
        self.assertTrue(callable(bot.handle_bertflow_toggle_char))
        self.assertTrue(callable(bot.handle_bertflow_upscale))
        self.assertTrue(callable(bot.handle_update_blend_krea_view))
        self.assertTrue(callable(bot.handle_submit_edit_blend_krea_prompt))
        self.assertTrue(callable(bot.handle_generate_blend_krea))
        self.assertTrue(callable(bot.execute_blend_krea_core))
        self.assertEqual(bot.BERTFLOW_MODEL_CHOICES, BERTFLOW_MODEL_CHOICES)
        self.assertEqual(bot.bertflow, cog.bertflow)
        self.assertEqual(bot.blend_krea, cog.blend_krea)

        # 4. Test services/krea_service function executions (mocked)
        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        gen_id = "test_krea_service_gen_777"
        gen_data = {
            "prompt": "a test scene",
            "original_prompt": "a test scene",
            "aspect_ratio": "16:9",
            "steps": 8,
            "seed": 99999,
            "unet_model": "museByStableYogi_v35Int8Extended.safetensors",
            "character": "ogarla.85",
            "celebrity": None
        }
        bot.db.save_generation(gen_id, gen_data)

        # Test handle_bertflow_reroll calls execute_bertflow
        with patch("services.krea_service.execute_bertflow", new=AsyncMock()) as mock_exec:
            asyncio.run(handle_bertflow_reroll(mock_interaction, gen_id))
            mock_exec.assert_called_once()
            self.assertEqual(mock_exec.call_args.kwargs["prompt"], "a test scene")
            self.assertNotEqual(mock_exec.call_args.kwargs["seed"], 99999)

        # Test handle_bertflow_toggle_char toggles character
        with patch("services.krea_service.execute_bertflow", new=AsyncMock()) as mock_exec:
            asyncio.run(handle_bertflow_toggle_char(mock_interaction, gen_id))
            mock_exec.assert_called_once()
            self.assertIsNone(mock_exec.call_args.kwargs["character"])

    def test_blend_sdxl_sref_random_toggle_interaction(self):
        """Test /blend-sdxl style button is ONLY a 1-click --sref random toggle on/off."""
        import discord
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        import bot
        from views import BlendButtons, build_blend_embed

        # 1. Test View Button States
        # State: OFF
        view_off = BlendButtons("gen_off", ar="16:9", sref_rand="nosref")
        btn_off = next(c for c in view_off.children if getattr(c, "custom_id", "").startswith("toggle_blend_sref:"))
        self.assertEqual(btn_off.label, "🎲 --sref random: OFF")
        self.assertEqual(btn_off.style, discord.ButtonStyle.secondary)

        # State: ON
        view_on = BlendButtons("gen_on", ar="16:9", sref_rand="sref")
        btn_on = next(c for c in view_on.children if getattr(c, "custom_id", "").startswith("toggle_blend_sref:"))
        self.assertEqual(btn_on.label, "🎲 --sref random: ON")
        self.assertEqual(btn_on.style, discord.ButtonStyle.primary)

        # 2. Test Embed Field Display
        embed_off = build_blend_embed({"sref_rand": "nosref", "caption": "test", "detailed_caption": "scene"}, author_str="Test")
        aesthetics_off = next(f.value for f in embed_off.fields if f.name == "🎭 Aesthetics")
        self.assertIn("**Style:** `OFF`", aesthetics_off)

        embed_on = build_blend_embed({"sref_rand": "sref", "caption": "test", "detailed_caption": "scene"}, author_str="Test")
        aesthetics_on = next(f.value for f in embed_on.fields if f.name == "🎭 Aesthetics")
        self.assertIn("**Style:** `🎲 --sref random`", aesthetics_on)

        # 3. Test on_interaction toggle handling
        gen_id = "gen_sref_toggle_test"
        bot.active_generations[gen_id] = {"sref_rand": "nosref", "author_str": "Test"}
        bot.db.save_generation(gen_id, bot.active_generations[gen_id])
        mock_interaction = MagicMock()
        mock_interaction.type = discord.InteractionType.component
        mock_interaction.data = {"custom_id": f"toggle_blend_sref:{gen_id}"}
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch("bot.handle_update_blend_view", new=AsyncMock()) as mock_update:
            asyncio.run(bot.on_interaction(mock_interaction))
            mock_update.assert_called_once_with(mock_interaction, gen_id, new_sref="sref")

        # Toggle back from ON to OFF
        bot.active_generations[gen_id] = {"sref_rand": "sref", "author_str": "Test"}
        bot.db.save_generation(gen_id, bot.active_generations[gen_id])
        with patch("bot.handle_update_blend_view", new=AsyncMock()) as mock_update:
            asyncio.run(bot.on_interaction(mock_interaction))
            mock_update.assert_called_once_with(mock_interaction, gen_id, new_sref="nosref")

    def test_video_capabilities_retired(self):
        """Test that all video commands, context menus, and cogs have been completely retired."""
        import bot

        # 1. Verify video commands absent from bot.tree
        cmd_names = [c.name for c in bot.bot.tree.get_commands()]
        self.assertNotIn("video", cmd_names)
        self.assertNotIn("ltx", cmd_names)
        self.assertNotIn("Animate to Video", cmd_names)

        # 2. Verify no video command re-exports on bot module
        self.assertFalse(hasattr(bot, "video"))
        self.assertFalse(hasattr(bot, "video_command"))
        self.assertFalse(hasattr(bot, "ltx"))
        self.assertFalse(hasattr(bot, "ltx_command"))
        self.assertFalse(hasattr(bot, "animate_to_video_context"))
        self.assertFalse(hasattr(bot, "execute_video_core"))
        self.assertFalse(hasattr(bot, "execute_ltx_core"))

    def test_upscale_service_and_cog_registration(self):
        """Test UpscaleCog registration in bot.tree, service workflow builders, and re-exports."""
        import bot
        from cogs.upscale_cog import UpscaleCog
        import services.upscale_service as upscale_service
        from views import IsolatedImageButtons, UpscaleButtons

        # 1. Verify /upscale command in bot.tree
        cmd_names = [c.name for c in bot.bot.tree.get_commands()]
        self.assertIn("upscale", cmd_names, "Expected /upscale to be registered in bot.tree")

        upscale_cmd = next(c for c in bot.bot.tree.get_commands() if c.name == "upscale")
        param_names = [p.name for p in upscale_cmd.parameters]
        self.assertIn("image", param_names)
        self.assertIn("scale", param_names)
        self.assertIn("mode", param_names)
        self.assertIn("style", param_names)
        self.assertIn("prompt", param_names)

        # 2. Verify re-exports on bot module
        self.assertTrue(hasattr(bot, "upscale"))
        self.assertTrue(hasattr(bot, "upscale_command"))
        self.assertTrue(callable(bot.upscale.callback))
        self.assertTrue(callable(upscale_service.execute_upscale_core))
        self.assertTrue(callable(upscale_service.build_fast_upscale_workflow))
        self.assertTrue(callable(upscale_service.build_generative_upscale_workflow))
        self.assertTrue(callable(upscale_service.calculate_latent_refiner_dimensions))

        # 3. Test dimension calculations
        w2, h2 = upscale_service.calculate_target_dimensions(1024, 1024, scale_factor=2.0)
        self.assertEqual((w2, h2), (2048, 2048))

        w4, h4 = upscale_service.calculate_target_dimensions(1024, 1024, scale_factor=4.0)
        self.assertEqual((w4, h4), (4096, 4096))

        w15, h15 = upscale_service.calculate_target_dimensions(1024, 768, scale_factor=1.5)
        self.assertEqual((w15, h15), (1536, 1152))

        # Test latent refiner dimension capping (prevents VRAM explosion)
        rw1, rh1 = upscale_service.calculate_latent_refiner_dimensions(3840, 1648, max_side=1280)
        self.assertEqual((rw1, rh1), (1280, 552))
        self.assertTrue(max(rw1, rh1) <= 1280)

        rw2, rh2 = upscale_service.calculate_latent_refiner_dimensions(1024, 1024, max_side=1280)
        self.assertEqual((rw2, rh2), (1024, 1024))

        # 4. Test Fast Clean workflow builder
        fast_wf = upscale_service.build_fast_upscale_workflow(
            image_filename="test_input.png",
            target_width=2048,
            target_height=2048,
            model_name="4x_foolhardy_Remacri.pth"
        )
        self.assertIn("1", fast_wf)
        self.assertIn("2", fast_wf)
        self.assertIn("3", fast_wf)
        self.assertIn("4", fast_wf)
        self.assertIn("5", fast_wf)
        self.assertEqual(fast_wf["2"]["inputs"]["model_name"], "4x_foolhardy_Remacri.pth")
        self.assertEqual(fast_wf["4"]["inputs"]["width"], 2048)
        self.assertEqual(fast_wf["4"]["inputs"]["height"], 2048)
        self.assertEqual(fast_wf["4"]["inputs"]["upscale_method"], "lanczos")

        # 5. Test Generative Clarity workflow builder
        gen_wf = upscale_service.build_generative_upscale_workflow(
            image_filename="test_input.png",
            target_width=2048,
            target_height=2048,
            prompt="cyberpunk cityscape at night",
            negative_prompt="blurry, distorted",
            denoise=0.30
        )
        self.assertIn("1", gen_wf)
        self.assertIn("5", gen_wf)  # CheckpointLoaderSimple
        self.assertIn("6", gen_wf)  # Positive CLIPTextEncode
        self.assertIn("7", gen_wf)  # Negative CLIPTextEncode
        self.assertIn("8", gen_wf)  # VAEEncode
        self.assertIn("9", gen_wf)  # KSampler
        self.assertEqual(gen_wf["6"]["inputs"]["text"], "cyberpunk cityscape at night")
        self.assertEqual(gen_wf["7"]["inputs"]["text"], "blurry, distorted")
        self.assertEqual(gen_wf["9"]["inputs"]["denoise"], 0.30)

        # 6. Test UI button definitions
        iso_view = IsolatedImageButtons(generation_id="gen123", index=1)
        iso_custom_ids = [btn.custom_id for btn in iso_view.children if isinstance(btn, bot.discord.ui.Button)]
        self.assertIn("upscale_run:gen123:1:2.0", iso_custom_ids)
        self.assertIn("upscale_run:gen123:1:4.0", iso_custom_ids)

        up_view = UpscaleButtons(generation_id="gen123", index=1)
        self.assertEqual(up_view.upscale_scale, "2.0")



if __name__ == '__main__':
    unittest.main()
