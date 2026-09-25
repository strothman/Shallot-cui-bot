"""
Comprehensive Button Audit & Interaction Dispatch Test Suite for Shallot-CUI Bot.
Verifies that all 50+ interactive component button, dropdown, and modal triggers
resolve their handlers without NameError, AttributeError, or unhandled exceptions.
"""

import unittest
import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch
import discord

import db
from services.interaction_dispatcher import dispatch_interaction
from views.dynamic_items import (
    CancelGenDynamicButton,
    IsolateDynamicButton,
    VariationDynamicButton,
    RerollDynamicButton,
    RemixDynamicButton,
    OutpaintDynamicButton,
)


class TestButtonAudit(unittest.TestCase):
    def setUp(self):
        self.gen_id = "test_audit_gen_123"
        self.gen_data = {
            "prompt": "a beautiful fantasy landscape, high resolution",
            "original_prompt": "a beautiful fantasy landscape, high resolution",
            "negative_prompt": "blurry, low quality",
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "checkpoint": "waiIllustriousSDXL_v170.safetensors",
            "model_choice": "wai",
            "caption": "fantasy landscape",
            "detailed_caption": "a detailed fantasy landscape with misty mountains",
            "uploaded_image_name": "test_img.png",
            "image_url": "https://example.com/test.png",
            "ar": "16:9",
            "sr": "sr75",
            "oga": False,
            "char_choice": "none",
            "comp_strength": "style",
            "sref_rand": "nosref",
            "author_str": "testuser",
            "steps": 8,
            "wetness": -2.0,
            "composition": "off",
            "cref_weight": 0.20,
            "semi_realism_weight": 0.85,
            "semi_realism": True,
            "random_sref": False,
            "status": "completed",
        }
        db.save_generation(self.gen_id, self.gen_data)

    def _make_interaction(self, custom_id: str, values: list = None):
        inter = MagicMock(spec=discord.Interaction)
        inter.type = discord.InteractionType.component
        inter.data = {"custom_id": custom_id}
        if values is not None:
            inter.data["values"] = values
        inter.response = MagicMock()
        inter.response.is_done.return_value = False
        inter.response.defer = AsyncMock()
        inter.response.send_message = AsyncMock()
        inter.response.edit_message = AsyncMock()
        inter.response.send_modal = AsyncMock()
        inter.followup = MagicMock()
        inter.followup.send = AsyncMock(return_value=MagicMock(id=999))
        inter.message = MagicMock(id=888)
        inter.user = MagicMock()
        inter.user.id = 12345
        inter.user.name = "testuser"
        inter.user.display_name = "testuser"
        inter.user.mention = "<@12345>"
        return inter

    def test_dynamic_buttons_callbacks(self):
        """Verify all 6 persistent dynamic button classes execute callbacks without crashing."""
        from services.grid_actions_service import comfy_client
        inter = self._make_interaction(f"cancel_gen:{self.gen_id}")

        from PIL import Image
        bio = io.BytesIO()
        Image.new("RGB", (64, 64), color="red").save(bio, "PNG")
        valid_png = bio.getvalue()

        with patch.object(comfy_client, "pause_generation", new=AsyncMock(return_value=True)), \
             patch("services.grid_actions_service.send_followup_fallback", new=AsyncMock()), \
             patch("services.grid_actions_service.save_quadrant_images_async", new=AsyncMock()), \
             patch("services.grid_actions_service.edit_message_fallback", new=AsyncMock()):
            
            # 1. Cancel
            cancel_btn = CancelGenDynamicButton(self.gen_id)
            asyncio.run(cancel_btn.callback(inter))

            # 2. Isolate
            with patch("services.grid_actions_service.get_quadrant_bytes_async", new=AsyncMock(return_value=valid_png)), \
                 patch("services.grid_actions_service.embed_metadata_async", new=AsyncMock(return_value=io.BytesIO(valid_png))):
                iso_btn = IsolateDynamicButton(self.gen_id, 1)
                asyncio.run(iso_btn.callback(inter))

            # 3. Variation
            with patch.object(comfy_client, "generate", new=AsyncMock(return_value=[valid_png])), \
                 patch("services.grid_actions_service.complete_grid_generation", new=AsyncMock()):
                var_btn = VariationDynamicButton(self.gen_id, 1)
                asyncio.run(var_btn.callback(inter))

            # 4. Reroll
            with patch.object(comfy_client, "generate", new=AsyncMock(return_value=[valid_png])), \
                 patch("services.grid_actions_service.complete_grid_generation", new=AsyncMock()):
                reroll_btn = RerollDynamicButton(self.gen_id)
                asyncio.run(reroll_btn.callback(inter))

            # 5. Remix
            remix_btn = RemixDynamicButton(self.gen_id)
            asyncio.run(remix_btn.callback(inter))
            inter.response.send_modal.assert_called()

            # 6. Outpaint
            with patch("services.grid_actions_service.get_quadrant_bytes_async", new=AsyncMock(return_value=valid_png)), \
                 patch.object(comfy_client, "generate", new=AsyncMock(return_value=[valid_png])), \
                 patch.object(comfy_client, "upload_image", new=AsyncMock(return_value={"name": "pad.png"})), \
                 patch("services.grid_actions_service.complete_grid_generation", new=AsyncMock()):
                outpaint_btn = OutpaintDynamicButton(self.gen_id, 1, "right")
                asyncio.run(outpaint_btn.callback(inter))

    def test_all_dispatcher_custom_ids(self):
        """Verify all custom ID patterns handled by dispatch_interaction execute without error."""
        from services.generation_service import comfy_client
        from PIL import Image
        bio = io.BytesIO()
        Image.new("RGB", (64, 64), color="blue").save(bio, "PNG")
        valid_png = bio.getvalue()

        patches = [
            patch("comfy_client.ComfyClient.generate", new=AsyncMock(return_value=[valid_png])),
            patch("comfy_client.ComfyClient.upload_image", new=AsyncMock(return_value={"name": "up.png"})),
            patch("comfy_client.ComfyClient.pause_generation", new=AsyncMock(return_value=True)),
            patch("services.grid_actions_service.get_quadrant_bytes_async", new=AsyncMock(return_value=valid_png)),
            patch("services.grid_actions_service.embed_metadata_async", new=AsyncMock(return_value=io.BytesIO(valid_png))),
            patch("services.grid_actions_service.complete_grid_generation", new=AsyncMock()),
            patch("services.grid_actions_service.save_quadrant_images_async", new=AsyncMock()),
            patch("services.grid_actions_service.edit_message_fallback", new=AsyncMock()),
            patch("services.grid_actions_service.send_followup_fallback", new=AsyncMock()),
            patch("services.generation_service.save_quadrant_images_async", new=AsyncMock()),
            patch("services.blend_generation_service.save_quadrant_images_async", new=AsyncMock()),
            patch("services.blend_generation_service.create_grid_async", new=AsyncMock(return_value=io.BytesIO(valid_png))),
            patch("services.blend_generation_service.edit_message_fallback", new=AsyncMock()),
            patch("services.generation_service.execute_imagine", new=AsyncMock()),
            patch("services.krea_service.execute_bertflow", new=AsyncMock()),
            patch("core_helpers.download_image", new=AsyncMock(return_value=valid_png)),
            patch("aiohttp.ClientSession.get")
        ]

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], \
             patches[12], patches[13], patches[14], patches[15], patches[16] as mock_http_get:

            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=valid_png)
            mock_http_get.return_value.__aenter__.return_value = mock_resp

            custom_ids_to_test = [
                # Stasis
                f"stasis_pause:{self.gen_id}:12345",
                f"stasis_resume:{self.gen_id}:12345",
                # Core Grid Actions
                f"cancel_gen:{self.gen_id}",
                f"remix:{self.gen_id}",
                f"reroll:{self.gen_id}",
                f"upscale:{self.gen_id}:1",
                f"variation:{self.gen_id}:1",
                f"vary_subtle:{self.gen_id}:1",
                f"vary_strong:{self.gen_id}:1",
                # Bertflow Buttons
                f"bertflow_reroll:{self.gen_id}",
                f"bertflow_remix:{self.gen_id}",
                f"bertflow_toggle_char:{self.gen_id}",
                f"bertflow_upscale:{self.gen_id}",
                # Favorites & Copy
                f"fav_style:{self.gen_id}",
                f"fav_prompt:{self.gen_id}",
                f"copy_prompt:{self.gen_id}",
                # Upscale & Outpaint
                f"upscale_run:{self.gen_id}:1:2.0",
                f"upscale_redo:{self.gen_id}:1:2.0",
                f"outpaint:{self.gen_id}:1:16:9",
                f"outpaint:{self.gen_id}:1:right",
                # SREF Management
                f"sref_change_random:{self.gen_id}:1",
                f"sref_change_saved:{self.gen_id}:1",
                f"sref_change_custom:{self.gen_id}:1",
                # Describe Controls
                f"gen_desc:{self.gen_id}:caption",
                f"gen_desc:{self.gen_id}:detailed",
                f"set_desc_ar:{self.gen_id}:16:9:sr75:oga:hyphoria",
                # Blend Krea Controls
                f"set_blend_krea_ar:{self.gen_id}:16:9",
                f"toggle_blend_krea_model:{self.gen_id}",
                f"toggle_blend_krea_steps:{self.gen_id}",
                f"toggle_blend_krea_wetness:{self.gen_id}",
                f"toggle_blend_krea_composition:{self.gen_id}",
                f"edit_blend_krea_prompt:{self.gen_id}",
                f"gen_blend_krea:{self.gen_id}",
                # Blend SDXL Controls
                f"edit_blend_prompt:{self.gen_id}",
                f"cycle_blend_ar:{self.gen_id}",
                f"cycle_blend_comp:{self.gen_id}",
                f"toggle_blend_sref:{self.gen_id}",
                f"switch_blend_tab:{self.gen_id}:canvas",
                f"toggle_blend_sr:{self.gen_id}:sr75",
                f"toggle_blend_oga:{self.gen_id}:oga",
                f"blend_desc:{self.gen_id}:blend",
                f"reblend:{self.gen_id}",
                # Adopt Actions
                f"adopt_imagine:{self.gen_id}",
                f"adopt_toggle_oga:{self.gen_id}",
                f"adopt_toggle_sr:{self.gen_id}",
                f"adopt_toggle_sref:{self.gen_id}",
                f"adopt_cycle_cw:{self.gen_id}",
                f"adopt_edit_prompt:{self.gen_id}",
                f"adopt_copy:{self.gen_id}",
                f"adopt_save:{self.gen_id}",
            ]

            dropdown_items_to_test = [
                (f"set_blend_krea_comp:{self.gen_id}", ["medium"]),
                (f"set_blend_krea_char:{self.gen_id}", ["ogarla"]),
                (f"set_blend_krea_celeb:{self.gen_id}", ["none"]),
                (f"set_blend_char:{self.gen_id}", ["mageill_e4"]),
                (f"set_blend_sr:{self.gen_id}", ["sr80"]),
                (f"set_blend_style:{self.gen_id}", ["nosref"]),
                (f"set_blend_ar:{self.gen_id}", ["3:5"]),
                (f"set_blend_model:{self.gen_id}", ["wai"]),
                (f"set_blend_comp:{self.gen_id}", ["low"]),
            ]

            async def _run_all_tests():
                for cid in custom_ids_to_test:
                    inter = self._make_interaction(cid)
                    result = await dispatch_interaction(inter)
                    self.assertTrue(result, f"Failed to dispatch custom_id: {cid}")

                for cid, vals in dropdown_items_to_test:
                    inter = self._make_interaction(cid, values=vals)
                    result = await dispatch_interaction(inter)
                    self.assertTrue(result, f"Failed to dispatch dropdown custom_id: {cid} with values {vals}")

            asyncio.run(_run_all_tests())


if __name__ == "__main__":
    unittest.main()
