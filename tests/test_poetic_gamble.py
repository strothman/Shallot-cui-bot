"""
Automated Unit Tests for the Elevated Poetic Gamble Studio.
Tests prompt synthesis, dual-engine formatting (SDXL vs Krea 2), flag parsing, UI views, and Cog registration.
"""

import unittest
import discord
from unittest.mock import MagicMock, AsyncMock

from parsers.poetic import (
    PoeticMood,
    MOOD_DISPLAY_NAMES,
    PoeticPromptResult,
    synthesize_poetic_prompt,
)
from parsers.prompts import (
    RE_GAMBLE,
    parse_gamble_prompt,
)
from views.gamble_views import (
    GambleButtons,
    build_gamble_embed,
)
from cogs.gamble_cog import GambleCog


class TestPoeticGambleSynthesis(unittest.TestCase):
    """Tests for the poetic allegory synthesizer."""

    def test_explicit_seed_word(self):
        res = synthesize_poetic_prompt(seed="solitude", mood=PoeticMood.ETHEREAL, engine="sdxl", rng_seed=42)
        self.assertEqual(res.seed_word, "solitude")
        self.assertEqual(res.mood, PoeticMood.ETHEREAL)
        self.assertEqual(res.engine, "sdxl")
        self.assertIn("solitude", res.prompt)
        self.assertIn("solitude", res.stanza)

    def test_wildcard_seed_when_empty(self):
        res = synthesize_poetic_prompt(seed="", mood=PoeticMood.WILD, rng_seed=123)
        self.assertTrue(len(res.seed_word) > 0)
        self.assertIn(res.seed_word, res.prompt)

    def test_all_mood_archetypes(self):
        for mood in [PoeticMood.ETHEREAL, PoeticMood.GOTHIC, PoeticMood.NOIR, PoeticMood.MYTHIC, PoeticMood.SURREAL]:
            res = synthesize_poetic_prompt(seed="memory", mood=mood, engine="sdxl", rng_seed=100)
            self.assertEqual(res.mood, mood)
            self.assertTrue(len(res.metaphor) > 0)
            self.assertTrue(len(res.setting) > 0)
            self.assertTrue(len(res.color_palette) > 0)

    def test_dual_engine_formatting(self):
        # SDXL should have painterly fine art descriptors
        res_sdxl = synthesize_poetic_prompt(seed="ember", engine="sdxl", rng_seed=77)
        self.assertEqual(res_sdxl.engine, "sdxl")
        self.assertIn("fine art", res_sdxl.prompt.lower())
        self.assertIn("painterly", res_sdxl.prompt.lower())

        # Krea 2 should have 35mm film photograph realism descriptors
        res_krea = synthesize_poetic_prompt(seed="ember", engine="krea2", rng_seed=77)
        self.assertEqual(res_krea.engine, "krea2")
        self.assertIn("35mm film photograph", res_krea.prompt.lower())
        self.assertIn("tactile textures", res_krea.prompt.lower())

    def test_no_meta_prompt_pollution(self):
        """Ensures old broken CLIP tokens from ComfyUI workflows are strictly avoided."""
        res = synthesize_poetic_prompt(seed="whisper", rng_seed=999)
        prompt_lower = res.prompt.lower()
        self.assertNotIn("interpret the one word seed", prompt_lower)
        self.assertNotIn("one word origin", prompt_lower)
        self.assertNotIn("dictionary definition", prompt_lower)
        self.assertNotIn("abstraction level:", prompt_lower)
        self.assertNotIn("primary metaphor:", prompt_lower)

    def test_two_line_stanza(self):
        res = synthesize_poetic_prompt(seed="tide", rng_seed=50)
        lines = res.stanza.strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("*tide*", lines[0])


class TestGamblePromptParser(unittest.TestCase):
    """Tests for --gamble and --poetic flag parsing in prompt strings."""

    def test_parse_gamble_flag(self):
        raw = "a lonely pier at night --gamble"
        prompt, is_gamble, result = parse_gamble_prompt(raw, engine="sdxl")
        self.assertTrue(is_gamble)
        self.assertIsNotNone(result)
        self.assertNotIn("--gamble", prompt)
        self.assertIn("allegory", prompt.lower())

    def test_parse_poetic_flag(self):
        raw = "whisper --poetic"
        prompt, is_gamble, result = parse_gamble_prompt(raw, engine="krea2")
        self.assertTrue(is_gamble)
        self.assertIsNotNone(result)
        self.assertEqual(result.seed_word, "whisper")
        self.assertEqual(result.engine, "krea2")

    def test_normal_prompt_unaffected(self):
        raw = "a cozy cottage in the woods --ar 16:9"
        prompt, is_gamble, result = parse_gamble_prompt(raw)
        self.assertFalse(is_gamble)
        self.assertIsNone(result)
        self.assertEqual(prompt, raw)


class TestGambleUIAndEmbed(unittest.TestCase):
    """Tests for GambleButtons view and build_gamble_embed."""

    def test_sdxl_buttons_structure(self):
        view = GambleButtons("test_gen_123", engine="sdxl", mood="gothic")
        custom_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]

        # Should have U1-U4, quick upscale, reroll, mood, switch to krea2, remix
        self.assertIn("upscale:test_gen_123:1", custom_ids)
        self.assertIn("upscale:test_gen_123:4", custom_ids)
        self.assertIn("gamble_reroll:test_gen_123", custom_ids)
        self.assertIn("gamble_mood:test_gen_123", custom_ids)
        self.assertIn("gamble_switch:test_gen_123:krea2", custom_ids)

        # Ensure no row exceeds 5 items (Discord API constraint)
        row_counts = {}
        for item in view.children:
            r = getattr(item, "row", 0)
            row_counts[r] = row_counts.get(r, 0) + 1
        for r, count in row_counts.items():
            self.assertLessEqual(count, 5, f"Row {r} exceeded Discord's 5-item limit!")

    def test_krea2_buttons_structure(self):
        view = GambleButtons("test_krea_456", engine="krea2", mood="ethereal")
        custom_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]

        self.assertIn("gamble_reroll:test_krea_456", custom_ids)
        self.assertIn("gamble_mood:test_krea_456", custom_ids)
        self.assertIn("gamble_switch:test_krea_456:sdxl", custom_ids)
        self.assertIn("remix:test_krea_456", custom_ids)

    def test_build_gamble_embed(self):
        gamble_info = {
            "seed_word": "solitude",
            "mood": "gothic",
            "stanza": "Line 1\nLine 2",
            "color_palette": "black-green & gold"
        }
        embed = build_gamble_embed(
            gamble_info=gamble_info,
            seed=42,
            width=1024,
            height=1024,
            model_name="sdxl_base.safetensors",
            user_name="TestUser",
            user_id=12345,
            engine="sdxl"
        )
        self.assertIn("Solitude", embed.title)
        self.assertIn("Gothic", str(embed.fields))
        self.assertIn("SDXL", str(embed.fields))
        self.assertIn("Line 1", embed.description)


class TestGambleCog(unittest.TestCase):
    """Tests for GambleCog registration and commands."""

    def test_cog_initialization(self):
        bot_mock = MagicMock()
        cog = GambleCog(bot_mock)
        self.assertEqual(cog.bot, bot_mock)
        # Check command exists
        cmd = getattr(cog, "gamble", None)
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "gamble")


class TestGambleExecutionService(unittest.IsolatedAsyncioTestCase):
    """Tests for execute_bertflow and gamble_service execution routing."""

    async def test_bertflow_with_gamble_info_completion(self):
        from services.krea_service import execute_bertflow

        interaction_mock = MagicMock()
        interaction_mock.user.display_name = "Tester"
        interaction_mock.user.id = 999
        interaction_mock.user.display_avatar.url = "http://example.com/avatar.png"
        interaction_mock.followup.send = AsyncMock()
        status_msg_mock = AsyncMock()
        status_msg_ref = [status_msg_mock]

        client_mock = AsyncMock()
        client_mock.free_memory = AsyncMock()
        client_mock.generate = AsyncMock(return_value=[b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest"])
        client_mock.get_execution_timing = MagicMock(return_value={"sampling_duration": 1.5, "init_duration": 0.5})

        gamble_info = {
            "seed_word": "ember",
            "mood": "noir",
            "stanza": "A poetic stanza line 1\nLine 2",
            "color_palette": "charcoal & neon"
        }

        # Should complete cleanly without raising NameError for 'file'
        await execute_bertflow(
            interaction=interaction_mock,
            prompt="A poetic allegory of ember",
            status_msg_ref=status_msg_ref,
            client=client_mock,
            gamble_info=gamble_info
        )

        self.assertTrue(status_msg_mock.edit.await_count >= 1)
        call_kwargs = status_msg_mock.edit.call_args.kwargs
        self.assertIn("attachments", call_kwargs)
        self.assertIn("view", call_kwargs)
        self.assertIsInstance(call_kwargs["view"], GambleButtons)
        # Verify gamble is included in the output filename
        attachment_file = call_kwargs["attachments"][0]
        self.assertTrue(attachment_file.filename.startswith("gamble_krea_"))


class TestGambleIsolationDispatch(unittest.IsolatedAsyncioTestCase):
    """Verifies that U1-U4 quadrant isolation buttons (both legacy isolate: and standard upscale:) route properly."""

    async def test_isolate_and_upscale_dispatch_routes(self):
        from services.interaction_dispatcher import dispatch_interaction
        from unittest.mock import patch, AsyncMock

        for prefix in ["isolate", "upscale"]:
            custom_id = f"{prefix}:test_gen_999:4"
            interaction = AsyncMock()
            interaction.type = discord.InteractionType.component
            interaction.data = {"custom_id": custom_id}

            mock_handler = AsyncMock()
            with patch("services.interaction_dispatcher._resolve_handler", return_value=mock_handler):
                handled = await dispatch_interaction(interaction)
                self.assertTrue(handled, f"Dispatcher failed to route custom_id: {custom_id}")
                mock_handler.assert_awaited_once_with(interaction, "test_gen_999", 4)


if __name__ == "__main__":
    unittest.main()
