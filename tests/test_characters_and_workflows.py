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


class TestCharactersAndWorkflows(unittest.TestCase):
    def test_module5_blend_workflow_construction(self):
        """Test dynamic construction of IPAdapter multi-image workflow."""
        filenames = ["img1.png", "img2.png", "img3.png"]
        wf = build_blend_workflow(filenames, "masterpiece blend", "blurry", "v1-5", 512, 512, 999, 4.0)

        # Verify IPAdapter load nodes (blend_img_0, blend_img_1, blend_img_2) and IPAdapterAdvanced nodes
        self.assertIn("blend_img_0", wf)
        self.assertIn("blend_ip_0", wf)
        self.assertIn("blend_img_1", wf)
        self.assertIn("blend_ip_1", wf)
        self.assertIn("blend_img_2", wf)
        self.assertIn("blend_ip_2", wf)
        self.assertEqual(wf["blend_img_0"]["inputs"]["image"], "img1.png")
        self.assertEqual(wf["blend_ip_0"]["inputs"]["end_at"], 0.85)

        # Test negative prompt sketch avoidance when positive prompt contains sketch keywords
        wf_sketch = build_blend_workflow(["img1.png"], "pencil sketch drawing of girl", "bad quality", "v1-5", 512, 512, 999, 3.5)
        self.assertNotIn("line art only, sketch", wf_sketch["7"]["inputs"]["text"])

        # Test img2img composition scaling
        img2img_template = {
            "3": {"inputs": {}},
            "4": {"inputs": {}},
            "6": {"inputs": {}},
            "7": {"inputs": {}},
            "9": {"inputs": {}, "class_type": "SaveImage"},
            "30": {"inputs": {}},
            "31": {"inputs": {}}
        }
        wf_comp = build_blend_workflow(["img1.png"], "blended image", "blurry", "v1-5", 512, 512, 999, 3.5, workflow_template=img2img_template)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["weight"], 0.35)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["end_at"], 0.75)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["weight_type"], "ease in-out")

        # Test pure img2img composition retention (med comp decouples IPAdapter)
        wf_pure_img2img = build_blend_workflow(["img1.png"], "blended image", "blurry", "v1-5", 512, 512, 999, 3.5, workflow_template=img2img_template, comp_strength="med")
        self.assertNotIn("blend_ip_0", wf_pure_img2img, "Pure img2img composition mode should not strangle the latent with IPAdapter")

    def test_module5b_blend_style_calibration_and_checkpoint_configs(self):
        """Test Style Only IP-Adapter gentle weighting, checkpoint config enforcement, and negative prompt sanitization."""
        from parsers import parse_loras

        # 1. Test that --sr does NOT corrupt --sref
        test_prompt = "Semi-realism, girl kneeling on red carpet --sr.75 --ogarla.70 --ar 16:9 --sref 570053"
        cleaned_p, loras = parse_loras(test_prompt)
        self.assertIn("--sref 570053", cleaned_p, "--sref flag should remain completely intact")
        self.assertNotIn(" ef 570053", cleaned_p, "--sref should not be mutilated into isolated ef")
        self.assertEqual(len(loras), 2)

        # 2. Test Style Only IP-Adapter weighting & ease-out
        wf_style = build_blend_workflow(
            ["test_blend.png"],
            "Semi-realism, masterpiece, best quality. ogarla, girl on red carpet --sr.75",
            "low quality, blurry, photorealistic, anime",
            "waiIllustriousSDXL_v170.safetensors",
            1536, 640, 12345, 4.0,
            comp_strength="style"
        )
        self.assertEqual(wf_style["blend_ip_0"]["inputs"]["weight"], 0.20, "Style Only should use gentle 0.20 weight")
        self.assertEqual(wf_style["blend_ip_0"]["inputs"]["end_at"], 0.65, "Style Only should end at 0.65 to let LoRAs detail the face")
        self.assertEqual(wf_style["blend_ip_0"]["inputs"]["weight_type"], "ease out", "Style Only should use ease out curve")

        # 3. Test that Wai Illustrious Checkpoint Configs are applied
        self.assertEqual(wf_style["3"]["inputs"]["sampler_name"], "dpmpp_2m", "Illustrious should use dpmpp_2m sampler")
        self.assertEqual(wf_style["3"]["inputs"]["scheduler"], "karras")
        self.assertEqual(wf_style["3"]["inputs"]["steps"], 35, "Illustrious should use 35 steps")
        self.assertEqual(wf_style["3"]["inputs"]["cfg"], 3.5, "Illustrious should use calibrated CFG 3.5")

        # 4. Test negative prompt sanitization
        neg_text = wf_style["7"]["inputs"]["text"]
        self.assertNotIn("photorealistic", neg_text, "Semi-Realism should sanitize 'photorealistic' out of negative prompt")
        self.assertNotIn("anime", neg_text, "Illustrious model should sanitize 'anime' out of negative prompt")

    def test_module15_wan_workflow_json(self):
        """Test loading and validating Wan 2.2 workflow JSON template."""
        workflow_path = "workflows/archive/wan22_i2v.json"
        self.assertTrue(os.path.exists(workflow_path))
        with open(workflow_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("1", wf)
        self.assertIn("5", wf)
        self.assertIn("9", wf)
        self.assertEqual(wf["1"]["class_type"], "LoadImage")
        self.assertEqual(wf["2"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(wf["21"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(wf["22"]["class_type"], "ModelSamplingSD3")
        self.assertEqual(wf["22"]["inputs"]["shift"], 8.0)
        self.assertEqual(wf["3"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(wf["31"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(wf["8"]["class_type"], "WanImageToVideo")
        self.assertEqual(wf["108"]["class_type"], "CLIPVisionLoader")
        self.assertEqual(wf["108"]["inputs"]["clip_name"], "clip_vision_h.safetensors")
        self.assertEqual(wf["107"]["class_type"], "CLIPVisionEncode")
        self.assertEqual(wf["76"]["class_type"], "easy cleanGpuUsed")
        self.assertEqual(wf["75"]["class_type"], "RIFE VFI")
        self.assertEqual(wf["75"]["inputs"]["ckpt_name"], "rife49.pth")
        self.assertEqual(wf["9"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(wf["9"]["inputs"]["frame_rate"], 32)
        self.assertNotIn("audio", wf["9"]["inputs"])
        self.assertNotIn("150", wf)
        self.assertNotIn("151", wf)
        self.assertNotIn("152", wf)

    def test_module20_flux_workflow_and_lora(self):
        """Test Flux LoRA parsing (--ogarla for Flux) and LoraLoaderModelOnly injection."""
        from parsers import parse_loras, apply_loras_to_workflow

        # 1. Verify --ogarla maps to ogarlaflux_epoch_5 when is_flux=True and harmonizes trigger
        cleaned, loras = parse_loras("cyberpunk street --ogarla.80", is_flux=True)
        self.assertEqual(cleaned, "ogarlaflux, cyberpunk street")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.8)

        # 2. Verify Flux workflow pre-wired Node 76
        with open("workflows/archive/flux_lowres.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        modified_wf = apply_loras_to_workflow(wf, loras)
        self.assertIn("76", modified_wf)
        self.assertEqual(modified_wf["76"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(modified_wf["76"]["inputs"]["lora_name"], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(modified_wf["76"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(modified_wf["76"]["inputs"]["model"], ["1", 0])
        self.assertEqual(modified_wf["11"]["inputs"]["model"], ["76", 0])

    def test_module21_comfyui_models_directory(self):
        """Test that ComfyUI models/loras directory exists and contains registered LoRA files."""
        comfy_loras_dir = r"C:\ComfyUI\ComfyUI\models\loras"
        if os.path.exists(comfy_loras_dir):
            files = os.listdir(comfy_loras_dir)
            self.assertIn("ogarlaflux_epoch_1.safetensors", files)
            self.assertIn("ogarla_epoch_5.safetensors", files)
            self.assertIn("Semi-realism_illustrious.safetensors", files)
        else:
            self.skipTest(f"ComfyUI directory not found at {comfy_loras_dir}")

    def test_module23_checkpoint_configs_registry(self):
        """Test that CHECKPOINT_CONFIGS registry contains valid photorealistic checkpoint entries and default configs."""
        from bot import CHECKPOINT_CONFIGS
        self.assertIn("RealVisXL_V4.0.safetensors", CHECKPOINT_CONFIGS)
        self.assertIn("juggernautXL_ragnarok.safetensors", CHECKPOINT_CONFIGS)
        self.assertIn("CopaxTimeLessXL.safetensors", CHECKPOINT_CONFIGS)
        
        realvis = CHECKPOINT_CONFIGS["RealVisXL_V4.0.safetensors"]
        self.assertEqual(realvis["sampler_name"], "dpmpp_2m_sde")
        self.assertEqual(realvis["scheduler"], "karras")
        self.assertEqual(realvis["cfg"], 4.5)
        self.assertIn("cgi", realvis["negative_addon"])

    def test_module24_com_flux_gguf_workflow(self):
        """Test community Flux.1 GGUF workflow structure, nodes, guidance, and dual CLIP loaders."""
        wf_path = "workflows/archive/com_flux_gguf.json"
        self.assertTrue(os.path.exists(wf_path))
        with open(wf_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("1", wf)
        self.assertEqual(wf["1"]["class_type"], "UnetLoaderGGUF")
        self.assertIn("12", wf)
        self.assertEqual(wf["12"]["class_type"], "DualCLIPLoaderGGUF")
        self.assertIn("13", wf)
        self.assertEqual(wf["13"]["class_type"], "FluxGuidance")
        self.assertEqual(wf["13"]["inputs"]["guidance"], 3.5)
        self.assertIn("11", wf)
        self.assertEqual(wf["11"]["class_type"], "KSampler")
        self.assertEqual(wf["11"]["inputs"]["positive"], ["13", 0])

        # Test Ogarla Flux LoRA parsing & injection
        cleaned, loras = parse_loras("ogarla in a cafe --ogarla.80", is_flux=True)
        self.assertEqual(cleaned, "ogarlaflux in a cafe")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.8)

        injected_wf = apply_loras_to_workflow(wf, loras)
        self.assertIn("76", injected_wf)
        self.assertEqual(injected_wf["76"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(injected_wf["76"]["inputs"]["lora_name"], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(injected_wf["76"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(injected_wf["76"]["inputs"]["model"], ["1", 0])
        self.assertEqual(injected_wf["11"]["inputs"]["model"], ["76", 0])

    def test_module25_sdxl_powerhouse_2stage_workflow(self):
        """Test 2-stage SDXL powerhouse workflow with FreeU V2, Latent Upscale 1.35x, and Stage 2 refiner."""
        wf_path = "workflows/sdxl_powerhouse_2stage.json"
        self.assertTrue(os.path.exists(wf_path))
        with open(wf_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("4", wf)
        self.assertEqual(wf["4"]["class_type"], "CheckpointLoaderSimple")
        self.assertIn("75", wf)
        self.assertEqual(wf["75"]["class_type"], "LoraLoader")
        self.assertIn("76", wf)
        self.assertEqual(wf["76"]["class_type"], "LoraLoader")
        self.assertIn("20", wf)
        self.assertEqual(wf["20"]["class_type"], "FreeU_V2")
        self.assertEqual(wf["20"]["inputs"]["b1"], 1.3)
        self.assertEqual(wf["20"]["inputs"]["b2"], 1.4)
        self.assertIn("3", wf)
        self.assertEqual(wf["3"]["inputs"]["model"], ["20", 0])
        self.assertIn("14", wf)
        self.assertEqual(wf["14"]["class_type"], "LatentUpscaleBy")
        self.assertEqual(wf["14"]["inputs"]["scale_by"], 1.35)
        self.assertIn("15", wf)
        self.assertEqual(wf["15"]["inputs"]["denoise"], 0.48)
        self.assertEqual(wf["15"]["inputs"]["model"], ["20", 0])
        self.assertEqual(wf["8"]["inputs"]["samples"], ["15", 0])

        # Test Ogarla SDXL LoRA parsing & injection into 2-stage FreeU workflow
        cleaned, loras = parse_loras("ogarla in a temple --ogarla.75", is_flux=False)
        self.assertEqual(cleaned, "ogarla in a temple")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarla_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.75)

        injected_wf = apply_loras_to_workflow(wf, loras)
        self.assertEqual(injected_wf["76"]["inputs"]["lora_name"], "ogarla_epoch_5.safetensors")
        self.assertEqual(injected_wf["76"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(injected_wf["76"]["inputs"]["strength_clip"], 0.75)
        self.assertEqual(injected_wf["75"]["inputs"]["model"], ["4", 0])
        self.assertEqual(injected_wf["76"]["inputs"]["model"], ["75", 0])
        self.assertEqual(injected_wf["20"]["inputs"]["model"], ["76", 0])
        self.assertEqual(injected_wf["3"]["inputs"]["model"], ["20", 0])
        self.assertEqual(injected_wf["15"]["inputs"]["model"], ["20", 0])
        self.assertEqual(injected_wf["6"]["inputs"]["clip"], ["76", 1])
        self.assertEqual(injected_wf["7"]["inputs"]["clip"], ["76", 1])

    def test_module26_sdxl_checkpoint_isolation(self):
        """Test that SDXL_CHECKPOINT_CHOICES only contains valid SDXL models and excludes Flux/Video models."""
        from bot import SDXL_CHECKPOINT_CHOICES
        self.assertTrue(len(SDXL_CHECKPOINT_CHOICES) > 0)
        for choice in SDXL_CHECKPOINT_CHOICES:
            val = choice.value.lower()
            self.assertTrue(val.endswith(".safetensors"))
            self.assertNotIn("flux", val)
            self.assertNotIn("ltx", val)
            self.assertNotIn("wan", val)
            self.assertNotIn("hunyuan", val)

    def test_module30_face_detailer_sdxl_injection(self):
        """Test that apply_face_detailer_to_workflow properly attaches detector and FaceDetailer with 8GB VRAM settings."""
        from parsers import apply_face_detailer_to_workflow
        with open("workflows/txt2img_lowres.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        det_wf = apply_face_detailer_to_workflow(
            wf,
            seed=998877,
            cfg=4.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            steps=20,
            denoise=0.40,
            guide_size=512,
            max_size=768
        )

        # 1. Check detector node
        self.assertIn("80", det_wf)
        self.assertEqual(det_wf["80"]["class_type"], "UltralyticsDetectorProvider")
        self.assertEqual(det_wf["80"]["inputs"]["model_name"], "bbox/face_yolov8m.pt")

        # 2. Check FaceDetailer node
        self.assertIn("85", det_wf)
        self.assertEqual(det_wf["85"]["class_type"], "FaceDetailer")
        self.assertEqual(det_wf["85"]["inputs"]["seed"], 998877)
        self.assertEqual(det_wf["85"]["inputs"]["cfg"], 4.5)
        self.assertEqual(det_wf["85"]["inputs"]["sampler_name"], "dpmpp_2m_sde")
        self.assertEqual(det_wf["85"]["inputs"]["scheduler"], "karras")
        self.assertEqual(det_wf["85"]["inputs"]["steps"], 20)
        self.assertEqual(det_wf["85"]["inputs"]["denoise"], 0.40)
        self.assertEqual(det_wf["85"]["inputs"]["guide_size"], 512)
        self.assertEqual(det_wf["85"]["inputs"]["max_size"], 768)
        self.assertEqual(det_wf["85"]["inputs"]["bbox_detector"], ["80", 0])
        self.assertEqual(det_wf["85"]["inputs"]["image"], ["8", 0])

        # 3. Check Preview node connects to FaceDetailer output
        self.assertEqual(det_wf["9"]["inputs"]["images"], ["85", 0])

    def test_architecture_detection_and_classification(self):
        """Test model and LoRA architecture detection across all supported families."""
        from model_architecture import (
            detect_model_architecture, 
            Architecture, 
            SubType, 
            ModelType,
            get_architecture_badge
        )

        # 1. SDXL Models & LoRAs
        mtype, arch, subtype = detect_model_architecture("waiIllustriousSDXL_v170.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.ILLUSTRIOUS)

        mtype, arch, subtype = detect_model_architecture("RealVisXL_V4.0.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.REALISTIC)

        mtype, arch, subtype = detect_model_architecture("ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.PONY)

        mtype, arch, subtype = detect_model_architecture("Semi-realism_illustrious.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(mtype, ModelType.LORA)

        # 2. Flux Models & LoRAs
        mtype, arch, subtype = detect_model_architecture("ogarlaflux_epoch_5.safetensors")
        self.assertEqual(arch, Architecture.FLUX)
        self.assertEqual(mtype, ModelType.LORA)

        mtype, arch, subtype = detect_model_architecture("flux1-dev.safetensors")
        self.assertEqual(arch, Architecture.FLUX)

        # 3. Video Models (Wan & LTX)
        mtype, arch, subtype = detect_model_architecture("WAN-2.2-I2V-Handjob-HIGH-v1.safetensors")
        self.assertEqual(arch, Architecture.WAN)
        self.assertEqual(subtype, SubType.HIGH_NOISE)

        mtype, arch, subtype = detect_model_architecture("ltx-video-2b-v0.9.1.safetensors")
        self.assertEqual(arch, Architecture.LTX)

        # 4. Architecture Badges
        self.assertIn("SDXL", get_architecture_badge("sdxl"))
        self.assertIn("FLUX", get_architecture_badge("flux"))
        self.assertIn("WAN", get_architecture_badge("wan"))

    def test_lora_variant_autorouting_and_compatibility(self):
        """Test LoRA variant auto-routing and compatibility validator guards."""
        from model_architecture import (
            resolve_lora_for_architecture, 
            validate_architecture_compatibility, 
            Architecture, 
            SubType
        )
        from parsers import validate_workflow_loras

        # 1. Test auto-routing for Ogarla variants
        flux_ogarla = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", Architecture.FLUX)
        self.assertEqual(flux_ogarla, "ogarlaflux_epoch_5.safetensors")

        sdxl_ogarla = resolve_lora_for_architecture("ogarlaflux_epoch_5.safetensors", Architecture.SDXL)
        self.assertEqual(sdxl_ogarla, "ogarla_epoch_5.safetensors")

        pony_ogarla = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", Architecture.SDXL, SubType.PONY)
        self.assertEqual(pony_ogarla, "ogarlapony_epoch_6.safetensors")

        # 2. Test compatibility validation: Matching pairing
        is_compat, msg, sug = validate_architecture_compatibility(
            "waiIllustriousSDXL_v170.safetensors", 
            "Semi-realism_illustrious.safetensors"
        )
        self.assertTrue(is_compat)

        # 3. Test compatibility validation: Mismatched pairing with auto-route suggestion
        is_compat, msg, sug = validate_architecture_compatibility(
            "flux1-dev.safetensors", 
            "ogarla_epoch_5.safetensors"
        )
        self.assertFalse(is_compat)
        self.assertEqual(sug, "ogarlaflux_epoch_5.safetensors")

        # 4. Test compatibility validation: Incompatible pairing without replacement
        is_compat, msg, sug = validate_architecture_compatibility(
            "ltx-video-2b-v0.9.1.safetensors", 
            "Semi-realism_illustrious.safetensors"
        )
        self.assertFalse(is_compat)
        self.assertIsNone(sug)

        # 5. Test validate_workflow_loras
        valid, msgs = validate_workflow_loras(
            {}, 
            "waiIllustriousSDXL_v170.safetensors", 
            [("Semi-realism_illustrious.safetensors", 0.7)]
        )
        self.assertTrue(valid)

    def test_character_registry_and_masking(self):
        """Test character registry lookups, pseudonym resolution, and prompt trigger masking."""
        from characters import get_character, mask_character_in_prompt, inject_trained_trigger_in_prompt, CHARACTERS

        # Lookups
        val = get_character("valerie")
        self.assertIsNotNone(val)
        self.assertEqual(val.trained_trigger, "jen")
        self.assertEqual(val.lora_sdxl, "jen_epoch_5.safetensors")

        val_alias = get_character("val")
        self.assertEqual(val_alias.id, "valerie")

        oga = get_character("ogarla")
        self.assertIsNotNone(oga)
        self.assertEqual(oga.lora_sdxl, "ogarla_epoch_5.safetensors")

        # Sully checks
        sul = get_character("sully")
        self.assertIsNotNone(sul)
        self.assertEqual(sul.trained_trigger, "susa")
        self.assertEqual(sul.lora_sdxl, "susa_epoch_6.safetensors")
        self.assertEqual(sul.base_prompt_traits, "black hair, thin rim glasses")

        sul_alias = get_character("sul")
        self.assertEqual(sul_alias.id, "sully")

        # Masking trained trigger -> display alias
        masked = mask_character_in_prompt("jen sitting in a coffee shop")
        self.assertEqual(masked, "valerie sitting in a coffee shop")

        masked_sully = mask_character_in_prompt("susa walking down the street")
        self.assertEqual(masked_sully, "sully walking down the street")

        # Injecting trained trigger -> ComfyUI prompt
        injected = inject_trained_trigger_in_prompt("valerie sitting in a coffee shop", "valerie")
        self.assertEqual(injected, "jen, brown hair, dark brown eyes, realistic skin texture sitting in a coffee shop")

        injected_sully = inject_trained_trigger_in_prompt("sully, in a library", "sully")
        self.assertEqual(injected_sully, "susa, black hair, thin rim glasses, in a library")

        # Mageill character checks (original character, not masked)
        mag = get_character("mageill")
        self.assertIsNotNone(mag)
        self.assertEqual(mag.lora_sdxl, "mageill_epoch_5.safetensors")
        self.assertFalse(mag.is_private)

        mag3 = get_character("mageill3")
        self.assertIsNotNone(mag3)
        self.assertEqual(mag3.lora_sdxl, "mageill_epoch_3.safetensors")

        mag6 = get_character("mag6")
        self.assertIsNotNone(mag6)
        self.assertEqual(mag6.lora_sdxl, "mageill_epoch_6.safetensors")

        # Unmasked prompt stays unmasked
        masked_mag = mask_character_in_prompt("mageill casting a spell")
        self.assertEqual(masked_mag, "mageill casting a spell")

        # Cheri character checks (original character with blonde hair trait)
        che = get_character("cheri")
        self.assertIsNotNone(che)
        self.assertEqual(che.lora_sdxl, "cheri_epoch_6.safetensors")
        self.assertEqual(che.base_prompt_traits, "blonde hair")
        self.assertFalse(che.is_private)

        che4 = get_character("che4")
        self.assertIsNotNone(che4)
        self.assertEqual(che4.lora_sdxl, "cheri_epoch_4.safetensors")

        masked_che = mask_character_in_prompt("cheri smiling in the sunlight")
        self.assertEqual(masked_che, "cheri smiling in the sunlight")

        # Test unified character display badges
        from characters import get_character_display_badge
        self.assertEqual(get_character_display_badge("none"), "None")
        self.assertEqual(get_character_display_badge(None), "None")
        self.assertEqual(get_character_display_badge("valerie.90", architecture="krea2"), "✨ Valerie (.90 - Default)")
        self.assertEqual(get_character_display_badge("valerie.70", architecture="krea2"), "✨ Valerie (.70 - Light)")
        self.assertEqual(get_character_display_badge("ogarla.85", architecture="krea2"), "🌿 Ogarla (.85 - Default)")
        self.assertEqual(get_character_display_badge("ogarla.70", architecture="krea2"), "🌿 Ogarla (.70 - Light)")
        self.assertEqual(get_character_display_badge("ogarla", architecture="sdxl"), "🌿 Ogarla (--ogarla.70)")
        self.assertEqual(get_character_display_badge("valerie", architecture="sdxl"), "👩 Valerie (--valerie.85)")
        self.assertEqual(get_character_display_badge("sully", architecture="sdxl"), "👓 Sully (--sully.85)")
        self.assertEqual(get_character_display_badge("none", architecture="sdxl", prompt="photo of a woman --valerie.85"), "👩 Valerie (--valerie.85)")
        self.assertEqual(get_character_display_badge("none", architecture="sdxl", prompt="photo of a woman --sully"), "👓 Sully (--sully.85)")

        # Test dynamic character autocomplete choices
        from characters import get_character_autocomplete_choices
        from model_architecture import Architecture

        # 1. Krea 2 autocomplete
        krea_choices = get_character_autocomplete_choices("", Architecture.KREA2)
        krea_vals = [c.value for c in krea_choices]
        self.assertNotIn("valerie.90", krea_vals)
        self.assertNotIn("valerie.70", krea_vals)
        self.assertIn("ogarla.85", krea_vals)
        self.assertIn("ogarla.70", krea_vals)

        # Krea 2 filtering
        krea_filtered = get_character_autocomplete_choices("oga", Architecture.KREA2)
        self.assertTrue(all("ogarla" in c.value for c in krea_filtered))

        # 2. SDXL autocomplete
        sdxl_choices = get_character_autocomplete_choices("", Architecture.SDXL)
        sdxl_vals = [c.value for c in sdxl_choices]
        self.assertIn("mageill.85", sdxl_vals)
        self.assertIn("ogarla.85", sdxl_vals)
        self.assertIn("valerie.85", sdxl_vals)
        self.assertIn("sully.85", sdxl_vals)
        self.assertIn("cheri.85", sdxl_vals)

        # SDXL filtering
        sdxl_filtered = get_character_autocomplete_choices("sully", Architecture.SDXL)
        self.assertTrue(all("sully" in c.value for c in sdxl_filtered))

        # 3. Flux autocomplete
        flux_choices = get_character_autocomplete_choices("", Architecture.FLUX)
        flux_vals = [c.value for c in flux_choices]
        self.assertIn("ogarla.85", flux_vals)
        self.assertIn("ogarla.70", flux_vals)

    def test_blend_character_loras_and_embed_badges(self):
        """Test build_blend_embed 3-column dashboard and handle_generate_blended character LoRAs & style presets."""
        import asyncio
        from unittest.mock import AsyncMock, patch, MagicMock
        from views import build_blend_embed, BlendButtons
        from image_utils import detect_closest_aspect_ratio, detect_closest_krea_aspect_ratio
        import bot

        # 1. Test detect_closest_aspect_ratio
        self.assertEqual(detect_closest_aspect_ratio(1920, 1080), "16:9")
        self.assertEqual(detect_closest_aspect_ratio(1080, 1920), "9:16")
        self.assertEqual(detect_closest_aspect_ratio(800, 1200), "3:5")
        self.assertEqual(detect_closest_aspect_ratio(1000, 1000), "1:1")
        self.assertEqual(detect_closest_aspect_ratio(2560, 1080), "21:9")
        self.assertEqual(detect_closest_aspect_ratio(1000, 700), "10:7")
        self.assertEqual(detect_closest_aspect_ratio(0, 100), "16:9")

        # 1b. Test detect_closest_krea_aspect_ratio (Krea 2 Studio 5-button set)
        self.assertEqual(detect_closest_krea_aspect_ratio(2560, 1080), "21:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1920, 1080), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1024, 768), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1500, 1000), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 1000), "1:1")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 950), "1:1")
        self.assertEqual(detect_closest_krea_aspect_ratio(768, 1024), "3:4")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 1500), "3:4")
        self.assertEqual(detect_closest_krea_aspect_ratio(1080, 1920), "9:16")
        self.assertEqual(detect_closest_krea_aspect_ratio(1170, 2532), "9:16")
        self.assertEqual(detect_closest_krea_aspect_ratio(0, 100), "16:9")

        # 1c. Test create_thumbnail_bytes
        from image_utils import create_thumbnail_bytes
        from PIL import Image
        import io
        img = Image.new("RGBA", (800, 600), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb_bytes = create_thumbnail_bytes(buf.getvalue(), max_dim=256)
        self.assertIsNotNone(thumb_bytes)
        with Image.open(io.BytesIO(thumb_bytes)) as thumb_img:
            self.assertLessEqual(thumb_img.width, 256)
            self.assertLessEqual(thumb_img.height, 256)

        # 2. Test build_blend_embed 3-column inline dashboard formatting
        gen_data = {
            "caption": "a woman standing in a garden",
            "detailed_caption": "detailed description of woman in garden",
            "ar": "3:5",
            "model_choice": "wai",
            "comp_strength": "style",
            "sr": "sr75",
            "char_choice": "sully",
            "sref_rand": "preset_junji_ito",
        }
        embed = build_blend_embed(gen_data, author_str="TestUser")
        field_dict = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", field_dict)
        self.assertIn("🤖 Checkpoint", field_dict)
        self.assertIn("🎭 Aesthetics", field_dict)

        self.assertIn("`3:5`", field_dict["📐 Canvas & Framing"])
        self.assertIn("--sr.75", field_dict["🤖 Checkpoint"])
        self.assertIn("👓 Sully (--sully.85)", field_dict["🎭 Aesthetics"])
        self.assertIn("🖋️ Junji Ito", field_dict["🎭 Aesthetics"])

        # 3. Test BlendButtons unified view and --sref random toggle
        user_favs = [
            {"code": 112233, "name": "Vaporwave Neon", "prompt": "cyberpunk neon colors"}
        ]
        view = BlendButtons("gen_test_view", ar="3:5", sr="sr70", char_choice="valerie", sref_rand="sref", user_favorites=user_favs)
        self.assertLessEqual(len(view.children), 25)
        rows = set(child.row for child in view.children)
        self.assertLessEqual(len(rows), 5)
        # Verify --sref random toggle button
        style_btn = next((c for c in view.children if hasattr(c, "custom_id") and c.custom_id and c.custom_id.startswith("toggle_blend_sref:")), None)
        self.assertIsNotNone(style_btn)
        self.assertIn("--sref random: ON", style_btn.label)

        # 4. Test handle_generate_blended character flag assembly and saved style resolution
        bot.db.save_generation("gen_test_char", gen_data)
        bot.active_generations["gen_test_char"] = gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()
        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test character flag injection: Valerie
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="3:5", use_sr="sr70", char_choice="valerie", use_sref_rand="nosref"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("valerie,", called_prompt)
            self.assertIn("--valerie.85", called_prompt)
            self.assertIn("--sr.70", called_prompt)
            self.assertIn("--ar 3:5", called_prompt)

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test style preset injection: Junji Ito
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="16:9", use_sr="nosr", char_choice="cheri", use_sref_rand="preset_junji_ito"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("cheri,", called_prompt)
            self.assertIn("--cheri.85", called_prompt)
            self.assertIn("Junji Ito", called_prompt)

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test saved style injection from user favorites
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="1:1", use_sr="nosr", char_choice="nochar", use_sref_rand="saved_112233"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("--sref 112233", called_prompt)

    def test_module42_bertflow_workflow_and_command(self):
        """Test Bertflow workflow generation, aspect ratio resolver, UNET detection, and button controls."""
        from parsers import (
            resolve_bertflow_dimensions,
            get_bertflow_unet_model,
            prepare_bertflow_workflow,
            BERTFLOW_ASPECT_RATIOS
        )
        from views import BertflowButtons

        # 1. Test aspect ratio resolution
        self.assertEqual(resolve_bertflow_dimensions("a portrait", "1:1")[1:], (1224, 1224))
        self.assertEqual(resolve_bertflow_dimensions("a landscape", "16:9")[1:], (1632, 920))
        self.assertEqual(resolve_bertflow_dimensions("a tall portrait", "9:16")[1:], (920, 1632))
        self.assertEqual(resolve_bertflow_dimensions("an ultrawide", "21:9")[1:], (1872, 800))

        # Dynamic --ar flag
        p_clean, w, h = resolve_bertflow_dimensions("a cinematic shot --ar 16:9", None)
        self.assertEqual(p_clean, "a cinematic shot")
        self.assertEqual((w, h), (1632, 920))

        # 2. Test UNET model detection
        active_unet = get_bertflow_unet_model()
        self.assertTrue(active_unet.endswith(".safetensors"))
        self.assertIn(active_unet, ["museByStableYogi_v35Int8Extended.safetensors", "pornmasterKrea2_v1FP8.safetensors"])

        # 3. Test prepare_bertflow_workflow
        wf = prepare_bertflow_workflow(
            prompt="Scott kneeling in living room",
            width=1224,
            height=1224,
            seed=424242,
            steps=8,
            unet_model="pornmasterKrea2_v1FP8.safetensors"
        )
        self.assertEqual(wf["627"]["inputs"]["text"], "Scott kneeling in living room")
        self.assertEqual(wf["649"]["inputs"]["seed"], 424242)
        self.assertEqual(wf["698"]["inputs"]["width"], 1224)
        self.assertEqual(wf["698"]["inputs"]["height"], 1224)
        self.assertEqual(wf["599"]["inputs"]["steps"], 8)
        self.assertEqual(wf["761"]["inputs"]["unet_name"], "pornmasterKrea2_v1FP8.safetensors")
        self.assertEqual(wf["822"]["inputs"]["lora_1"]["strength"], -2.0)
        self.assertEqual(wf["763"]["class_type"], "ConditioningZeroOut")

        # 4. Test BertflowButtons (9 buttons: Row 0 actions [reroll, remix, toggle char, upscale] + Row 1 pan controls)
        view = BertflowButtons(generation_id="bert_test_123", character=None)
        self.assertEqual(len(view.children), 9)
        btn_ids = [c.custom_id for c in view.children]
        self.assertIn("bertflow_reroll:bert_test_123", btn_ids)
        self.assertIn("bertflow_remix:bert_test_123", btn_ids)
        self.assertIn("bertflow_toggle_char:bert_test_123", btn_ids)
        self.assertIn("bertflow_upscale:bert_test_123", btn_ids)
        self.assertIn("outpaint:bert_test_123:1:up", btn_ids)
        self.assertIn("outpaint:bert_test_123:1:down", btn_ids)
        self.assertEqual(view.toggle_char_btn.label, "🌿 Ogarla: OFF")

        # Test BertflowButtons with active character
        view_char = BertflowButtons(generation_id="bert_test_456", character="ogarla.85")
        self.assertEqual(view_char.toggle_char_btn.label, "🌿 Ogarla: ON")

        # 5. Test command registration and parameters
        from bot import bot
        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("bertflow", commands)
        bert_cmd = commands["bertflow"]
        param_names = [p.name for p in bert_cmd.parameters]
        self.assertIn("favorite_prompt", param_names)
        self.assertIn("free", commands)
        self.assertNotIn("purge-vram", commands)

        # 6. Test PromptPaginationView with bertflow_callback
        from views import PromptPaginationView
        sample_prompts = [{"id": 1, "prompt_name": "Test Krea Prompt", "prompt_text": "A photo of a cyberpunk city"}]
        view_pg = PromptPaginationView(user_id=123, prompts=sample_prompts, per_page=5, imagine_callback=lambda inter, p: None, bertflow_callback=lambda inter, p: None)
        labels = [c.label for c in view_pg.children if hasattr(c, "label") and c.label]
        self.assertIn("⚡ Bertflow", labels)
        self.assertIn("🎨 Imagine", labels)

    def test_bertflow_composition_modes(self):
        """Verify prepare_bertflow_workflow and BlendKreaButtons direct composition options."""
        from parsers import prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed

        # 1. Test 'off' mode (pure txt2img default)
        wf_off = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1632,
            height=920,
            init_image="test_input.png",
            comp_strength="off"
        )
        self.assertNotIn("900", wf_off)
        self.assertNotIn("901", wf_off)
        self.assertNotIn("902", wf_off)
        self.assertEqual(wf_off["599"]["inputs"]["latent_image"], ["698", 0])
        self.assertEqual(wf_off["599"]["inputs"]["start_at_step"], 0)

        # 2. Test 'medium' composition mode (70% denoise)
        wf_med = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1632,
            height=920,
            init_image="test_input.png",
            comp_strength="medium"
        )
        self.assertIn("900", wf_med)
        self.assertEqual(wf_med["900"]["inputs"]["image"], "test_input.png")
        self.assertIn("901", wf_med)
        self.assertEqual(wf_med["901"]["inputs"]["width"], 1632)
        self.assertEqual(wf_med["901"]["inputs"]["height"], 920)
        self.assertEqual(wf_med["901"]["inputs"]["crop"], "center")
        self.assertIn("902", wf_med)
        self.assertEqual(wf_med["902"]["inputs"]["vae"], ["757", 0])
        self.assertEqual(wf_med["599"]["inputs"]["latent_image"], ["902", 0])
        self.assertEqual(wf_med["599"]["inputs"]["start_at_step"], 2)
        self.assertEqual(wf_med["599"]["inputs"]["add_noise"], "enable")
        self.assertEqual(wf_med["599"]["inputs"]["return_with_leftover_noise"], "disable")

        # 3. Test 'strong' composition mode (50% denoise)
        wf_strong = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1224,
            height=1224,
            init_image="test_input.png",
            comp_strength="strong"
        )
        self.assertEqual(wf_strong["599"]["inputs"]["start_at_step"], 4)

        # 4. Test BlendKreaButtons with composition dropdown
        view_off = BlendKreaButtons(generation_id="gen_krea_test", composition="off")
        comp_select_off = next(item for item in view_off.children if getattr(item, "custom_id", "") == "set_blend_krea_comp:gen_krea_test")
        self.assertEqual(comp_select_off.options[0].value, "off")
        self.assertTrue(comp_select_off.options[0].default)

        view_med = BlendKreaButtons(generation_id="gen_krea_test", composition="medium")
        comp_select_med = next(item for item in view_med.children if getattr(item, "custom_id", "") == "set_blend_krea_comp:gen_krea_test")
        self.assertEqual(comp_select_med.options[2].value, "medium")
        self.assertTrue(comp_select_med.options[2].default)

        # 5. Test embed display with composition
        embed = build_blend_krea_embed({"krea2_prompt": "Test", "composition": "medium"})
        pipeline_field = next(f.value for f in embed.fields if f.name == "⚙️ Pipeline Settings")
        self.assertIn("Direct Comp:** `Medium (Balanced Silhouette & Pose - 70% Denoise)`", pipeline_field)

    def test_module64_photo_dataset_builder(self):
        """Verify prompt matrix, IPAdapter multi-image workflow, and AI-Toolkit config generation."""
        import tempfile
        from tools.create_character_dataset_from_photos import (
            generate_prompt_matrix,
            build_ipadapter_workflow,
            write_ai_toolkit_config
        )

        # 1. Prompt matrix generation
        prompts = generate_prompt_matrix(count=20)
        self.assertEqual(len(prompts), 20)
        self.assertTrue(any("full-body" in p for p in prompts))
        self.assertTrue(any("medium" in p or "waist-up" in p or "cowboy" in p for p in prompts))
        self.assertTrue(any("portrait" in p or "close-up" in p for p in prompts))

        # 2. IPAdapter workflow generation for 3 reference images
        ref_images = ["photo1.png", "photo2.png", "photo3.png"]
        wf = build_ipadapter_workflow(
            reference_image_names=ref_images,
            prompt_text="a fashion photoshoot of a woman",
            seed=42,
            checkpoint="RealVisXL_V4.0.safetensors",
            resolution=1024
        )
        self.assertIn("20", wf)  # IPAdapterUnifiedLoader
        self.assertEqual(wf["20"]["class_type"], "IPAdapterUnifiedLoader")
        self.assertEqual(wf["20"]["inputs"]["preset"], "PLUS (high strength)")

        # Verify load nodes for all 3 images
        self.assertIn("ref_img_0", wf)
        self.assertIn("ref_img_1", wf)
        self.assertIn("ref_img_2", wf)
        self.assertEqual(wf["ref_img_0"]["inputs"]["image"], "photo1.png")
        self.assertEqual(wf["ref_img_2"]["inputs"]["image"], "photo3.png")

        # Verify chained IPAdapterAdvanced nodes
        self.assertIn("ref_ip_0", wf)
        self.assertIn("ref_ip_1", wf)
        self.assertIn("ref_ip_2", wf)
        # KSampler model input points to the final IPAdapter in chain
        self.assertEqual(wf["3"]["inputs"]["model"], ["ref_ip_2", 0])

        # 3. AI-Toolkit config generation
        with tempfile.TemporaryDirectory() as tmpdir:
            out_yaml = os.path.join(tmpdir, "test_config.yaml")
            write_ai_toolkit_config("my_test_char", tmpdir, out_yaml)
            self.assertTrue(os.path.exists(out_yaml))
            with open(out_yaml, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn('name: "my_test_char_krea2"', content)
            self.assertIn('trigger_word: "my_test_char"', content)
            self.assertIn('arch: "krea2"', content)
            self.assertIn('noise_scheduler: "flowmatch"', content)

    def test_module66_krea2_celebrity_presets(self):
        """Verify Krea 2 celebrity registry, prompt injection, UI dropdowns, and slash command choices."""
        from celebrities import (
            FAVORITE_CELEBRITIES,
            get_celebrity,
            get_celebrity_display_badge,
            inject_celebrity_in_prompt,
            get_celebrity_autocomplete_choices,
            CELEBRITY_CHOICES_KREA2
        )
        from parsers import prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed
        import bot

        # 1. Verify all 16 favorite celebrities are present
        expected_celebs = [
            "Audrey Hepburn", "Grace Kelly", "Nicole Kidman", "Margot Robbie",
            "Sandra Bullock", "Emma Stone", "Anya Taylor-Joy", "Zendaya",
            "Cameron Diaz", "Saoirse Ronan", "Emma Watson", "Michelle Pfeiffer",
            "Gal Gadot", "Taylor Swift", "Ariana Grande", "Keira Knightley"
        ]
        self.assertEqual(len(FAVORITE_CELEBRITIES), 16)
        for name in expected_celebs:
            celeb = get_celebrity(name)
            self.assertIsNotNone(celeb, f"Celebrity '{name}' should be resolved by get_celebrity")
            self.assertEqual(celeb.display_name, name)

        # 2. Test display badges
        self.assertEqual(get_celebrity_display_badge("none"), "None")
        self.assertEqual(get_celebrity_display_badge(None), "None")
        self.assertIn("Audrey Hepburn", get_celebrity_display_badge("audrey_hepburn"))
        self.assertIn("Zendaya", get_celebrity_display_badge("zendaya"))

        # 3. Test prompt injection
        injected = inject_celebrity_in_prompt("walking through Paris in spring", "audrey_hepburn")
        self.assertTrue(injected.startswith("Audrey Hepburn,"))
        self.assertIn("walking through Paris in spring", injected)

        # No double injection if already present
        already_in = inject_celebrity_in_prompt("photo of Audrey Hepburn in Paris", "audrey_hepburn")
        self.assertEqual(already_in, "photo of Audrey Hepburn in Paris")

        # 4. Test prepare_bertflow_workflow injection via argument and prompt flag
        wf_arg = prepare_bertflow_workflow("a sunny afternoon", celebrity="zendaya")
        self.assertTrue(wf_arg["627"]["inputs"]["text"].startswith("Zendaya,"))

        wf_flag = prepare_bertflow_workflow("a red carpet event --margot")
        self.assertTrue(wf_flag["627"]["inputs"]["text"].startswith("Margot Robbie,"))

        # Both character LoRA and celebrity preset can coexist
        wf_both = prepare_bertflow_workflow("a stylish portrait", character="ogarla.85", celebrity="audrey_hepburn")
        self.assertIn("ogarla", wf_both["627"]["inputs"]["text"].lower())
        self.assertIn("Audrey Hepburn", wf_both["627"]["inputs"]["text"])

        # 5. Test BlendKreaButtons has both Character and Celebrity dropdowns
        view = BlendKreaButtons(
            generation_id="krea_celeb_test",
            ar="16:9",
            model_choice="muse",
            wetness=-2.0,
            composition="off",
            character="ogarla.85",
            celebrity="zendaya"
        )
        celeb_select = next(item for item in view.children if getattr(item, "custom_id", None) == "krea_celeb_test" or getattr(item, "custom_id", None) == "set_blend_krea_celeb:krea_celeb_test")
        self.assertIsNotNone(celeb_select)
        celeb_vals = [opt.value for opt in celeb_select.options]
        self.assertIn("none", celeb_vals)
        self.assertIn("audrey_hepburn", celeb_vals)
        self.assertIn("zendaya", celeb_vals)
        self.assertIn("keira_knightley", celeb_vals)
        self.assertEqual(len(celeb_vals), 17)  # None + 16 celebrities

        # Character select still present and functional
        char_select = next(item for item in view.children if getattr(item, "custom_id", None) == "set_blend_krea_char:krea_celeb_test")
        self.assertIsNotNone(char_select)

        # 6. Test build_blend_krea_embed shows both Character and Celebrity badges
        gen_data = {
            "krea2_prompt": "Portrait in natural sunlight",
            "fused_prompt": "Portrait in natural sunlight",
            "ar": "16:9",
            "model_choice": "muse",
            "wetness": -2.0,
            "char_choice": "ogarla.85",
            "celeb_choice": "audrey_hepburn"
        }
        embed = build_blend_krea_embed(gen_data)
        settings_field = next(f for f in embed.fields if f.name == "⚙️ Pipeline Settings")
        self.assertIn("Character:", settings_field.value)
        self.assertIn("Ogarla", settings_field.value)
        self.assertIn("Celebrity:", settings_field.value)
        self.assertIn("Audrey Hepburn", settings_field.value)

        # 7. Test slash command parameters
        bert_cmd = next(c for c in bot.bot.tree.get_commands() if c.name == "bertflow")
        self.assertIn("character", [p.name for p in bert_cmd.parameters])
        self.assertIn("celebrity", [p.name for p in bert_cmd.parameters])

        blend_cmd = next(c for c in bot.bot.tree.get_commands() if c.name == "blend-krea")
        self.assertIn("image", [p.name for p in blend_cmd.parameters])
        self.assertIn("steps", [p.name for p in blend_cmd.parameters])

    def test_module67_lora_workflow_architecture_audit(self):
        """Audit all workflows, character profiles, resolvers, and parsers for 100% LoRA architecture match."""
        from scripts.audit_loras import (
            audit_workflows,
            audit_characters,
            audit_parsers_and_presets,
            audit_ui_choices
        )
        wf_issues = audit_workflows()
        self.assertEqual(wf_issues, [], f"Workflow LoRA audit issues: {wf_issues}")

        char_issues = audit_characters()
        self.assertEqual(char_issues, [], f"Character LoRA audit issues: {char_issues}")

        preset_issues = audit_parsers_and_presets()
        self.assertEqual(preset_issues, [], f"Preset LoRA audit issues: {preset_issues}")

        ui_issues = audit_ui_choices()
        self.assertEqual(ui_issues, [], f"UI LoRA audit issues: {ui_issues}")

    def test_semantic_workflow_discovery(self):
        """Test semantic node discovery by class_type, title, and input keys."""
        from services.workflow_adapter import find_nodes, find_node, get_positive_negative_nodes

        wf = {
            "node_a": {"class_type": "KSampler", "_meta": {"title": "Main Sampler"}, "inputs": {"seed": 100, "positive": ["node_b", 0], "negative": ["node_c", 0]}},
            "node_b": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text (Positive)"}, "inputs": {"text": "hello"}},
            "node_c": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text (Negative)"}, "inputs": {"text": "bad"}},
            "node_d": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}}
        }

        # 1. Discover by class
        samplers = find_nodes(wf, class_type="KSampler")
        self.assertEqual(len(samplers), 1)
        self.assertEqual(samplers[0][0], "node_a")

        # 2. Discover by title
        pos_by_title = find_node(wf, title_contains="positive")
        self.assertIsNotNone(pos_by_title)
        self.assertEqual(pos_by_title[0], "node_b")

        # 3. Discover positive/negative via graph tracing
        pos_match, neg_match = get_positive_negative_nodes(wf)
        self.assertIsNotNone(pos_match)
        self.assertIsNotNone(neg_match)
        self.assertEqual(pos_match[0], "node_b")
        self.assertEqual(neg_match[0], "node_c")

    def test_semantic_workflow_setters(self):
        """Test semantic setters update prompts, seed, dimensions, and checkpoint."""
        from services.workflow_adapter import (
            set_workflow_prompt,
            set_workflow_seed,
            set_workflow_dimensions,
            set_workflow_checkpoint,
            set_workflow_sampler_params,
            set_workflow_filename_prefix
        )

        wf = {
            "node_sampler": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 4.0, "positive": ["node_pos", 0], "negative": ["node_neg", 0]}},
            "node_pos": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}, "inputs": {"text": "old pos"}},
            "node_neg": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative"}, "inputs": {"text": "old neg"}},
            "node_latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "node_ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}},
            "node_save": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old_prefix"}}
        }

        # Set prompts
        set_workflow_prompt(wf, positive="masterpiece, new character", negative="low quality, blur")
        self.assertEqual(wf["node_pos"]["inputs"]["text"], "masterpiece, new character")
        self.assertEqual(wf["node_neg"]["inputs"]["text"], "low quality, blur")

        # Set seed
        set_workflow_seed(wf, 888999)
        self.assertEqual(wf["node_sampler"]["inputs"]["seed"], 888999)

        # Set dimensions
        set_workflow_dimensions(wf, width=1920, height=1080, batch_size=4)
        self.assertEqual(wf["node_latent"]["inputs"]["width"], 1920)
        self.assertEqual(wf["node_latent"]["inputs"]["height"], 1080)
        self.assertEqual(wf["node_latent"]["inputs"]["batch_size"], 4)

        # Set checkpoint & sampler params
        set_workflow_checkpoint(wf, ckpt_name="waiIllustriousSDXL_v170.safetensors")
        self.assertEqual(wf["node_ckpt"]["inputs"]["ckpt_name"], "waiIllustriousSDXL_v170.safetensors")

        set_workflow_sampler_params(wf, steps=35, cfg=6.5)
        self.assertEqual(wf["node_sampler"]["inputs"]["steps"], 35)
        self.assertEqual(wf["node_sampler"]["inputs"]["cfg"], 6.5)

        # Set filename prefix
        set_workflow_filename_prefix(wf, "Discord Bot/custom/new_output")
        self.assertEqual(wf["node_save"]["inputs"]["filename_prefix"], "Discord Bot/custom/new_output")

    def test_renumbered_workflow_immunity(self):
        """
        Critical Resilience Test:
        Scrambles all node IDs in a full SDXL workflow to arbitrary 4-digit numbers,
        rewires link pointers, and verifies semantic adapter successfully modifies
        the renumbered workflow without any KeyErrors or failures.
        """
        from parsers import load_workflow_template, apply_loras_to_workflow
        from services.workflow_adapter import (
            set_workflow_prompt,
            set_workflow_seed,
            set_workflow_dimensions,
            set_workflow_checkpoint
        )

        orig_wf = load_workflow_template("workflows/txt2img_lowres.json")

        # Mapping of original node IDs to completely scrambled new IDs
        id_map = {
            "3": "8103",   # KSampler
            "4": "8104",   # CheckpointLoaderSimple
            "5": "8105",   # EmptyLatentImage
            "6": "8106",   # CLIPTextEncode (Positive)
            "7": "8107",   # CLIPTextEncode (Negative)
            "8": "8108",   # VAEDecode
            "9": "8109",   # PreviewImage
            "75": "8175",  # Load LoRA: Semi-Realism
            "76": "8176",  # Load LoRA: Ogarla
        }

        # Build scrambled workflow with updated inter-node link references
        scrambled_wf = {}
        for old_id, node in orig_wf.items():
            new_id = id_map.get(old_id, f"9{old_id}")
            node_copy = copy.deepcopy(node)
            for k, val in node_copy.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) in id_map:
                    val[0] = id_map[str(val[0])]
            scrambled_wf[new_id] = node_copy

        # Confirm old numeric IDs DO NOT exist in scrambled workflow
        self.assertNotIn("3", scrambled_wf)
        self.assertNotIn("75", scrambled_wf)
        self.assertNotIn("76", scrambled_wf)

        # 1. Apply Prompt Semantically
        set_workflow_prompt(scrambled_wf, positive="scrambled test positive", negative="scrambled test negative")
        self.assertEqual(scrambled_wf["8106"]["inputs"]["text"], "scrambled test positive")
        self.assertEqual(scrambled_wf["8107"]["inputs"]["text"], "scrambled test negative")

        # 2. Apply Seed Semantically
        set_workflow_seed(scrambled_wf, 999111)
        self.assertEqual(scrambled_wf["8103"]["inputs"]["seed"], 999111)

        # 3. Apply Dimensions Semantically
        set_workflow_dimensions(scrambled_wf, width=1280, height=720, batch_size=2)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["width"], 1280)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["height"], 720)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["batch_size"], 2)

        # 4. Apply LoRAs Semantically to Renumbered Workflow
        updated_lora_wf = apply_loras_to_workflow(scrambled_wf, [("semi-realism", 0.85), ("ogarla", 0.70)])
        self.assertEqual(updated_lora_wf["8175"]["inputs"]["strength_model"], 0.85)
        self.assertEqual(updated_lora_wf["8176"]["inputs"]["strength_model"], 0.70)

    def test_pipeline_defaults_and_checkpoint_display_names(self):
        """Test centralized PipelineDefaults constants and get_checkpoint_display_name resolution."""
        from config import PipelineDefaults, get_checkpoint_display_name

        # Check default checkpoints
        self.assertTrue(PipelineDefaults.DEFAULT_SDXL_CHECKPOINT.endswith(".safetensors"))
        self.assertTrue(PipelineDefaults.DEFAULT_FLUX_CHECKPOINT.endswith(".safetensors"))
        self.assertTrue(PipelineDefaults.DEFAULT_WAN_CHECKPOINT.endswith(".safetensors"))

        # Check denoise values
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_SDXL, 0.55)
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_FLUX_SUBTLE, 0.26)
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_FLUX_MODERATE, 0.35)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_VERY_HIGH, 0.95)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["low"], 0.85)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["med"], 0.60)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["high"], 0.50)

        # Check display name resolution
        self.assertEqual(get_checkpoint_display_name("waiIllustriousSDXL_v170.safetensors"), "Wai Illustrious SDXL v1.70")
        self.assertEqual(get_checkpoint_display_name("RealVisXL_V4.0.safetensors"), "RealVisXL V4.0")
        self.assertEqual(get_checkpoint_display_name("wai"), "Wai Illustrious SDXL v1.70")
        self.assertEqual(get_checkpoint_display_name("realvis"), "RealVisXL V4.0")
        self.assertEqual(get_checkpoint_display_name(None), "Default Model")
        self.assertEqual(get_checkpoint_display_name("custom_model.safetensors"), "custom_model")

    def test_anima_architecture_and_detection(self):
        """Test Anima architecture registration, badge display, and model classification."""
        from model_architecture import Architecture, SubType, ModelType, ARCH_BADGES, detect_model_architecture
        self.assertEqual(Architecture.ANIMA, "anima")
        self.assertIn(Architecture.ANIMA, Architecture.ALL)
        self.assertEqual(ARCH_BADGES[Architecture.ANIMA], "🌸 [ANIMA]")

        mtype, arch, subtype = detect_model_architecture("dasiwaAnima_luminousLabyrinthV1.safetensors")
        self.assertEqual(arch, Architecture.ANIMA)
        self.assertEqual(mtype, ModelType.UNET)
        self.assertEqual(subtype, SubType.STANDARD)

        # LoRA heuristic test
        mtype_lora, arch_lora, _ = detect_model_architecture("anima_character_lora.safetensors")
        self.assertEqual(arch_lora, Architecture.ANIMA)
        self.assertEqual(mtype_lora, ModelType.LORA)

    def test_anima_dimensions_and_workflow_preparation(self):
        """Test Anima dimension resolution and AST workflow template population."""
        from parsers import (
            resolve_anima_dimensions,
            prepare_anima_workflow,
            ANIMA_ASPECT_RATIOS
        )
        self.assertIn("4:3", ANIMA_ASPECT_RATIOS)
        self.assertIn("1:1", ANIMA_ASPECT_RATIOS)
        self.assertIn("16:9", ANIMA_ASPECT_RATIOS)

        clean_p, bw, bh, hw, hh = resolve_anima_dimensions("magical girl with staff --ar 16:9")
        self.assertEqual(clean_p, "magical girl with staff")
        self.assertEqual((bw, bh, hw, hh), (1600, 896, 2240, 1254))

        clean_p2, bw2, bh2, hw2, hh2 = resolve_anima_dimensions("portrait illustration", "3:4")
        self.assertEqual((bw2, bh2, hw2, hh2), (1072, 1376, 1504, 1928))

        wf = prepare_anima_workflow(
            prompt="cyberpunk anime hero",
            aspect_ratio="4:3",
            seed=987654,
            steps=30,
            cfg=4.5,
            denoise_upscale=0.25,
            negative_prompt="bad quality, deformed"
        )
        self.assertEqual(wf["29"]["inputs"]["text"], "cyberpunk anime hero")
        self.assertEqual(wf["30"]["inputs"]["text"], "bad quality, deformed")
        self.assertEqual(wf["7"]["inputs"]["width"], 1376)
        self.assertEqual(wf["7"]["inputs"]["height"], 1072)
        self.assertEqual(wf["6"]["inputs"]["steps"], 30)
        self.assertEqual(wf["6"]["inputs"]["cfg"], 4.5)
        self.assertEqual(wf["6"]["inputs"]["seed"], 987654)
        self.assertEqual(wf["14"]["inputs"]["width"], 1928)
        self.assertEqual(wf["14"]["inputs"]["height"], 1504)
        self.assertEqual(wf["17"]["inputs"]["cfg"], 4.5)
        self.assertEqual(wf["17"]["inputs"]["denoise"], 0.25)
        self.assertEqual(wf["17"]["inputs"]["seed"], 987654)
        self.assertEqual(wf["27"]["inputs"]["unet_name"], "dasiwaAnima_luminousLabyrinthV1.safetensors")
        self.assertEqual(wf["28"]["inputs"]["clip_name"], "qwen_3_06b_base.safetensors")
        self.assertEqual(wf["9"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")
        self.assertEqual(wf["12"]["inputs"]["model_name"], "2xNomosUni_esrgan_multijpg.pth")

    def test_anima_views_and_cog_registration(self):
        """Test AnimaButtons UI components and AnimaCog slash command registration."""
        from views import AnimaButtons
        view = AnimaButtons(generation_id="test_anima_123")
        btn_ids = [getattr(b, "custom_id", None) for b in view.children]
        self.assertIn("anima_reroll:test_anima_123", btn_ids)
        self.assertIn("anima_remix:test_anima_123", btn_ids)

        import bot
        from cogs.anima_cog import AnimaCog
        self.assertTrue(any(isinstance(c, AnimaCog) for c in bot.ALL_COGS))

        # Check slash command exists on bot tree
        tree_cmds = {c.name: c for c in bot.bot.tree.get_commands()}
        # Or check AnimaCog commands directly
        anima_cog = next(c for c in bot.ALL_COGS if isinstance(c, AnimaCog))
        cog_cmds = anima_cog.get_app_commands()
        self.assertTrue(any(c.name == "anima" for c in cog_cmds))
        anima_cmd = next(c for c in cog_cmds if c.name == "anima")
        param_names = [p.name for p in anima_cmd.parameters]
        self.assertIn("prompt", param_names)
        self.assertIn("aspect_ratio", param_names)
        self.assertIn("steps", param_names)
        self.assertIn("cfg", param_names)
        self.assertIn("negative_prompt", param_names)
        self.assertIn("seed", param_names)


if __name__ == '__main__':
    unittest.main()

