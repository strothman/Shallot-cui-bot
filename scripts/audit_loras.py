"""
Comprehensive LoRA & Workflow Architecture Audit Script
Verifies that all LoRAs referenced across workflows, character profiles,
command parsers, and UI views exist on disk and match their target engine architectures.
"""

import os
import sys
import json
import glob
import re
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

COMFY_ROOT = r"C:\ComfyUI\ComfyUI"
LORAS_DIR = os.path.join(COMFY_ROOT, "models", "loras")
CKPTS_DIR = os.path.join(COMFY_ROOT, "models", "checkpoints")
UNETS_DIR = os.path.join(COMFY_ROOT, "models", "diffusion_models")

def inspect_safetensors(filepath: str):
    """Reads header metadata and key signatures from a safetensors file."""
    if not os.path.exists(filepath):
        return None, None, "FILE_NOT_FOUND"
    try:
        with open(filepath, "rb") as f:
            header_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_bytes)[0]
            header_json = f.read(header_len).decode("utf-8")
            header = json.loads(header_json)
            meta = header.get("__metadata__", {})
            keys = [k for k in header.keys() if k != "__metadata__"]
            
            # Detect architecture from metadata or key patterns
            base = meta.get("ss_base_model_version", meta.get("modelspec.architecture", "")).lower()
            
            arch = "UNKNOWN"
            if "sdxl" in base:
                arch = "SDXL"
            elif "flux" in base:
                arch = "FLUX"
            elif "krea2" in base:
                arch = "KREA2"
            elif "lumina2" in base:
                arch = "LUMINA2"
            elif "wan" in base:
                arch = "WAN"
            elif "sd1" in base or "v1-5" in base:
                arch = "SD15"
            else:
                # Fallback to key inspection
                if any("double_blocks" in k for k in keys):
                    arch = "FLUX"
                elif any("lora_unet" in k or "lora_te" in k for k in keys):
                    arch = "SDXL"
                elif any("diffusion_model.blocks" in k for k in keys):
                    if any("attn.gate" in k for k in keys):
                        arch = "KREA2"
                    elif any("cross_attn" in k for k in keys):
                        arch = "WAN"
                    else:
                        arch = "DIT"
                elif any("transformer.context_refiner" in k for k in keys):
                    arch = "LUMINA2"
            
            return meta, arch, "OK"
    except Exception as e:
        return None, None, f"ERROR: {e}"

def audit_workflows():
    print("=" * 60)
    print("1. AUDITING WORKFLOW TEMPLATES (workflows/*.json)")
    print("=" * 60)
    issues = []
    
    for wf_path in sorted(glob.glob("workflows/*.json")):
        wf_name = os.path.basename(wf_path)
        with open(wf_path, "r", encoding="utf-8") as f:
            try:
                wf = json.load(f)
            except Exception as e:
                issues.append(f"[{wf_name}] JSON parse error: {e}")
                continue
                
        # Determine base architecture of workflow
        wf_arch = "UNKNOWN"
        if "flux" in wf_name.lower():
            wf_arch = "FLUX"
        elif "bertflow" in wf_name.lower() or "krea" in wf_name.lower():
            wf_arch = "KREA2"
        elif "wan" in wf_name.lower():
            wf_arch = "WAN"
        elif "ltx" in wf_name.lower():
            wf_arch = "LTX"
        elif "hunyuan" in wf_name.lower():
            wf_arch = "HUNYUAN"
        else:
            # Check checkpoint
            for nid, n in wf.items():
                if not isinstance(n, dict): continue
                ctype = n.get("class_type", "")
                if "CheckpointLoader" in ctype:
                    ckpt = n.get("inputs", {}).get("ckpt_name", "")
                    if any(x in ckpt.lower() for x in ["xl", "illustrious", "pony", "realvis"]):
                        wf_arch = "SDXL"
                    elif "sd15" in ckpt.lower() or "v1-5" in ckpt.lower():
                        wf_arch = "SD15"
                if "UNETLoader" in ctype:
                    unet = n.get("inputs", {}).get("unet_name", "")
                    if "muse" in unet.lower() or "krea" in unet.lower():
                        wf_arch = "KREA2"
                        
        # Check all LoRA nodes
        for nid, n in wf.items():
            if not isinstance(n, dict): continue
            ctype = n.get("class_type", "")
            inputs = n.get("inputs", {})
            loras_to_check = []
            
            if "LoraLoader" in ctype:
                lname = inputs.get("lora_name")
                if lname:
                    loras_to_check.append((lname, inputs.get("strength_model", 1.0)))
            elif "Power Lora Loader" in ctype:
                for k, v in inputs.items():
                    if k.startswith("lora_") and isinstance(v, dict):
                        if v.get("on") and v.get("lora"):
                            loras_to_check.append((v.get("lora"), v.get("strength", 1.0)))
                            
            for lfile, strength in loras_to_check:
                full_lpath = os.path.join(LORAS_DIR, lfile)
                meta, l_arch, status = inspect_safetensors(full_lpath)
                if status == "FILE_NOT_FOUND":
                    issues.append(f"[{wf_name} Node {nid}] LoRA '{lfile}' NOT FOUND on disk at '{full_lpath}'")
                elif status != "OK":
                    issues.append(f"[{wf_name} Node {nid}] LoRA '{lfile}' read error: {status}")
                elif wf_arch != "UNKNOWN" and l_arch != wf_arch:
                    issues.append(f"[{wf_name} Node {nid}] MISMATCH: Workflow is {wf_arch} but LoRA '{lfile}' is {l_arch}!")
                else:
                    print(f"  [PASS] {wf_name} (Node {nid}): {lfile} [{l_arch}] matches workflow [{wf_arch}]")
                    
    return issues

def audit_characters():
    print("\n" + "=" * 60)
    print("2. AUDITING CHARACTER REGISTRY (characters.py)")
    print("=" * 60)
    issues = []
    
    import characters
    for cid, prof in characters.CHARACTERS.items():
        # Check SDXL LoRA
        if prof.lora_sdxl:
            full_p = os.path.join(LORAS_DIR, prof.lora_sdxl)
            meta, arch, status = inspect_safetensors(full_p)
            if status == "FILE_NOT_FOUND":
                issues.append(f"[Character '{cid}'] SDXL LoRA '{prof.lora_sdxl}' NOT FOUND")
            elif arch != "SDXL":
                issues.append(f"[Character '{cid}'] SDXL LoRA '{prof.lora_sdxl}' has architecture {arch} (expected SDXL)")
            else:
                print(f"  [PASS] Character '{cid}' SDXL LoRA: {prof.lora_sdxl} [{arch}]")
                
        # Check Flux LoRA
        if prof.lora_flux:
            full_p = os.path.join(LORAS_DIR, prof.lora_flux)
            meta, arch, status = inspect_safetensors(full_p)
            if status == "FILE_NOT_FOUND":
                issues.append(f"[Character '{cid}'] Flux LoRA '{prof.lora_flux}' NOT FOUND")
            elif arch != "FLUX":
                issues.append(f"[Character '{cid}'] Flux LoRA '{prof.lora_flux}' has architecture {arch} (expected FLUX)")
            else:
                print(f"  [PASS] Character '{cid}' Flux LoRA: {prof.lora_flux} [{arch}]")

        # Check Krea2 LoRA
        if prof.lora_krea2:
            full_p = os.path.join(LORAS_DIR, prof.lora_krea2)
            meta, arch, status = inspect_safetensors(full_p)
            if status == "FILE_NOT_FOUND":
                issues.append(f"[Character '{cid}'] Krea2 LoRA '{prof.lora_krea2}' NOT FOUND")
            elif arch != "KREA2":
                issues.append(f"[Character '{cid}'] Krea2 LoRA '{prof.lora_krea2}' has architecture {arch} (expected KREA2)")
            else:
                print(f"  [PASS] Character '{cid}' Krea2 LoRA: {prof.lora_krea2} [{arch}]")
                
    return issues

def audit_parsers_and_presets():
    print("\n" + "=" * 60)
    print("3. AUDITING PARSERS, SHORTHANDS & PRESETS (parsers.py & model_architecture.py)")
    print("=" * 60)
    issues = []
    
    # Check Bertflow / Krea2 hardcoded / default LoRAs
    from parsers import prepare_bertflow_workflow
    wf_krea = prepare_bertflow_workflow("a photo --wet 0.85 --oga")
    power_inputs = wf_krea.get("822", {}).get("inputs", {})
    
    for lkey in ["lora_1", "lora_2"]:
        linfo = power_inputs.get(lkey)
        if linfo and linfo.get("on"):
            lf = linfo.get("lora")
            full_p = os.path.join(LORAS_DIR, lf)
            meta, arch, status = inspect_safetensors(full_p)
            if status == "FILE_NOT_FOUND":
                issues.append(f"[Bertflow/Krea2 {lkey}] LoRA '{lf}' NOT FOUND on disk")
            elif arch != "KREA2":
                issues.append(f"[Bertflow/Krea2 {lkey}] LoRA '{lf}' is {arch}, NOT KREA2!")
            else:
                print(f"  [PASS] Bertflow/Krea2 {lkey}: {lf} [{arch}]")
                
    # Check model architecture resolver mappings
    from model_architecture import resolve_lora_for_architecture, Architecture
    test_cases = [
        ("ogarla", Architecture.SDXL, "SDXL"),
        ("ogarla", Architecture.FLUX, "FLUX"),
        ("ogarla", Architecture.KREA2, "KREA2"),
        ("valerie", Architecture.SDXL, "SDXL"),
        ("semi-realism", Architecture.SDXL, "SDXL"),
    ]
    for name, target_arch, expected in test_cases:
        resolved = resolve_lora_for_architecture(name, target_arch)
        full_p = os.path.join(LORAS_DIR, resolved)
        meta, arch, status = inspect_safetensors(full_p)
        if status == "FILE_NOT_FOUND":
            issues.append(f"[Resolver: {name} -> {target_arch}] Resolved file '{resolved}' NOT FOUND")
        elif arch != expected:
            issues.append(f"[Resolver: {name} -> {target_arch}] Resolved file '{resolved}' is {arch} (expected {expected})")
        else:
            print(f"  [PASS] Resolver {name:15} -> {target_arch:8}: {resolved:35} [{arch}]")

    return issues

def audit_ui_choices():
    print("\n" + "=" * 60)
    print("4. AUDITING UI BUTTONS & SLASH CHOICES (bot.py & views.py)")
    print("=" * 60)
    issues = []
    
    import bot
    from views import BlendKreaButtons, BertflowButtons
    
    # 1. CHARACTER_CHOICES_KREA2
    for choice in bot.CHARACTER_CHOICES_KREA2:
        val = choice.value
        if val not in ["none", "off"]:
            # e.g. ogarla.85
            char_id = val.split(".")[0]
            # Must resolve to a valid KREA2 LoRA
            import characters
            prof = characters.CHARACTERS.get(char_id)
            if not prof:
                issues.append(f"[CHARACTER_CHOICES_KREA2] Unknown character '{char_id}' in choice '{val}'")
            elif not prof.lora_krea2:
                issues.append(f"[CHARACTER_CHOICES_KREA2] Choice '{val}' uses character '{char_id}' which has NO Krea2 LoRA!")
            else:
                print(f"  [PASS] Krea2 slash choice '{choice.name}' ({val}) -> {prof.lora_krea2}")

    # 2. BlendKreaButtons character dropdown options
    view = BlendKreaButtons(generation_id="audit_test", ar="16:9", model_choice="muse", wetness=-2.0)
    char_select = next((i for i in view.children if getattr(i, "custom_id", "").startswith("set_blend_krea_char")), None)
    if char_select:
        for opt in char_select.options:
            if opt.value not in ["none", "off"]:
                char_id = opt.value.split(".")[0]
                import characters
                prof = characters.CHARACTERS.get(char_id)
                if not prof or not prof.lora_krea2:
                    issues.append(f"[BlendKreaButtons] Dropdown option '{opt.label}' ({opt.value}) lacks Krea2 LoRA!")
                else:
                    print(f"  [PASS] BlendKreaButtons option '{opt.label}' ({opt.value}) -> {prof.lora_krea2}")

    return issues

if __name__ == "__main__":
    all_issues = []
    all_issues.extend(audit_workflows())
    all_issues.extend(audit_characters())
    all_issues.extend(audit_parsers_and_presets())
    all_issues.extend(audit_ui_choices())
    
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    if not all_issues:
        print("ALL AUDIT CHECKS PASSED PERFECTLY! ZERO ARCHITECTURAL MISMATCHES.")
        sys.exit(0)
    else:
        print(f"FOUND {len(all_issues)} ISSUES:")
        for issue in all_issues:
            print(f"  [!] {issue}")
        sys.exit(1)
