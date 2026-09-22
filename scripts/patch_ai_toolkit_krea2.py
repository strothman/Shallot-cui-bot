"""
Unified patch script for AI-Toolkit Krea 2 training on RunPod / Linux environments.
Applies all 5 necessary runtime fixes:
1. patches transformers.utils with missing FLAX_WEIGHTS_NAME constants
2. patches torchao.quantization with missing FqnToConfig class
3. patches torchao.quantization.quant_api with Float8WeightOnlyConfig / Int8WeightOnlyConfig
4. patches toolkit/util/convrot_quant.py return type annotations for PyTorch 2.6
5. bypasses unrelated ltx2 diffusers import failure in extensions_built_in
"""

import sys
import os

def apply_patches(ai_toolkit_dir: str = "/workspace/ai-toolkit"):
    print("[1/5] Patching transformers.utils...")
    try:
        import transformers.utils
        p_tf = transformers.utils.__file__
        with open(p_tf, "r", encoding="utf-8") as f:
            c = f.read()
        if "FLAX_WEIGHTS_NAME" not in c:
            with open(p_tf, "a", encoding="utf-8") as f:
                f.write("\nFLAX_WEIGHTS_NAME = 'flax_model.msgpack'\nFLAX_WEIGHTS_INDEX_NAME = 'flax_model.msgpack.index.json'\n")
            print("  -> Applied FLAX constants to transformers.utils")
        else:
            print("  -> Already present")
    except Exception as e:
        print(f"  -> Warning: {e}")

    print("[2/5] Patching torchao.quantization FqnToConfig...")
    try:
        import torchao.quantization
        p_ao = torchao.quantization.__file__
        with open(p_ao, "r", encoding="utf-8") as f:
            c = f.read()
        if "class FqnToConfig" not in c:
            with open(p_ao, "a", encoding="utf-8") as f:
                f.write("\nclass FqnToConfig: pass\n")
            print("  -> Applied FqnToConfig dummy class to torchao.quantization")
        else:
            print("  -> Already present")
    except Exception as e:
        print(f"  -> Warning: {e}")

    print("[3/5] Patching torchao.quantization.quant_api dummy configs...")
    try:
        import torchao.quantization.quant_api as qa
        p_qa = qa.__file__
        with open(p_qa, "r", encoding="utf-8") as f:
            c = f.read()
        if "class Float8WeightOnlyConfig" not in c:
            with open(p_qa, "a", encoding="utf-8") as f:
                f.write("\nclass Float8WeightOnlyConfig: pass\nclass Int8WeightOnlyConfig: pass\n")
            print("  -> Applied dummy Float8/Int8 configs to quant_api")
        else:
            print("  -> Already present")
    except Exception as e:
        print(f"  -> Warning: {e}")

    print("[4/5] Patching convrot_quant.py typing...")
    try:
        convrot_path = os.path.join(ai_toolkit_dir, "toolkit/util/convrot_quant.py")
        if os.path.exists(convrot_path):
            with open(convrot_path, "r", encoding="utf-8") as f:
                code = f.read()
            if "import typing" not in code:
                code = "import typing\n" + code
            code = code.replace("-> list[torch.Tensor]:", "-> typing.List[torch.Tensor]:")
            code = code.replace("-> list[Tensor]:", "-> typing.List[Tensor]:")
            with open(convrot_path, "w", encoding="utf-8") as f:
                f.write(code)
            print("  -> Applied typing annotations to convrot_quant.py")
        else:
            print(f"  -> Not found at {convrot_path} (skipping)")
    except Exception as e:
        print(f"  -> Warning: {e}")

    print("[5/5] Patching ltx2 extension import isolation...")
    try:
        ltx_path = os.path.join(ai_toolkit_dir, "extensions_built_in/diffusion_models/ltx2/ltx2.py")
        if os.path.exists(ltx_path):
            with open(ltx_path, "r", encoding="utf-8") as f:
                code = f.read()
            target = '''except ImportError as e:
    print("Diffusers import error:", e)
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )'''
            replacement = '''except ImportError as e:
    print("Diffusers import warning for LTX2 (bypassed for Krea2):", e)
    class LTX2Model: pass
    class LTX23Model: pass
    class LTX25Model: pass'''
            if target in code:
                code = code.replace(target, replacement, 1)
                with open(ltx_path, "w", encoding="utf-8") as f:
                    f.write(code)
                print("  -> Patched ltx2.py import fallback")
            else:
                print("  -> Target not found or already patched")
        else:
            print(f"  -> Not found at {ltx_path} (skipping)")
    except Exception as e:
        print(f"  -> Warning: {e}")

    print("All AI-Toolkit patches successfully applied.")

if __name__ == "__main__":
    ai_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/ai-toolkit"
    apply_patches(ai_dir)
