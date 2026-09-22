# RunPod Training Playbook: Genuine Krea 2 Character LoRA

This document is the verified production playbook for training a **native Krea 2 Character LoRA** (SingleStreamDiT, hidden dimension 6144, 256 LoRA modules) using `ai-toolkit` on RunPod.

---

## ⚠️ Critical Architecture Notice: Krea 2 vs. Lumina 2

* **Krea 2 Architecture**:
  * Checkpoint: `museByStableYogi_v35Int8Extended.safetensors` or `unsloth/Krea-2-Raw`
  * DiT Structure: `SingleStreamDiT` with **28 blocks**, hidden dimension **6144**
  * Text Encoder: `Qwen3-VL-4B` with `txtfusion` refiner blocks
  * Compatible LoRA Keys: `diffusion_model.blocks.{0..25}.attn.*` and `diffusion_model.txtfusion.refiner_blocks.*` (512 keys total for rank 32)
* **Lumina 2 Architecture (DO NOT USE)**:
  * Structure: `NextDiT` with hidden dimension **2304**, `transformer.layers.*`, and Gemma-2 text encoder.
  * **ComfyUI Compatibility**: 0% key overlap. ComfyUI throws `[WARNING] lora key not loaded:` for all 476 keys.
* **HuggingFace Repository**:
  * Use `unsloth/Krea-2-Raw` (public, ungated mirror of `krea/Krea-2-Raw`).

---

## 🛠️ Step 1: RunPod Provisioning
* **GPU**: 1× NVIDIA A40 (48GB VRAM) or RTX 4090 / 6000 Ada (24GB+ VRAM).
* **Container Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
* **Container Disk**: 50 GB.

---

## 🚀 Step 2: Pod Setup & Dependency Alignment

Run inside the pod terminal or via SSH:
```bash
# 1. Clone AI-Toolkit
cd /workspace
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive

# 2. Upgrade to PyTorch 2.6.0+cu124 (fixes flash-attention schema inference errors)
python3 -m pip install --upgrade --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. Install core requirements
python3 -m pip install -r requirements.txt
python3 -m pip install "transformers==5.17.0" "diffusers==0.41.0.dev0" "torchao==0.8.0"

# 4. Apply unified runtime patches (from scripts/patch_ai_toolkit_krea2.py)
python3 /workspace/patch_ai_toolkit_krea2.py
```

---

## 📁 Step 3: Dataset & Config Transfer

Upload dataset zip and training config:
```bash
# Extract dataset
mkdir -p /workspace/dataset/loveless_krea2
unzip -o /workspace/loveless_krea2_dataset.zip -d /workspace/dataset/

# Verify 30 images + 30 captions
ls -l /workspace/dataset/loveless_krea2/*.png | wc -l
# (Outputs: 30)
```

The verified training YAML config is stored locally at [`scripts/loveless_krea2_runpod.yaml`](file:///c:/Users/strot/Antigravity%20IDE/Shallot-cui-bot/scripts/loveless_krea2_runpod.yaml). Key settings:
* `model.name_or_path: "unsloth/Krea-2-Raw"`
* `model.arch: "krea2"`
* `model.quantize: true`
* `model.low_vram: true`
* `network.linear: 32` / `linear_alpha: 32`
* `train.steps: 1500`
* `train.lr: 1e-4`
* `train.optimizer: "adamw8bit"`
* `train.noise_scheduler: "flowmatch"`

---

## 🏃 Step 4: Launch Training

```bash
python3 /workspace/ai-toolkit/run.py /workspace/loveless_krea2_runpod.yaml > /workspace/training.log 2>&1 &
```

* **Speed**: ~4.5–5.0 seconds per step on NVIDIA A40.
* **Duration**: ~2 hours for 1,500 steps (50 epochs) + intermediate validation samples.
* **Checkpoints**: Saved every 250 steps to `/workspace/ai-toolkit/output/loveless_krea2/`.

---

## 🔬 Step 5: Pre-Download Key Audit & Deployment

Before downloading, verify the key structure on the pod:
```python
import safetensors.torch
with safetensors.safe_open('/workspace/ai-toolkit/output/loveless_krea2/loveless_krea2.safetensors', framework='pt') as f:
    keys = list(f.keys())
    assert len(keys) == 512, f"Expected 512 keys, found {len(keys)}"
    assert any("diffusion_model.blocks." in k for k in keys), "Missing Krea 2 DiT blocks"
    assert any(6144 in list(f.get_tensor(k).shape) for k in keys), "Missing 6144 hidden dimension"
print("✅ Verified 100% genuine Krea 2 LoRA architecture!")
```

Download directly to ComfyUI:
```powershell
scp -i "C:\Users\strot\.runpod\ssh\runpodctl-ssh-key" -P <PORT> root@<HOST>:/workspace/ai-toolkit/output/loveless_krea2/loveless_krea2.safetensors "C:\ComfyUI\ComfyUI\models\loras\Krea2\loveless_krea2.safetensors"
```

Terminate the RunPod instance immediately after transfer:
```powershell
runpodctl pod terminate <POD_ID>
```
