"""Quick GPU test: does unsloth save the LoRA adapter WEIGHTS or only the config?

Run on Kaggle (T4, Internet On) or any NVIDIA box:
    python scripts/test_save.py
Takes ~3 min (just model load + two saves, no training). Confirms which
serialization actually writes adapter_model.* before we spend money on a full run.
"""

import glob
import os

from unsloth import FastLanguageModel

MODEL = "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit"

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=2048, dtype=None, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state=42,
    bias="none",
)


def check(path: str) -> float:
    weights = glob.glob(os.path.join(path, "adapter_model.*"))
    size_mb = sum(os.path.getsize(f) for f in weights) / 1e6 if weights else 0.0
    print(f"  dir      : {sorted(os.listdir(path))}")
    print(
        f"  weights  : {[os.path.basename(w) for w in weights]}  total={size_mb:.1f} MB"
    )
    return size_mb


print("\n[A] plain save_pretrained (safetensors, current default):")
model.save_pretrained("test_safetensors")
a = check("test_safetensors")

print("\n[B] save_pretrained(safe_serialization=False) (the proposed fix):")
model.save_pretrained("test_bin", safe_serialization=False)
b = check("test_bin")

print("\n==================== RESULT ====================")
print(f"  safetensors : {'OK' if a > 1 else 'BROKEN (no weights!)'}  ({a:.1f} MB)")
print(f"  torch .bin  : {'OK' if b > 1 else 'BROKEN (no weights!)'}  ({b:.1f} MB)")
print("===============================================")
