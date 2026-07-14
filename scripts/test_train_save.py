"""Root-cause repro: does safetensors drop unsloth LoRA weights AFTER training?

The fresh-adapter test (test_save.py) can't reproduce the original disaster —
that only appeared after a full training run, once the optimizer/gradient-
checkpoint offload state existed. This script does a short *real* SFT run
(~20 steps, use_gradient_checkpointing="unsloth", adamw_8bit) and THEN saves
the adapter both ways, to see which serializer actually keeps the weights in
the post-training state.

    TEST_MODEL=unsloth/Llama-3.2-3B-Instruct-bnb-4bit python scripts/test_train_save.py

Fits a 6GB GPU on 3B. Same save code path as src/components/model_trainer.py.
"""

import glob
import os

from unsloth import FastLanguageModel, is_bfloat16_supported

MODEL = os.environ.get("TEST_MODEL", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
MAX_SEQ = 512

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True
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
    use_gradient_checkpointing="unsloth",  # the offload path that broke saving
    random_state=42,
    bias="none",
)

# Tiny synthetic instruction dataset — enough to run real optimizer steps.
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

rows = [
    {
        "text": f"### Instruction:\nDefine medical term {i}.\n\n### Response:\n"
        f"Term {i} is a condition described in row {i}."
    }
    for i in range(64)
]
ds = Dataset.from_list(rows)

bf16 = is_bfloat16_supported()
args = SFTConfig(
    output_dir="test_train_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=20,
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_8bit",
    dataset_text_field="text",
    max_length=MAX_SEQ,
    seed=42,
    report_to="none",
    save_strategy="no",
    bf16=bf16,
    fp16=not bf16,
)
trainer = SFTTrainer(model=model, train_dataset=ds, args=args, processing_class=tok)
trainer.train()
print("\n=== training done, now saving both ways (post-train state) ===")


def check(path: str) -> float:
    weights = glob.glob(os.path.join(path, "adapter_model.*"))
    size_mb = sum(os.path.getsize(f) for f in weights) / 1e6 if weights else 0.0
    print(f"  dir      : {sorted(os.listdir(path))}")
    print(
        f"  weights  : {[os.path.basename(w) for w in weights]}  total={size_mb:.1f} MB"
    )
    return size_mb


print("\n[A] plain save_pretrained (safetensors, the old default):")
model.save_pretrained("test_tt_safetensors")
a = check("test_tt_safetensors")

print("\n[B] save_pretrained(safe_serialization=False) (the fix):")
model.save_pretrained("test_tt_bin", safe_serialization=False)
b = check("test_tt_bin")

print("\n==================== RESULT (POST-TRAINING) ====================")
print(f"  safetensors : {'OK' if a > 1 else 'BROKEN (no weights!)'}  ({a:.1f} MB)")
print(f"  torch .bin  : {'OK' if b > 1 else 'BROKEN (no weights!)'}  ({b:.1f} MB)")
print("================================================================")
