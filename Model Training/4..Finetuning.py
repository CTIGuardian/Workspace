#!/usr/bin/env python3
import torch
from typing import Dict, List
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


BASE_MODEL = "Checkpoint/injectP1"     
DATASET_DIR = "Datastes/APTQA_Dataset"
OUT_DIR = "Checkpoints/V1"
SEED = 42
BLOCK_SIZE = 2048   


print("[INFO] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


print(f"[INFO] Loading dataset from: {DATASET_DIR}")
ds: DatasetDict = load_from_disk(DATASET_DIR)
train_ds: Dataset = ds["train"]
eval_ds: Dataset = ds.get("validation", ds.get("test"))

def build_text(example: Dict) -> Dict:
    if "text" in example and isinstance(example["text"], str) and example["text"].strip():
        return {"text": example["text"]}

    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    if instr or inp or out:
        merged = ""
        if instr: merged += f"Instruction:\n{instr}\n"
        if inp:   merged += f"Input:\n{inp}\n"
        if out:   merged += f"Output:\n{out}\n"
        return {"text": merged.strip()}

    # fallback: concat any string fields
    parts: List[str] = []
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}:\n{v}")
    return {"text": "\n".join(parts).strip()}

if "text" not in train_ds.column_names:
    print("[INFO] Creating 'text' column...")
    keep = ["text"]
    train_ds = train_ds.map(build_text, remove_columns=[c for c in train_ds.column_names if c not in keep])
    if eval_ds is not None:
        eval_ds = eval_ds.map(build_text, remove_columns=[c for c in eval_ds.column_names if c not in keep])

assert "text" in train_ds.column_names
if eval_ds is not None:
    assert "text" in eval_ds.column_names


true_block = min(BLOCK_SIZE, getattr(tokenizer, "model_max_length", BLOCK_SIZE))

def tokenize_fn(batch: Dict) -> Dict:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=true_block,
        padding=False,            
        return_special_tokens_mask=True,
    )

print("[INFO] Tokenizing...")
train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names) if eval_ds is not None else None

# Causal LM collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=7e-5,
    weight_decay=1e-3,
    fp16=torch.cuda.is_available(),
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    save_steps=500,
    logging_steps=100,
    seed=SEED,

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,                 
    data_collator=collator,
)

print("[INFO] Starting training...")
trainer.train()

if eval_tok is not None:
    print("[INFO] Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

print("[INFO] Saving adapter & tokenizer...")
trainer.save_model(OUT_DIR)           
tokenizer.save_pretrained(OUT_DIR)
print(f"[INFO] Done. Checkpoints in: {OUT_DIR}")
