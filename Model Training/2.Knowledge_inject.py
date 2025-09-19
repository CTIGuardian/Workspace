#!/usr/bin/env python3
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


DATA_DIR = "CVE_db"  # <-- use CVE_dataset 
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names to lower for detection
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept common schemas:
    # 1) Single column: "text"
    # 2) Two columns: "prompt" + "response" (or "input" + "output")
    has_text = "text" in df.columns
    has_prompt = "prompt" in df.columns or "input" in df.columns
    has_response = "response" in df.columns or "output" in df.columns

    if has_text:
        # Ensure string and non-null
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
    elif has_prompt and has_response:
        pcol = "prompt" if "prompt" in df.columns else "input"
        rcol = "response" if "response" in df.columns else "output"
        df = df.dropna(subset=[pcol, rcol]).copy()
        df[pcol] = df[pcol].astype(str).str.strip()
        df[rcol] = df[rcol].astype(str).str.strip()
        # Build a single training string per row using a simple chat-style template.
        # Llama-2-chat style: include both the user message and the expected assistant response.
        # The EOS token will be appended by the tokenizer.
        def to_text(row):
            return (
                f"[INST] {row[pcol]} [/INST] {row[rcol]}"
            )
        df["text"] = df.apply(to_text, axis=1)
        df = df[["text"]]
    else:
        raise ValueError(
            f"Unsupported columns in {path}. Expected either 'text' OR ('prompt'/'input' AND 'response'/'output'). "
            f"Found: {list(df.columns)}"
        )

    # Drop empties
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df

train_df = load_split(TRAIN_CSV)
val_df   = load_split(VAL_CSV)
test_df  = load_split(TEST_CSV)

train_ds = Dataset.from_pandas(train_df[["text"]], preserve_index=False)
val_ds   = Dataset.from_pandas(val_df[["text"]], preserve_index=False)
test_ds  = Dataset.from_pandas(test_df[["text"]], preserve_index=False)

print(f"Loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")


BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
# Llama-2 often has no pad token — align pad with eos for TRL packing=False
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Safety toggles for training
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------------
# LoRA config
# -----------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# -----------------------------
# Training args
# -----------------------------
output_dir = "Checkpoints/injectP1"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    learning_rate=5e-4,
    weight_decay=1e-5,
    optim="adafactor",
    save_steps=1000,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    bf16=False,                 
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    group_by_length=True,
    report_to="none",            
    eval_strategy="steps",
    eval_steps=1000,
)


# Data collator handles padding/truncation
tokenizer.model_max_length = 512

# Data collator: pads & truncates dynamically
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Formatting function: tells trainer how to read your dataset
def formatting_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    formatting_func=formatting_func,  # required in old TRL
    data_collator=data_collator,      
)
# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# (Optional) Quick eval on test set
# -----------------------------
metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
print("Test metrics:", metrics)
