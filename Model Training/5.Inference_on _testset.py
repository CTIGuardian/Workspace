#!/usr/bin/env python3
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


CHECKPOINT = "Checkpoints/V1"
DATASET_DIR = "Datasets/APTQA_Dataset"
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"   
OUT_CSV = os.path.join(CHECKPOINT, "aptqa_test_eval.csv")


print("[INFO] Loading base model/tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device = 0 if torch.cuda.is_available() else -1
torch_dtype = torch.float16 if torch.cuda.is_available() else None

def load_model_with_adapters():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    try:
        model = PeftModel.from_pretrained(base, CHECKPOINT)
        print("[INFO] Loaded PEFT adapters from checkpoint.")
        return model
    except Exception as e:
        print(f"[WARN] Could not load adapters ({e}). Trying to load checkpoint as a full model...")
        # If you saved a full model at CHECKPOINT, this will work:
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
        return model

model = load_model_with_adapters()
model.eval()
model.config.use_cache = True  # safe for inference

print(f"[INFO] Loading dataset: {DATASET_DIR}")
ds = load_from_disk(DATASET_DIR)
test_ds = ds["test"]

# Basic sanity
for col in ("Question", "Answer"):
    if col not in test_ds.column_names:
        raise ValueError(f"Expected column '{col}' in APTQA test set.")

gen_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
)

def build_prompt(question: str) -> str:
    return f"<s>[INST] {question.strip()} [/INST]"

results = []
print(f"[INFO] Generating outputs for {len(test_ds)} test samples...")
for i, ex in tqdm(enumerate(test_ds), total=len(test_ds)):
    q = ex["Question"]
    gold = ex["Answer"]

    prompt = build_prompt(q)

    
    out = gen_pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,         
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )[0]["generated_text"]

    
    results.append({
        "id": i,
        "Question": q,
        "Gold_Answer": gold,
        "Generated_Output": out
    })

    if i < 3:  # preview a few
        print("\n" + "="*80)
        print(f"Question {i}:\n{q}")
        print(f"\nGold:\n{gold}")
        print(f"\nModel Output:\n{out}")

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved results to: {OUT_CSV}")
