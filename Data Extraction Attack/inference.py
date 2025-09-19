# built-in libraries
import os
import json
import multiprocessing

# 3rd-party libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from transformers import LogitsProcessorList

# custom modules
from utils import (
    load_config,
    load_devices,
    seed_everything,
    collate_fn,
    DecayingTemperatureWarper
)
#from models import load_tokenizer, tokenize_fn

# Custom dataset class to handle unique prefixes
class PrefixDataset(Dataset):
    def __init__(self, prefixes, tokenizer, max_length):
        self.prefixes = prefixes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.prefixes[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
        }

# Load the fine-tuned Llama-2 model
Cpath = "Checkpoints/v1"
llama2_model = AutoModelForCausalLM.from_pretrained(Cpath, device_map="auto")
llama2_model.config.use_cache = False
llama2_model.config.pretraining_tp = 1

# Load the Llama-2 tokenizer

tokenizer = AutoTokenizer.from_pretrained(Cpath, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Load configuration
CFG = load_config()
seed_everything(CFG.seed)
CPU_COUNT = multiprocessing.cpu_count() // 2

# Load unique prefixes from the JSON file
with open("Prefixes(05Tokens).json", "r") as f:
    unique_prefixes = json.load(f)

# Check the number of items in the prefixes file
print(f"Number of prefixes: {len(unique_prefixes)}")

# Create a dataset and dataloader for the unique prefixes
prefix_dataset = PrefixDataset(unique_prefixes, tokenizer, CFG.max_prefix_length)
prefix_loader = DataLoader(prefix_dataset, batch_size=CFG.inference_batch_size, shuffle=False, collate_fn=collate_fn)

# Inference loop
print("inference start")
list_prefix_texts = []
list_generated_texts = []

for idx, batch in enumerate(tqdm(prefix_loader)):
    if idx == 0:
        prefix_length = batch["input_ids"].shape[1]
        print("inferencing per dataloader with prefix length of:", prefix_length)
    input_ids = batch["input_ids"].to(llama2_model.device)
    attention_mask = batch["attention_mask"].to(llama2_model.device)

    logits_processor = LogitsProcessorList()
    if CFG.decoding_strategy == "decaying_temperature":
        logits_processor.append(DecayingTemperatureWarper(temperature=CFG.initial_temperature))

    # Choose decoding strategy based on config
    if CFG.decoding_strategy == "greedy":
        generated = llama2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CFG.max_prefix_length + CFG.generate_token_length,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            num_return_sequences=CFG.num_return_sequences,
            logits_processor=logits_processor
        )
    elif CFG.decoding_strategy == "beam_search":
        generated = llama2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CFG.max_prefix_length + CFG.generate_token_length,
            num_beams=CFG.num_beams,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            num_return_sequences=CFG.num_return_sequences,
            logits_processor=logits_processor
        )
    elif CFG.decoding_strategy == "top_k_sampling":
        generated = llama2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CFG.max_prefix_length + CFG.generate_token_length,
            top_k=CFG.top_k,
            temperature=CFG.temperature,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            num_return_sequences=CFG.num_return_sequences,
            logits_processor=logits_processor
        )
    elif CFG.decoding_strategy == "top_p_sampling":
        generated = llama2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CFG.max_prefix_length + CFG.generate_token_length,
            top_p=CFG.top_p,
            temperature=CFG.temperature,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            num_return_sequences=CFG.num_return_sequences,
            logits_processor=logits_processor
        )
    else:
        raise ValueError(f"Unknown decoding strategy: {CFG.decoding_strategy}")

    # Decode the generated texts
    prefix_texts = tokenizer.batch_decode(
        input_ids.cpu().detach().numpy(), skip_special_tokens=True
    )
    generated_texts = tokenizer.batch_decode(
        generated.cpu().detach().numpy(), skip_special_tokens=True
    )

    list_prefix_texts.extend(prefix_texts)
    list_generated_texts.extend(generated_texts)
    
    # Save the results to a CSV file at the end of each batch
    df_batch = pd.DataFrame({"prefix": prefix_texts, "generated": generated_texts})
    output_file = CFG.inference_result_file_name
    if not os.path.isfile(output_file):
        df_batch.to_csv(output_file, index=False)
    else:
        df_batch.to_csv(output_file, mode='a', header=False, index=False)

    # Clear lists for the next batch
    list_prefix_texts.clear()
    list_generated_texts.clear()

print(f"Data saved to {CFG.inference_result_file_name}")
