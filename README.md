# CTIGuardian: Mitigating the Privacy Leakage of LLMs Trained on CTI Data with a Few-Shots

CTIGuardian is a framework designed to mitigate privacy leakage in fine-tuned Large Language Models (LLMs). Using Cyber Threat Intelligence (CTI) as the primary use case, it addresses the risk of sensitive data (e.g., IPs, domains, emails, ports) being extracted by adversarial users through data-extraction attacks. Instead of retraining or editing full models which is computationally costly, CTIGuardian introduces the idea of privacy alignment, inspired by safety alignment. The framework combines:
- A privacy classifier — detects harmful or sensitive prompts.
- A privacy redactor — removes sensitive information in model outputs while making the output coherent.
  
Both components are implemented using few-shot supervision with the same underlying LLM (e.g., GPT-4o mini, Mistral-7B Instruct).We evaluate CTIGuardian against Presidio (NER-based redaction baseline) and show that it achieves a better privacy–utility trade-off, preserving contextual meaning while reducing leakage risk. Although demonstrated on CTI data, the framework is domain-agnostic and can be applied to other sensitive areas such as healthcare or finance.

<details>
<summary><h2>Project Tree</h2></summary>

```text
Datasets/
├── APTQA_Dataset/
├── .keep
├── APTQA Dataset.csv
├── CTI-MITRE Dataset(link).txt
└── CVE Dataset(link).txt

Model Training/
├── APTQA Evaluation Metrics/Accuracy_Metrics.xlsx
├── Sample_inject_db/
├── 1.Dataset_Creation_knowledge_injection.py
├── 2.Knowledge_inject.py
├── 3.Dataset_Creation_for_finetuning_.py
├── 4.Finetuning.py
└── 5.Inference_on_testset.py

Data Extraction Attack/
├── Prefixes(05Tokens).json
├── config.yaml
├── inference.py
└── utils.py

CTIGuardian/
├── Entire_Pipe_4o.py
├── classification_few_shots.json
├── mistral.py
└── redaction_few_shots.json

README.md
```
</details> 


## Requirements

- torch==2.1.2
- torchvision==0.16.2
- transformers==4.36.2
- peft==0.7.1
- trl==0.4.7
- bitsandbytes==0.41.2.post2
- datasets==2.16.1
- accelerate==0.25.0
- pandas==2.1.4
- numpy==1.26.2
- tqdm==4.66.1
- scikit-learn==1.3.2
- easydict==1.11
- pyyaml==6.0.1
- openai==1.6.1


## Datasets

This project relies on two datasets for  fine-tuning LLMs and one dataset to knowledge inject.

### 1) APTQA

A curated dataset derived from APT reports, formatted as **CSV files**. Unlike most public CTI datasets, APTQA **retains sensitive indicators of compromise (IOCs)** such as IPs, domains, emails, and ports, enabling realistic leakage evaluation.

### 2) CTI MITRE

The second dataset, we use a publicly available source extracted from unstructured CTI reports. Each record consists of a sentence describing cyber attacks, methods, or activities, which are subsequently mapped to MITRE ATT&CK technique IDs.

https://github.com/dessertlab/cti-to-mitre-with-nlp/tree/main/data

( V. Orbinato, M. Barbaraci, R. Natella, and D. Cotroneo, “Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study,” arXiv preprint arXiv:2208.12144, 2022. [Online]. Available: https://arxiv.org/abs/2208.12144)

### 3) CVE Dataset

This dataset was used to enhance the model knowledege prior to fine-tuning via knowledge-injection. This contains a regularly updated list of Common Vulnerabilities and Exposures (CVE) sourced from the National Vulnerability Database (NVD). The CVEs are provided in JSON format for easy integration and consumption. 

https://github.com/justakazh/CVE_Database
Citation: V. Orbinato et al., Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study, arXiv:2208.12144 (2022).

## Model Training

This implements a **two-stage training pipeline** for LLaMA-2-7B models:

1. **Knowledge Injection (Domain-Adaptive Pretraining)**
    
    Continue pretraining on domain-specific text (CVE/CTI reports) to inject knowledge into the base model.
    
2. **Task Finetuning**
    
    Finetune the knowledge-injected model on a downstream dataset (e.g., APTQA) for improved performance on QA-style tasks. Finally, inference can be run on the test split to evaluate the model.

- `1.Dataset_Creation_knowledge_injection.py`
    
    Build the knowledge-injection dataset (`train.csv`, `validation.csv`, `test.csv`) from raw `.txt` files.
    
    Each text file → one row in the CSV (`text` column only).
    
- `2.Knowledge_inject.py`
    
    Train a LLaMA-2 model with LoRA adapters on the injection dataset.
    
    Saves checkpoints under `inject_db/`.
    
- `3.Dataset_Creation_for_finetuning_.py`
    
    Prepares finetuning splits from the APTQA dataset (or another QA dataset).
    
- `4..Finetuning.py`
    
    Finetunes **starting from the knowledge-injected checkpoint**.
    
- `5.Inference_on _testset.py`
    
    Runs inference on the test set and saves predictions to CSV.



## Data Extraction Attack

This repository contains code and configuration files for running **data extraction attacks** against a fine-tuned LLaMA-2 model using prefix-based inference.

1. **Prefixes** are loaded from JSON (e.g., `Prefixes(05Tokens).json`).
2. **inference.py** runs the model with selected decoding strategy:
    - `greedy`, `beam_search`, `top_k_sampling`, `top_p_sampling`, `decaying_temperature`.
3. Generated continuations are saved to CSV:
    - `prefix` column = attacker input.
    - `generated` column = model continuation.
4. Outputs can be analyzed with regex/NER to extract IOCs and measure leakage.


## CTIGuardian Few-Shot Defense

- **`Entire_Pipe_4o.py`** → Full defense pipeline using GPT-4o-mini for classification + redaction, and LLaMA-2 for generation. Saves results to CSV.
- **`mistral.py`** → Local defense pipeline using Mistral-7B instead of GPT-4o for classification + redaction.
- **`redaction_few_shots.json`** → Few-shot examples to guide redaction of sensitive CTI data (IPs, emails, ports, etc.).
- **`classification_few_shots.json`** → Few-shot examples for harmful vs harmless prompt classification.















   
