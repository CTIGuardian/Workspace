### CTIGuardian: Mitigating the Privacy Leakage of LLMs Trained on CTI Data with a Few-Shots


## Datasets

This project relies on two datasets for evaluating privacy leakage in fine-tuned LLMs:

### 1) APTQA 
A curated dataset derived from APT reports, formatted as **CSV files**. Unlike most public CTI datasets, APTQA **retains sensitive indicators of compromise (IOCs)** such as IPs, domains, emails, and ports, enabling realistic leakage evaluation.

- **Location in repo:** `Datasets/APTQA/`


