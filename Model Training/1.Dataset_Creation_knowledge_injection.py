#!/usr/bin/env python3
import os
import re
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


INPUT_TXT_DIR = "Knowledge Injection Dataset/DB_Text"  # USE the CVE Dataset containing txt files from given link
OUTPUT_DIR = "inject_db"  # output folder for CSVs


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def load_txt_folder(txt_root: str, min_len: int = 1) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(txt_root, "**", "*.txt"), recursive=True))
    rows = []
    for p in tqdm(paths, desc="Reading .txt files"):
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = clean_text(raw)
            if len(text) >= min_len:
                rows.append({"text": text})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.len() > 0]
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def split_and_save(df: pd.DataFrame, out_dir: str,
                   test_size: float = 0.1, val_size: float = 0.1, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    # train / (val+test)
    train_df, temp_df = train_test_split(df, test_size=(test_size + val_size),
                                         random_state=seed, shuffle=True)
    # split temp into val / test
    val_rel = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.5
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_rel),
                                       random_state=seed, shuffle=True)

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Saved → {out_dir}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    df = load_txt_folder(INPUT_TXT_DIR, min_len=10)
    if df.empty:
        raise SystemExit("No valid text found. Check your input directory.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_path = os.path.join(OUTPUT_DIR, "all_data.csv")
    df.to_csv(all_path, index=False)
    print(f"All data: {len(df)} rows → {all_path}")

    split_and_save(df, OUTPUT_DIR)
