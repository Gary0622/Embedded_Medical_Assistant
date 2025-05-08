#!/usr/bin/env python3
# scripts/make_jsonl.py

import os
import json
import random
import pandas as pd

# 1) Define directories
RAW_DIR = "data/mimic-iii-clinical-database-1.4"
OUT_DIR = "data/jsonl"
os.makedirs(OUT_DIR, exist_ok=True)

# 2) Load diagnosis and prescription tables
diagnoses = pd.read_csv(
    os.path.join(RAW_DIR, "DIAGNOSES_ICD.csv.gz"),
    compression="gzip",
    usecols=["HADM_ID", "ICD9_CODE"],
    dtype={"ICD9_CODE": "string"},
    low_memory=False
)
prescriptions = pd.read_csv(
    os.path.join(RAW_DIR, "PRESCRIPTIONS.csv.gz"),
    compression="gzip",
    usecols=["HADM_ID", "DRUG"],
    dtype={"DRUG": "string"},
    low_memory=False
)

# 3) Load clinical notes
notes = pd.read_csv(
    os.path.join(RAW_DIR, "NOTEEVENTS.csv.gz"),
    compression="gzip",
    usecols=["HADM_ID", "TEXT"],
    dtype={"TEXT": "string"},
    low_memory=False
)
# Concatenate all notes for each hospital stay
note_map = (
    notes
    .dropna(subset=["TEXT"])
    .groupby("HADM_ID")["TEXT"]
    .apply(lambda texts: " ".join(texts.tolist()))
    .to_dict()
)

# 4) Build prompt/completion pairs
records = []
for stay_id, group in diagnoses.groupby("HADM_ID"):
    # Collect unique ICD9 codes
    codes = sorted(x for x in group["ICD9_CODE"].dropna().unique())
    # Collect unique prescribed drugs
    meds = sorted(
        prescriptions.loc[prescriptions["HADM_ID"] == stay_id, "DRUG"]
        .dropna()
        .unique()
    )
    if not codes or not meds:
        continue

    # Construct the prompt
    prompt_lines = [f"Diagnosis ICD9: {','.join(codes)}"]
    note_text = note_map.get(stay_id, "")
    if note_text:
        prompt_lines.append(f"Clinical note: {note_text}")
    prompt_lines.append("Recommended medications: ")
    prompt = "\n".join(prompt_lines)

    completion = ",".join(meds)
    records.append({"prompt": prompt, "completion": completion})

# 5) Shuffle and split into train/validation sets
random.shuffle(records)
split_idx = int(len(records) * 0.9)
train_records = records[:split_idx]
valid_records = records[split_idx:]

# 6) Write out JSONL files
for split_name, subset in [("train", train_records), ("valid", valid_records)]:
    path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in subset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(subset)} {split_name} records to {path}")