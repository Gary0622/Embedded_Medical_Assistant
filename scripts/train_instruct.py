#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

BASE_MODEL = "models/Llama-2-7b-hf"
OFFLOAD_DIR = "offload"
DATA_FILES = {
    "train": "data/jsonl/train.jsonl",
    "validation": "data/jsonl/valid.jsonl"
}
ADAPTER_DIR = "models/merged-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    local_files_only=True
)
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files=DATA_FILES)

def tokenize_and_label(batch):
    inputs = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["completion"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(
    tokenize_and_label,
    batched=True,
    remove_columns=["prompt", "completion"]
)

training_args = TrainingArguments(
    output_dir="outputs-instruct",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print(f"LoRA adapter and tokenizer saved to '{ADAPTER_DIR}'")