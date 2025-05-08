#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_dir = "models/merged-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", local_files_only=True)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
output = gen("Diagnosis ICD9: 4019\nRecommended medications: ", max_new_tokens=64)
print(output[0]["generated_text"])