#!/usr/bin/env python3
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = "models/Llama-2-7b-hf"
adapter = "models/merged-instruct"
out = "models/merged-full-instruct"

model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model.save_pretrained(out)