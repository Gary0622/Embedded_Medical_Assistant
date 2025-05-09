# Medical LLM Project (Llama-2-7b-hf QLoRA)

This repository contains code to fine-tune Llama-2-7b-hf on MIMIC-III for ICD9→medication recommendation.

## Installation

```bash
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
cd medical-llm-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
accelerate config   
```

## **Project Structure**
```
project/
├── raspi-device-ver-main/      # device source files
│   ├── app.py
│   ├── gui_voice_llm.py
│   ├── stt.py
│   ├── tts_coqui.py
│   └── voice2llm_vosk.py
├── training/
│   ├── make_jsonl.py
│   ├── train_instruct.py
│   ├── merge_lora.py
│   └── test_generate.py
├── requirements.txt
└── README.md

```

## **Usage**
	1.Prepare data
  	• Place MIMIC-III CSV files under data/mimic-iii-clinical-database-1.4/
  	• Run python scripts/make_jsonl.py to produce data/jsonl/train.jsonl & valid.jsonl
 	2.Fine-tune
  	• Clone weights: git clone https://huggingface.co/meta-llama/Llama-2-7b-hf models/Llama-2-7b-hf && cd models/Llama-2-7b-hf && git lfs pull
  	• Run accelerate launch scripts/train_instruct.py
 	3.Merge & Infer
  	• python scripts/merge_lora.py
  	• python scripts/test_generate.py --model models/merged-instruct --prompt "Diagnosis ICD9: 4019\nRecommended medications: "
