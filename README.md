# vast-ai-unsloth-automation

This repo is intended to provide tools to automate the fine tuning of LLM models using Vast.ai 

## Notes

With a few changes, if you want you could adapt the code to train your model of choice !

The following docker base image, Graphic and disk has been selected:

torch 2.2.0 CUDA 12.1
RTX 3090
40GB Disk (less is possible)

## !! Important !!

!! The script destroy instance after completition and push to hub !!

1) I push ALWAYS to HF the full model or the Lora adapters
2) I always ship metrics to Wandb
3) I do not suggest to merge Lora adapters to base it tends to give higher perplexity, load the base in 4bits via Bitsandbytes
   and load the adapters on top instead
4) I love Ollama since it also allows you to load the adapters via Model config file, if you want additional info on how to do that, drop a message I will be happy to help


### Installation

1) Load your dataset on HF
2) move the axolotl config yaml file into the root repo's folder
3) set the following vars as needed
```
preparation.py:

VASTAI_KEY = "YOUR KEY"
TARGET_TAG = 'gemma'

run_training.py

# HF parameters
HF_TOKEN = "YOUR TOKEN"
HF_DATASET = "YOUR DATASET"
#MERGE_PUSH = "WasamiKirua/Samantha2.0-Gemma2-9b"
LORA_PUSH = "YOUR OUTPUT on HF"

# Instance
TARGET_TAG = 'gemma'

# Vast.ai
VASTAI_KEY = "YOUR KEY"

# Wandb parameters
WANDB_TOKEN = "YOUR TOKEN"
WANDB_NAME = "RUN'S NAME"

# Training parameters (your turn!)
MAX_SEQ = 4096
TRAIN_BATCH_SIZE = 2
GRADIENT_ACC_STEPS = 4
LORA_LOCAL = "lora_model"
WARM_UP = 500
```

```
$ conda create --name unsloth-vastai python=3.10
$ conda activate unsloth-vastai
$ pip install colorama vastai
```
4) Instanciate your instance and wait for availability, be sure to tag the instance with the corresponding value of 'TARGET_TAG'
   
```
$ python preparation.py
```

NOTE: The script will log you in automatically

```
$ cd /workspace
$ chmod +x deps.sh
$ ./deps.sh
$ python run_training.py
```
 

