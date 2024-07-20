import os
import torch
import wandb
import json
from colorama import Fore, Style
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
from time import sleep

# HF parameters
HF_TOKEN = ""
HF_DATASET = "WasamiKirua/Samantha2.0-Gemma2-9B"
MERGE_PUSH = "WasamiKirua/Samantha2.0-Gemma2-9b"
LORA_PUSH = "WasamiKirua/Samantha2.0-Gemma2-9b-Lora"

# Instance
TARGET_TAG = 'gemma'

# Vast.ai
VASTAI_KEY = ""

# Wandb parameters
WANDB_TOKEN = ""
WANDB_NAME = ""

# Training parameters
MAX_SEQ = 4096
TRAIN_BATCH_SIZE = 2
GRADIENT_ACC_STEPS = 4
LORA_LOCAL = "lora_model"
WARM_UP = 500


#print(f"{Fore.CYAN}Installing necessary python packages ...{Style.RESET_ALL}\n")
#os.system("pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton --index-url https://download.pytorch.org/whl/cu121")
#os.system("pip install unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git")

print(f"{Fore.CYAN}Logging to HF Hub{Style.RESET_ALL}\n")
os.system(f"huggingface-cli login --token {HF_TOKEN}")

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",          # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b",
    #max_seq_length = max_seq_length,
    max_seq_length = MAX_SEQ,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = load_dataset(HF_DATASET, split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

print(dataset[5]["conversations"])
print("\n\n")
print(dataset[5]["text"])
print("\n")


wandb.login(key = WANDB_TOKEN)

run = wandb.init(
    project='Gemma9b', 
    job_type="training", 
    anonymous="allow",
    name=WANDB_NAME
)

#steps_per_epoch=len(dataset["train"])//(per_device_train_batch_size*gradient_accumulation_steps)

#steps_per_epoch=len(dataset["text"])//(TRAIN_BATCH_SIZE*GRADIENT_ACC_STEPS)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
    per_device_train_batch_size = TRAIN_BATCH_SIZE, # Adjust according to your GPU memory
    gradient_accumulation_steps = GRADIENT_ACC_STEPS,
    warmup_steps = WARM_UP,
    num_train_epochs = 4,
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.1,
    lr_scheduler_type = "cosine",
    seed = 3407,
    output_dir = "outputs",
    report_to="wandb"
    ),
)

print(f"{Fore.CYAN} Starting Training ...{Style.RESET_ALL}")
trainer_stats = trainer.train()


def lora_to_hub():
    model.push_to_hub(LORA_PUSH, token=HF_TOKEN)
    tokenizer.push_to_hub(LORA_PUSH, token=HF_TOKEN)

def local_save():
    model.save_pretrained(LORA_LOCAL) # Local saving
    tokenizer.save_pretrained(LORA_LOCAL)

def merge_to_hub():
    model.push_to_hub(MERGE_PUSH, token=HF_TOKEN)
    tokenizer.push_to_hub(MERGE_PUSH, token=HF_TOKEN)

print(f"{Fore.CYAN}Pushing Lora Adapters to Hub ...{Style.RESET_ALL}\n")
lora_to_hub()


print(f"{Fore.GREEN}Instance infos:{Style.RESET_ALL}\n")
with open("infos.json", "r") as f:
    data = json.load(f)

    # Extract the desired fields
    extracted_data = []
    # Extract the desired fields
    extracted_data = []
    for instance in data:
        ssh_port = instance["ports"]['22/tcp'][0]['HostPort']
        instance_info = {
            "instance_id": instance["id"],
            "public_ip": instance["public_ipaddr"],
            "ssh_host": instance["ssh_host"],
            "ssh_port": ssh_port,
            "label": instance["label"],
            "-----": "-----"
        }
        extracted_data.append(instance_info)

for instance in extracted_data:
    # Print each key-value pair for the current instance
    for key, value in instance.items():
        print(f"{Fore.GREEN}{key.replace('_', ' ').capitalize()}: {value}{Style.RESET_ALL}")
        print()
        if key == "label" and value == TARGET_TAG:
            print(f"{Fore.CYAN}Setting Vast.ai api-key{Style.RESET_ALL}\n")
            os.system(f"vastai set api-key {VASTAI_KEY}")

            print(f"{Fore.RED}Destroying instance{Style.RESET_ALL}\n")
            os.system(f"vastai destroy instance {instance['instance_id']}")

