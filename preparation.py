# Script Optimized for Vast.ai Template torch 2.2.0 CUDA 12.1
# 40 GB Disk - RTX 3090
# https://cloud.vast.ai/templates/edit?templateHashId=886c5741378aa948e0e41edeac0caaab

from colorama import Fore, Style
import os
import json

VASTAI_KEY = ""
TARGET_TAG = 'gemma'

print(f"{Fore.GREEN}Setting Vast.ai api key{Style.RESET_ALL}\n")
os.system(f"vastai set api-key {VASTAI_KEY}")
print()

print(f"{Fore.GREEN}Getting Instance info{Style.RESET_ALL}\n")
instance_infos = os.system("vastai show instances --raw > infos.json")
print()

print(f"{Fore.GREEN}Instance infos:{Style.RESET_ALL}\n")
with open("infos.json", "r") as f:
    data = json.load(f)

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
            print("************")
            print(f"{Fore.CYAN}Found {Fore.RED}{instance['label']}{Style.RESET_ALL} target instance !{Style.RESET_ALL}\n")
            print("************")
            print(f"{Fore.CYAN}Copying Notebook for dependencies installation to instance {instance['label']}{Style.RESET_ALL}\n")
            os.system(f"scp -P {instance['ssh_port']} -o StrictHostKeyChecking=accept-new deps.sh root@{instance['public_ip']}:/workspace")
            print()
            print(f"{Fore.CYAN}Copying JSON to instance {instance['label']}{Style.RESET_ALL}\n")
            os.system(f"scp -P {instance['ssh_port']} -o StrictHostKeyChecking=accept-new infos.json root@{instance['public_ip']}:/workspace")
            print()
            print(f"{Fore.CYAN}Copying training script to instance {instance['label']}{Style.RESET_ALL}\n")
            os.system(f"scp -P {instance['ssh_port']} -o StrictHostKeyChecking=accept-new run_training.py root@{instance['public_ip']}:/workspace")
            print()
            print(f"{Fore.CYAN}Connecting to instance {instance['label']}{Style.RESET_ALL}\n")
            os.system(f"ssh -p {instance['ssh_port']} -o StrictHostKeyChecking=accept-new root@{instance['public_ip']}")
            print()
