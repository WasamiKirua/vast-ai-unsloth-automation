#!/bin/bash

pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install wandb colorama vastai
