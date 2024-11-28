#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install PyTorch and dependencies
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install detectron2 from GitHub
pip install git+https://github.com/facebookresearch/detectron2.git
