#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version) and TorchVision
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install other required libraries
pip install -r requirements.txt
