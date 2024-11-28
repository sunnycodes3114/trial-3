#!/bin/bash

echo "Starting setup.sh script..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

# Install PyTorch (CPU version) and TorchVision
echo "Installing PyTorch and TorchVision..."
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html || { echo "Failed to install PyTorch"; exit 1; }

# Install Detectron2 from GitHub
echo "Installing Detectron2..."
pip install git+https://github.com/facebookresearch/detectron2.git || { echo "Failed to install Detectron2"; exit 1; }

# Install other dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || { echo "Failed to install requirements.txt dependencies"; exit 1; }

echo "setup.sh completed successfully."
