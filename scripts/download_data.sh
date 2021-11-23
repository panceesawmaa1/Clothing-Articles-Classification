#!/bin/bash

#-------------------------------------------------------------------------------

#             Fashion-MNIST Dataset Downloading Script

#-------------------------------------------------------------------------------

echo "This script assumes you have root access privileges ! "

# Store current dir path to be restored back after data download
dir="$(pwd)"

# Step back to project home dir
cd ..

# Download Training data and labels
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

# Download Testing data and labels
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P data/fashion http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz


# Restore curr_dir
cd "$dir" || exit