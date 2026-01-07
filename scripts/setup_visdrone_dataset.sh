#!/bin/bash
# Script to setup VisDrone dataset structure

# Navigate to Project directory
cd /Users/harishshankar/Documents/Project

# Clone the VisDrone dataset repository
echo "Cloning VisDrone dataset repository..."
git clone https://github.com/VisDrone/VisDrone-Dataset.git

# Note: After cloning, you need to download the actual dataset files
# The repository only contains README and links to download the dataset
# Download links are available at: https://github.com/VisDrone/VisDrone-Dataset

echo ""
echo "Next steps:"
echo "1. Download the VisDrone-DET dataset files from the links in the repository"
echo "2. Extract the downloaded files"
echo "3. Organize them according to the expected structure (see README.md)"
echo ""
echo "Expected structure after organization:"
echo "  visdrone2021/"
echo "  ├── images/"
echo "  │   ├── train/"
echo "  │   ├── val/"
echo "  │   └── test/"
echo "  └── annotations/"
echo "      ├── train/"
echo "      ├── val/"
echo "      └── test/"

