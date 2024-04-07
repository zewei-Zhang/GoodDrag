#!/bin/bash

# Temporary file for modified requirements
TEMP_REQ_FILE="temp_requirements.txt"

# Detect CUDA version using nvcc
CUDA_VER_FULL=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\),.*/\1\2/p')

# Set the CUDA tag
CUDA_TAG="cu$CUDA_VER_FULL"

echo "Detected CUDA Tag: $CUDA_TAG"

# Modify the torch and torchvision lines in requirements.txt to include the CUDA version
while IFS= read -r line; do
    if [[ "$line" == torch==* ]]; then
        echo "torch==2.0.1+$CUDA_TAG" >> "$TEMP_REQ_FILE"
    elif [[ "$line" == torchvision==* ]]; then
        echo "torchvision==0.15.2+$CUDA_TAG" >> "$TEMP_REQ_FILE"
    else
        echo "$line" >> "$TEMP_REQ_FILE"
    fi
done < requirements.txt

# Replace the original requirements file with the modified one
mv "$TEMP_REQ_FILE" requirements.txt

# Define the virtual environment directory
VENV_DIR="GoodDrag"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it first."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Python script
echo "Starting gooddrag_ui.py..."
python3 gooddrag_ui.py

# Deactivate the virtual environment on script exit
deactivate

echo "Script finished."