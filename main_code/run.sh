#!/bin/bash

# Check if a Python file is provided
if [ -z "$1" ]; then
  echo "❌ Usage: $0 <python_file> [options]"
  echo "Example: $0 sphereface.py --batch-size 2048 --epochs 30 --lr 0.05"
  exit 1
fi

PYTHON_FILE=$1
shift  # Remove the first argument (the script name) from $@

# Default values
BATCH_SIZE=1024
EPOCHS=200
LR=0.1
BACKBONE="resnet18"

# Parse command-line arguments for overrides
for arg in "$@"; do
  case $arg in
    --batch-size=*)
      BATCH_SIZE="${arg#*=}"
      ;;
    --epochs=*)
      EPOCHS="${arg#*=}"
      ;;
    --lr=*)
      LR="${arg#*=}"
      ;;
  esac
done

# Print what will actually be run
echo "🚀 Running: python $PYTHON_FILE --batch-size $BATCH_SIZE --epochs $EPOCHS --lr $LR $@"

# Run the Python script
python "$PYTHON_FILE" --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR" --backbone "$BACKBONE" "$@"
