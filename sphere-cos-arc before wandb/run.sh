# Check if a Python file is provided
if [ -z "$1" ]; then
  echo "❌ Usage: $0 <python_file> [options]"
  echo "Example: $0 sphereface.py --batch-size 2048 --epochs 200 --lr 0.05"
  exit 1
fi

PYTHON_FILE=$1
shift  # Remove the first argument (the script name) from $@

# Default arguments (can be overridden by command-line ones)
BATCH_SIZE=2048
EPOCHS=200
LR=0.05

# Run the Python file with custom or default args
echo "🚀 Running: python $PYTHON_FILE $@"
python "$PYTHON_FILE" "$@"
