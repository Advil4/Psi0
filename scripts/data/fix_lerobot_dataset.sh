#!/bin/bash
# Universal LeRobot dataset fix script
# This is a wrapper that runs the comprehensive Python fix script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default dataset
DATASET_NAME="${1:-Pick_up_an_apple}"

echo "=========================================="
echo "LeRobot Dataset Fix Script"
echo "=========================================="
echo ""
echo "This script will fix all compatibility issues with the"
echo "LeRobot dataset for training."
echo ""
echo "Dataset: $DATASET_NAME"
echo "Location: $PROJECT_ROOT/real/$DATASET_NAME"
echo ""

# Check if dataset exists
if [ ! -d "$PROJECT_ROOT/real/$DATASET_NAME" ]; then
    echo "ERROR: Dataset directory not found!"
    echo "Expected: $PROJECT_ROOT/real/$DATASET_NAME"
    echo ""
    echo "Available datasets:"
    ls -1 "$PROJECT_ROOT/real/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Run the Python fix script
echo "Running comprehensive fix script..."
echo ""

cd "$PROJECT_ROOT"
python scripts/data/fix_lerobot_dataset.py --dataset "$DATASET_NAME"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Fix completed successfully!"
    echo "=========================================="
    echo ""
    echo "You can now run training:"
    echo "  bash scripts/train/psi0/finetune-real-psi0.sh $DATASET_NAME"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Fix failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "For detailed documentation, see:"
    echo "  scripts/data/LEROBOT_DATASET_FIX_README.md"
    echo ""
fi

exit $EXIT_CODE
