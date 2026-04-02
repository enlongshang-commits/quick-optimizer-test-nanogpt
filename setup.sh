#!/bin/bash
# Setup script for gpt2-optimizer-bench
# Run once from the project root: bash setup.sh

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Cloning nanoGPT ==="
cd "$PROJ_DIR"
if [ ! -d "nanoGPT" ]; then
    git clone https://github.com/karpathy/nanoGPT.git
fi

echo "=== Copying nanoGPT files ==="
cp nanoGPT/model.py        "$PROJ_DIR/model.py"
cp nanoGPT/configurator.py "$PROJ_DIR/configurator.py"
# Copy data preparation scripts
cp -r nanoGPT/data "$PROJ_DIR/data" 2>/dev/null || true

echo "=== Fetching muon.py and soap.py ==="
curl -fsSL https://raw.githubusercontent.com/KellerJordan/Muon/master/muon.py \
    -o "$PROJ_DIR/muon.py"
curl -fsSL https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py \
    -o "$PROJ_DIR/soap.py"

echo "=== Installing Python packages ==="
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Prepare dataset (e.g. Shakespeare):"
echo "     python data/shakespeare_char/prepare.py"
echo "     Then set: dataset = 'shakespeare_char' in your config"
echo ""
echo "  2. Run training:"
echo "     python train.py configs/gpt2_adamw.py"
echo "     python train.py configs/gpt2_adam_mini.py"
echo "     python train.py configs/gpt2_muon.py"
echo "     python train.py configs/gpt2_soap.py"
echo ""
echo "  3. For DDP (multi-GPU):"
echo "     torchrun --standalone --nproc_per_node=4 train.py configs/gpt2_adamw.py"
