# GPT-2 Optimizer Benchmark

A clean benchmark for comparing optimizers on GPT-2 training.
Built on top of [nanoGPT](https://github.com/karpathy/nanoGPT).

## Features

- Plug-and-play optimizer registration via `@register`
- Four SOTA baselines out of the box: AdamW, Adam-mini, Muon, SOAP
- Supports single GPU and multi-GPU (DDP) training
- WandB logging for loss curve comparison

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gpt2-optimizer-bench
cd gpt2-optimizer-bench
bash setup.sh
```

`setup.sh` will automatically:
- Clone nanoGPT and copy `model.py`, `configurator.py`, `data/`
- Download `muon.py` and `soap.py`
- Install Python dependencies via `pip install -r requirements.txt`

## Prepare Dataset

Quick test (Shakespeare, ~1MB):
```bash
python data/shakespeare_char/prepare.py
```

Full training (OpenWebText, ~54GB):
```bash
python data/openwebtext/prepare.py
```

## Run Training

### Quick benchmark (small model, 200 iters, Shakespeare)

For fast sanity-check on any machine. Set `device` to `cuda` / `mps` / `cpu` as needed.

```bash
python train.py configs/bench_quick_adamw.py
python train.py configs/bench_quick_adam_mini.py
python train.py configs/bench_quick_muon.py
python train.py configs/bench_quick_soap.py
```

### Full GPT-2 (124M, OpenWebText)

```bash
python train.py configs/gpt2_adamw.py
python train.py configs/gpt2_adam_mini.py
python train.py configs/gpt2_muon.py
python train.py configs/gpt2_soap.py
```

For multi-GPU (DDP):
```bash
torchrun --standalone --nproc_per_node=4 train.py configs/gpt2_adamw.py
```

### Override config via command line

Any config value can be overridden directly on the command line without editing files:

```bash
python train.py configs/gpt2_adamw.py \
  dataset=shakespeare_char \
  device=mps \
  dtype=float32 \
  compile=False \
  max_iters=500 \
  batch_size=32 \
  gradient_accumulation_steps=1
```

## Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer_type` | `adamw` | `adamw` / `adam_mini` / `muon` / `soap` |
| `learning_rate` | `6e-4` | Base learning rate |
| `weight_decay` | `0.1` | Weight decay |
| `beta1` | `0.9` | Adam β₁ |
| `beta2` | `0.95` | Adam β₂ |
| `grad_clip` | `1.0` | Gradient clipping (`0` = disabled) |
| `warmup_iters` | `2000` | LR warmup steps |
| `lr_decay_iters` | `600000` | Steps to decay LR over |
| `min_lr` | `6e-5` | Minimum LR (cosine schedule floor) |
| `max_iters` | `600000` | Total training iterations |
| `batch_size` | `12` | Micro-batch size per step |
| `gradient_accumulation_steps` | `40` | Gradient accumulation steps |
| `block_size` | `1024` | Context length (tokens) |
| `n_layer` | `12` | Number of transformer layers |
| `n_head` | `12` | Number of attention heads |
| `n_embd` | `768` | Embedding dimension |
| `dropout` | `0.0` | Dropout rate |
| `bias` | `False` | Use bias in Linear/LayerNorm |
| `dataset` | `openwebtext` | Dataset name (must match `data/<name>/`) |
| `device` | `cuda` | `cuda` / `mps` / `cpu` |
| `dtype` | `bfloat16` | `float32` / `bfloat16` / `float16` |
| `compile` | `True` | `torch.compile` (disable on MPS/CPU) |
| `eval_interval` | `2000` | Evaluate every N iters |
| `eval_iters` | `200` | Batches to average for eval loss |
| `log_interval` | `10` | Print loss every N iters |
| `wandb_log` | `False` | Enable WandB logging |
| `wandb_project` | `gpt2-bench` | WandB project name |
| `wandb_run_name` | `run` | WandB run name |
| `out_dir` | `out` | Directory to save checkpoints |
| `always_save_checkpoint` | `True` | Save checkpoint on every eval |
| `init_from` | `scratch` | `scratch` / `resume` / `gpt2*` |

## Adding a New Optimizer

1. Copy `new_optimizer_template.py` and rename it (e.g. `my_optimizer.py`)
2. Implement and register your optimizer:

```python
# my_optimizer.py
from optimizers import register

@register('my_optimizer')
def _create(model, learning_rate, weight_decay, **kwargs):
    return MyOptimizer(model.parameters(), lr=learning_rate)
```

3. Add one import line to `train.py`:

```python
import my_optimizer   # triggers registration
```

4. Create a config:

```python
# configs/gpt2_my_optimizer.py
optimizer_type = 'my_optimizer'
out_dir = 'out-my-optimizer'
learning_rate = 1e-3
```

5. Run:

```bash
python train.py configs/gpt2_my_optimizer.py
```

## Compare Results with WandB

Enable logging in your config:
```python
wandb_log = True
wandb_project = 'gpt2-bench'
wandb_run_name = 'gpt2-adamw'
```

Then select multiple runs on the WandB dashboard to overlay loss curves.

## Project Structure

```
gpt2-optimizer-bench/
├── train.py                    # Main training script
├── optimizers.py               # Optimizer registry and SOTA baselines
├── new_optimizer_template.py   # Template for adding new optimizers
├── setup.sh                    # Environment setup
├── requirements.txt            # Python dependencies
├── configs/
│   ├── bench_quick_adamw.py    # Quick benchmark (200 iters, Shakespeare)
│   ├── bench_quick_adam_mini.py
│   ├── bench_quick_muon.py
│   ├── bench_quick_soap.py
│   ├── gpt2_adamw.py           # Full GPT-2 (124M, OpenWebText)
│   ├── gpt2_adam_mini.py
│   ├── gpt2_muon.py
│   └── gpt2_soap.py
│
│ (generated by setup.sh, not committed)
├── model.py                    # GPT-2 model (from nanoGPT)
├── configurator.py             # Config loader (from nanoGPT)
├── muon.py                     # Muon optimizer
├── soap.py                     # SOAP optimizer
└── data/                       # Dataset preparation scripts
```

## Baselines

| Optimizer | Paper | Recommended LR | Notes |
|-----------|-------|----------------|-------|
| AdamW | Loshchilov & Hutter, 2019 | 6e-4 | Standard baseline |
| Adam-mini | Zhang et al., 2024 | 6e-4 | Reduces memory by using fewer lr values |
| Muon | Jordan et al., 2024 | 0.02 | Orthogonal gradient update for hidden weights |
| SOAP | Vyas et al., 2024 | 3e-3 | Shampoo-like preconditioning in Adam's eigenbasis |

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Adam-mini](https://github.com/zyushun/Adam-mini)
- [Muon](https://github.com/KellerJordan/Muon)
- [SOAP](https://github.com/nikhilvyas/SOAP)
