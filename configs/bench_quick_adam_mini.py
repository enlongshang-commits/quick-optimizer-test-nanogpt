# Quick benchmark — Adam-mini
# Small model + Shakespeare for fast sanity-check on any machine (CPU / MPS / GPU)
# ~200 iters, no checkpoint saved
optimizer_type = 'adam_mini'
wandb_log = False
wandb_run_name = 'bench-quick-adam-mini'
out_dir = 'out-bench-adam-mini'

# Dataset
dataset = 'shakespeare_char'

# Model — GPT-mini (10M params)
n_layer = 6
n_head  = 6
n_embd  = 384
block_size = 256
dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-3
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
warmup_iters   = 20
lr_decay_iters = 200
min_lr         = 1e-4
decay_lr = True

# Training
max_iters = 200
batch_size = 32
gradient_accumulation_steps = 1
eval_interval = 50
eval_iters = 20
log_interval = 10
always_save_checkpoint = False

# System — change device to 'cuda' or 'cpu' as needed
device = 'mps'
dtype = 'float32'
compile = False
