# Quick benchmark — SOAP
# Small model + Shakespeare for fast sanity-check on any machine (CPU / MPS / GPU)
# ~200 iters, no checkpoint saved
optimizer_type = 'soap'
wandb_log = True
wandb_run_name = 'bench-quick-soap'
out_dir = 'out-bench-soap'

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
learning_rate = 3e-3   # SOAP default; higher than AdamW
weight_decay  = 0.01
beta1 = 0.95
beta2 = 0.95
grad_clip = 1.0

# LR schedule
warmup_iters   = 20
lr_decay_iters = 200
min_lr         = 3e-4  # min_lr = lr / 10
decay_lr = True

# Training
max_iters = 200
batch_size = 32
gradient_accumulation_steps = 1
eval_interval = 20
eval_iters = 20
log_interval = 10
always_save_checkpoint = False

# System — change device to 'cuda' or 'cpu' as needed
device = 'mps'
dtype = 'float32'
compile = False
