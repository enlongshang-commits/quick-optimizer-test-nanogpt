# GPT-2 (124M) — Adam-mini
# Paper: https://arxiv.org/abs/2406.16793
# Install: pip install adam-mini
optimizer_type = 'adam_mini'
wandb_run_name = 'gpt2-adam-mini'
out_dir = 'out-adam-mini'

# model (GPT-2 small)
n_layer = 12
n_head  = 12
n_embd  = 768

# optimizer
# Adam-mini recommended lr is similar to AdamW; start with the same
learning_rate = 6e-4
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule
warmup_iters    = 2000
lr_decay_iters  = 600000
min_lr          = 6e-5

# training
max_iters                   = 600000
batch_size                  = 12
gradient_accumulation_steps = 40
