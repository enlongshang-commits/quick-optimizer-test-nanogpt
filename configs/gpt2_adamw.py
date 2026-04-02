# GPT-2 (124M) — AdamW baseline (nanoGPT default)
optimizer_type = 'adamw'
wandb_run_name = 'gpt2-adamw'
out_dir = 'out-adamw'

# model (GPT-2 small)
n_layer = 12
n_head  = 12
n_embd  = 768

# optimizer
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
gradient_accumulation_steps = 40   # adjust for your GPU memory
