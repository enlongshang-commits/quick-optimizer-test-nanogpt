# GPT-2 (124M) — Muon
# Paper: https://arxiv.org/abs/2409.20325
# Requires muon.py copied into the project root
optimizer_type = 'muon'
wandb_run_name = 'gpt2-muon'
out_dir = 'out-muon'

# model (GPT-2 small)
n_layer = 12
n_head  = 12
n_embd  = 768

# optimizer
# Muon typically uses a higher lr than AdamW
learning_rate = 0.02   # Muon default for matrix params
weight_decay  = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule
warmup_iters    = 2000
lr_decay_iters  = 600000
min_lr          = 2e-3   # ~lr/10

# training
max_iters                   = 600000
batch_size                  = 12
gradient_accumulation_steps = 40
