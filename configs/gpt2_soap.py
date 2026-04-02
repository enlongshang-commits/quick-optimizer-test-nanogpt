# GPT-2 (124M) — SOAP
# Paper: https://arxiv.org/abs/2409.11321
# Requires soap.py in project root (setup.sh downloads it)
optimizer_type = 'soap'
wandb_run_name = 'gpt2-soap'
out_dir = 'out-soap'

n_layer = 12
n_head  = 12
n_embd  = 768

# SOAP recommended lr is higher than AdamW
learning_rate = 3e-3
weight_decay  = 0.01
beta1 = 0.95
beta2 = 0.95
grad_clip = 1.0

warmup_iters    = 2000
lr_decay_iters  = 600000
min_lr          = 3e-4

max_iters                   = 600000
batch_size                  = 12
gradient_accumulation_steps = 40
