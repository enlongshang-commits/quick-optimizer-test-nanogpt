"""
Template for adding a new optimizer to gpt2-optimizer-bench.

Steps:
  1. Copy this file, rename it (e.g. mobundle.py)
  2. Implement your optimizer class
  3. Fill in the @register factory function
  4. Add `import mobundle` at the top of train.py
  5. Create configs/gpt2_mobundle.py with optimizer_type = 'mobundle'
  6. Run: python train.py configs/gpt2_mobundle.py
"""

import torch
from optimizers import register


# ── 1. Implement your optimizer ───────────────────────────────────────────────

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                # --- your update rule here ---
                p.add_(grad, alpha=-group['lr'])

        return loss


# ── 2. Register factory function ──────────────────────────────────────────────

@register('my_optimizer')      # ← change this name
def _create_my_optimizer(model, weight_decay, learning_rate, betas, **kwargs):
    # All params passed by train.py are available here:
    #   betas=(beta1, beta2), muon_momentum, device_type, ddp, n_embd, n_head
    # Unused ones are safely captured by **kwargs.
    return MyOptimizer(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
