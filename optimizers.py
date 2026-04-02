"""
Optimizer factory for gpt2-optimizer-bench.

SOTA baselines: adamw, adam_mini, muon, soap (registered below)
New optimizers:  create a new file (e.g. my_opt.py), use @register('name'),
                 then add `import my_opt` at the top of train.py.
"""
import inspect
import torch

# ── Registry ──────────────────────────────────────────────────────────────────

OPTIMIZERS = {}

def register(name):
    """Decorator: register an optimizer factory function under `name`."""
    def decorator(fn):
        OPTIMIZERS[name] = fn
        return fn
    return decorator

def create_optimizer(optimizer_type, model, weight_decay, learning_rate, betas,
                     device_type, ddp=False, n_embd=768, n_head=12):
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type!r}. "
            f"Available: {list(OPTIMIZERS)}"
        )
    return OPTIMIZERS[optimizer_type](
        model=model,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=betas,
        device_type=device_type,
        ddp=ddp,
        n_embd=n_embd,
        n_head=n_head,
    )

# ── LR scaling ────────────────────────────────────────────────────────────────

def scale_lr(optimizer, lr, optimizer_type, initial_lr):
    """Apply cosine-decayed lr to all param groups."""
    ratio = lr / initial_lr
    for group in optimizer.param_groups:
        if optimizer_type == 'adam_mini':
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']
            group['lr'] = group['initial_lr'] * ratio
        else:
            group['lr'] = lr

# ── SOTA baselines ─────────────────────────────────────────────────────────────

@register('adamw')
def _create_adamw(model, weight_decay, learning_rate, betas, device_type, **kwargs):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params,   'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas,
        **(dict(fused=True) if use_fused else {})
    )
    print(f"[optimizer] AdamW  (fused={use_fused})")
    return optimizer


@register('adam_mini')
def _create_adam_mini(model, weight_decay, learning_rate, betas, n_embd, n_head, **kwargs):
    try:
        from adam_mini import Adam_mini
    except ImportError:
        raise ImportError("Run: pip install adam-mini")
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        model_sharding=False,
        dim=n_embd,
        n_heads=n_head,
    )
    print(f"[optimizer] Adam-mini  (dim={n_embd}, n_heads={n_head})")
    return optimizer


@register('soap')
def _create_soap(model, weight_decay, learning_rate, betas, **kwargs):
    try:
        from soap import SOAP
    except ImportError:
        raise ImportError("soap.py not found. Run setup.sh to download it.")
    # SOAP works best on 2D+ weight matrices; apply to all params
    optimizer = SOAP(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        precondition_frequency=10,
    )
    print(f"[optimizer] SOAP  (lr={learning_rate}, precondition_frequency=10)")
    return optimizer


@register('muon')
def _create_muon(model, weight_decay, learning_rate, ddp, **kwargs):
    try:
        if ddp:
            from muon import MuonWithAuxAdam as MuonCls
        else:
            from muon import SingleDeviceMuonWithAuxAdam as MuonCls
    except ImportError:
        raise ImportError("muon.py not found. Run setup.sh to download it.")

    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_matrix = param.dim() >= 2
        is_embedding_or_head = any(k in name for k in ('wte', 'wpe', 'lm_head'))
        if is_matrix and not is_embedding_or_head:
            muon_params.append(param)
        else:
            adam_params.append(param)

    param_groups = [
        {'params': muon_params, 'use_muon': True,  'lr': learning_rate},
        {'params': adam_params, 'use_muon': False, 'lr': learning_rate, 'weight_decay': weight_decay},
    ]
    optimizer = MuonCls(param_groups)
    print(f"[optimizer] Muon  (ddp={ddp}, muon_params={len(muon_params)}, adam_params={len(adam_params)})")
    return optimizer
