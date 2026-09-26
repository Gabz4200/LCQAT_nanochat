"""Tests for pure AdamW optimizer without Muon."""

import torch

from nanochat.optim import AdamW


def test_adamw_step_deterministic():
    """Verify AdamW optimizer steps deterministically on CPU."""
    torch.manual_seed(42)
    p1 = torch.nn.Parameter(torch.randn(32, 32))
    p2 = torch.nn.Parameter(p1.clone().detach())

    opt1 = AdamW(
        [
            {
                "params": [p1],
                "lr": 1e-3,
                "betas": (0.9, 0.99),
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        ]
    )
    opt2 = torch.optim.AdamW(
        [p2], lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01
    )

    for _ in range(5):
        grad = torch.randn(32, 32)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        opt1.step()
        opt2.step()

    assert torch.allclose(p1, p2, atol=1e-4, rtol=1e-4)


def test_adamw_param_groups_multi():
    """Multiple param groups with custom hyperparameters."""
    p1 = torch.nn.Parameter(torch.randn(10))
    p2 = torch.nn.Parameter(torch.randn(20))

    opt = AdamW(
        [
            {
                "params": [p1],
                "lr": 1e-3,
                "betas": (0.8, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
            {
                "params": [p2],
                "lr": 5e-4,
                "betas": (0.9, 0.99),
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        ]
    )

    p1.grad = torch.ones_like(p1)
    p2.grad = torch.ones_like(p2)
    opt.step()

    assert opt.state[p1]["step"] == 1
    assert opt.state[p2]["step"] == 1
