"""
A clean, efficient AdamW optimizer with fused kernel and ZeRO-2 style distributed support.
Adapted from modded-nanogpt with Muon removed in favor of pure AdamW for DiffusionBlocks.
"""

import torch
import torch.distributed as dist
from torch import Tensor

"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,  # parameter tensor
    grad: Tensor,  # gradient, same shape as p
    exp_avg: Tensor,  # first moment, same shape as p
    exp_avg_sq: Tensor,  # second moment, same shape as p
    step_t: Tensor,  # () - 0-D CPU tensor, step count
    lr_t: Tensor,  # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,  # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,  # () - 0-D CPU tensor, beta2
    eps_t: Tensor,  # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,  # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    """
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    p32.mul_(1 - lr_t * wd_t)
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)


class AdamW(torch.optim.Optimizer):
    """
    Fused AdamW optimizer with optional ZeRO-2 style distributed sharding.
    Drop-in replacement that eliminates Muon completely for DiffusionBlocks.
    """

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue
            if world_size == 1:
                param_infos[p] = dict(future=None, grad_slice=grad, is_small=True)
            elif p.numel() < 1024:
                future = dist.all_reduce(
                    grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0, (
                    f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                )
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(
                    grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                param_infos[p] = dict(
                    future=future, grad_slice=grad_slice, is_small=False
                )
        return dict(param_infos=param_infos)

    def _compute_adamw(
        self,
        group: dict,
        info: dict,
        gather_list: list,
        rank: int,
        world_size: int,
    ) -> None:
        """Wait for reduce, compute AdamW updates, launch gather."""
        param_infos = info["param_infos"]
        for p, pinfo in param_infos.items():
            if pinfo["future"] is not None:
                pinfo["future"].wait()

            state = self.state[p]
            grad_slice = pinfo["grad_slice"]
            is_small = pinfo["is_small"]

            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(grad_slice)
                state["exp_avg_sq"] = torch.zeros_like(grad_slice)

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            self._adamw_step_t.fill_(step)
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            if is_small:
                adamw_step_fused(
                    p,
                    grad_slice,
                    exp_avg,
                    exp_avg_sq,
                    self._adamw_step_t,
                    self._adamw_lr_t,
                    self._adamw_beta1_t,
                    self._adamw_beta2_t,
                    self._adamw_eps_t,
                    self._adamw_wd_t,
                )
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size : (rank + 1) * rank_size]
                adamw_step_fused(
                    p_slice,
                    grad_slice,
                    exp_avg,
                    exp_avg_sq,
                    self._adamw_step_t,
                    self._adamw_lr_t,
                    self._adamw_beta1_t,
                    self._adamw_beta2_t,
                    self._adamw_eps_t,
                    self._adamw_wd_t,
                )
                future = dist.all_gather_into_tensor(
                    p, p_slice, async_op=True
                ).get_future()
                gather_list.append(dict(future=future, params=None))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers to complete."""
        for info in gather_list:
            if info["future"] is not None:
                info["future"].wait()

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        reduce_infos: list[dict] = []
        for group in self.param_groups:
            # All groups are AdamW; 'kind' key tolerated for backward compatibility
            reduce_infos.append(self._reduce_adamw(group, world_size))

        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            self._compute_adamw(group, info, gather_list, rank, world_size)

        self._finish_gathers(gather_list)
        return loss
