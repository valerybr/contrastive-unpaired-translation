"""Distributed (DDP) helpers.

Detection is via torchrun env vars: WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR,
MASTER_PORT. If WORLD_SIZE > 1 the run is treated as DDP; otherwise this module
no-ops, leaving the legacy DataParallel / single-GPU code paths intact.
"""
import os
import torch
import torch.distributed as dist


def is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main():
    return get_rank() == 0


def is_initialized():
    return dist.is_available() and dist.is_initialized()


def init_process_group(backend="nccl"):
    """Initialize the default process group if running under torchrun.

    Safe to call more than once; second call is a no-op.
    """
    if not is_ddp():
        return
    if is_initialized():
        return
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def cleanup():
    if is_initialized():
        dist.destroy_process_group()


def barrier():
    if is_initialized():
        dist.barrier()


@torch.no_grad()
def broadcast_module(module, src=0):
    """Broadcast all parameters and buffers of `module` from `src` to every rank.

    Used after lazy parameter materialization (e.g. PatchSampleF MLPs) to make
    sure every rank starts from identical weights regardless of per-rank RNG.
    """
    if not is_initialized():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


@torch.no_grad()
def broadcast_bool(value, src=0, device=None):
    """Broadcast a Python bool from `src` so every rank gets the same answer.

    Returns a Python bool. No-op (returns the input) when not in DDP.
    """
    if not is_initialized():
        return bool(value)
    if device is None:
        # Pick a device that matches the backend: nccl needs CUDA, gloo can
        # use CPU. Falls back to CPU when CUDA is unavailable.
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", get_local_rank())
        else:
            device = torch.device("cpu")
    t = torch.tensor(int(bool(value)), dtype=torch.int64, device=device)
    dist.broadcast(t, src=src)
    return bool(t.item())
