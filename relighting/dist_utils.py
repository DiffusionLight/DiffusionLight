"""
Helpers for distributed training.
"""

import io
import os
import socket

try:
    import blobfile as bf
except:
    pass

try:
    from mpi4py import MPI
except:
    pass

import torch as th
import torch.distributed as dist
import builtins
import datetime

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3
def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def setup_dist_multinode(args):
    """
    Setup a distributed process group.
    """
    if not dist.is_available() or not dist.is_initialized():
        th.distributed.init_process_group(backend="nccl", init_method='env://')
        world_size = dist.get_world_size()
        local_rank = int(os.getenv('LOCAL_RANK'))
        print("rank",local_rank)
        device = local_rank
        th.cuda.set_device(device)
        setup_for_distributed(device == 0)

        synchronize()
    else:
        print("ddp failed!")
        exit()

def setup_dist(global_seed):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    th.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    th.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=5400))

    # fix seed
    rank = dist.get_rank()
    device = rank % th.cuda.device_count()
    seed = global_seed * dist.get_world_size() + rank
    th.manual_seed(seed)
    th.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    synchronize()

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()