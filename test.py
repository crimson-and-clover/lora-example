# test_nccl_ddp.py
import os
import torch
import torch.distributed as dist

def main():
    # 由 torchrun 注入的环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # 每个 rank 放一个张量 = 自己的 rank 值
    x = torch.tensor([float(rank)], device=f"cuda:{local_rank}")

    # 所有 GPU 做一次求和
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # 仅 rank 0 打印结果
    if rank == 0:
        expected = world_size * (world_size - 1) / 2.0
        print(f"[OK] world_size={world_size}, all_reduce sum={x.item()}, expected={expected}")
        # 简单一致性校验
        assert abs(x.item() - expected) < 1e-6, "AllReduce result mismatch!"

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
