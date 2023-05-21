import torch
import numpy as np
import os
import time

from model import PreDefinedViT

image_size = 224
num_classes = 1000
using_torch_compile = False
warmup_steps = 10
steps = 100


def ddp_train(rank, world_size, model_name, patch_size, batch_size_per_gpu, method):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.distributed.barrier()

    pre_rms_training = method in ['pre-rms', 'pre-customized-rms']
    raw_model = PreDefinedViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, variant=model_name, method=method, pre_rms_training=pre_rms_training).to(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(raw_model, device_ids=[rank])
    model = torch.compile(ddp_model) if using_torch_compile else ddp_model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

    def train_one_step(x, y, model, optimizer):
        with torch.cuda.amp.autocast():
            loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    x = torch.randn(batch_size_per_gpu, 3, image_size, image_size).to(rank)
    y = torch.randint(0, num_classes, (batch_size_per_gpu,)).to(rank)

    for _ in range(warmup_steps):
        train_one_step(x, y, model, optimizer)

    result = []
    for _ in range(steps):
        torch.distributed.barrier()
        start = time.time()
        train_one_step(x, y, model, optimizer)
        torch.distributed.barrier()
        end = time.time()
        result.append(end - start)
    result = np.array(result)
    q25, q50, q75 = np.percentile(result, np.array([25, 50, 75]))
    mean, std = result.mean(), result.std()
    print(f"rank {rank}, q25 {q25}, q50 {q50}, q75 {q75}, mean {mean}, std {std}")


def main():
    world_size = torch.cuda.device_count()
    for model_setting in [['Tiny', 16, 512], ['Small', 16, 512], ['Base', 16, 512], ['Large', 16, 256], ['Huge', 14, 64], ['Giant', 14, 64]]:
        model_name, patch_size, batch_size_per_gpu = model_setting
        for method in ['pre-ln', 'pre-apex-ln', 'pre-customized-ln', 'no-normalization', 'pre-apex-rms', 'pre-rms', 'pre-crms', 'pre-crms-same-dim']:
            print(f"ViT-{model_name}-{patch_size} {method}, batch size per GPU: {batch_size_per_gpu}")
            torch.multiprocessing.spawn(ddp_train, args=(world_size, model_name, patch_size, batch_size_per_gpu, method), nprocs=world_size, join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
