import torch
import torch.utils.benchmark as benchmark
from model import PreDefinedViT

image_size = 224
num_classes = 1000
using_torch_compile = False

num_threads_list = [1]
min_run_time = 30


def train_one_step(x, y, model, optimizer):
    with torch.cuda.amp.autocast():
        loss = torch.nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


results = []
for model_setting in [['Tiny', 16, 512], ['Small', 16, 512], ['Base', 16, 512], ['Large', 16, 256], ['Huge', 14, 64], ['Giant', 14, 64]]:
    model_name, patch_size, batch_size = model_setting
    for method in ['pre-ln', 'pre-apex-ln', 'pre-customized-ln', 'no-normalization', 'pre-apex-rms', 'pre-rms', 'pre-crms', 'pre-crms-same-dim']:
        pre_rms_training = method in ['pre-rms', 'pre-customized-rms']
        raw_model = PreDefinedViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, variant=model_name, method=method, pre_rms_training=pre_rms_training).cuda()
        model = torch.compile(raw_model) if using_torch_compile else raw_model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)

        for num_threads in num_threads_list:
            x = torch.randn(batch_size, 3, image_size, image_size).cuda()
            y = torch.randint(0, num_classes, (batch_size,)).cuda()
            result = benchmark.Timer(stmt='train_one_step(x, y, model, optimizer)',
                                     setup='from __main__ import train_one_step',
                                     globals={'x': x, 'y': y, 'model': model, 'optimizer': optimizer},
                                     num_threads=num_threads,
                                     sub_label=f'batch_size {batch_size}, method {method}',
                                     description=model_name,
                                     ).blocked_autorange(min_run_time=min_run_time)
            results.append(result)
            print(result)

compare = benchmark.Compare(results)
compare.print()
