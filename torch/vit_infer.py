import torch
import torch.utils.benchmark as benchmark
from model import PreDefinedViT

image_size = 224
num_classes = 1000
using_torch_compile = False
device = torch.device('cuda')

batch_size_list = [1, 4, 16, 64, 256, 1024]
num_threads_list = [1]
min_run_time = 10

results = []
for model_variant in [['Tiny', 16], ['Small', 16], ['Base', 16], ['Large', 16], ['Huge', 14], ['Giant', 14]]:
    model_name, patch_size = model_variant
    for method in ['pre-ln', 'pre-apex-ln', 'pre-customized-ln', 'no-normalization', 'pre-apex-rms', 'pre-rms', 'pre-crms', 'pre-crms-same-dim']:

        raw_model = PreDefinedViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, variant=model_name, method=method).to(device)
        model = torch.compile(raw_model) if using_torch_compile else raw_model
        model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch_size in batch_size_list:
                    for num_threads in num_threads_list:
                        x = torch.randn(batch_size, 3, image_size, image_size).to(device)
                        result = benchmark.Timer(stmt='y = model(x)',
                                                 setup='from __main__ import model',
                                                 globals={'x': x},
                                                 num_threads=num_threads,
                                                 sub_label=f'batch_size {batch_size} method {method}',
                                                 description=model_name,
                                                 ).blocked_autorange(min_run_time=min_run_time)
                        results.append(result)
                        print(result)

compare = benchmark.Compare(results)
compare.print()
