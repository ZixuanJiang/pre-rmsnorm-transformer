import torch
import torch.utils.benchmark as benchmark
from model import PreDefinedGPT3

vocab_size = 50257
max_seq_length = 2048
using_torch_compile = False
device = torch.device('cuda')

batch_size_list = [1, 16]
seq_len_list = [128, 512, 2048]
num_threads_list = [1]
min_run_time = 10

results = []
for model_name in ['Small', 'Medium', 'Large', 'XL', '2.7B', '6.7B', '13B', '175B']:
    for method in ['pre-ln', 'pre-apex-ln', 'pre-customized-ln', 'no-normalization', 'pre-apex-rms', 'pre-rms', 'pre-crms', 'pre-crms-same-dim']:

        raw_model = PreDefinedGPT3(vocab_size=vocab_size, max_seq_length=max_seq_length, variant=model_name, method=method).to(device)
        model = torch.compile(raw_model) if using_torch_compile else raw_model
        model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch_size in batch_size_list:
                    for seq_len in seq_len_list:
                        for num_threads in num_threads_list:
                            x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
                            result = benchmark.Timer(stmt='y = model(x)',
                                                     setup='from __main__ import model',
                                                     globals={'x': x},
                                                     num_threads=num_threads,
                                                     sub_label=f'batch_size {batch_size} seq_len {seq_len}',
                                                     description=model_name + ' ' + method,
                                                     ).blocked_autorange(min_run_time=min_run_time)
                            results.append(result)
                            print(result)

compare = benchmark.Compare(results)
compare.print()
