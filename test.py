"""
Given trained prompt and corresponding template, we ask test API for predictions
Shallow version
We train one prompt after embedding layer, so we use sentence_fn and embedding_and_attention_mask_fn.
Baseline code is in bbt.py
"""

import os
import torch
from test_api import test_api
import numpy as np
import csv
from config import config

tokenizer = config.tokenizer


def sentence_fn_factory(task_name):
    prompt_initialization = tokenizer.decode(list(range(1000, 1050))) + ' . '
    # prompt_initialization = ""
    if task_name == 'SST-2':
        def sentence_fn(test_data):
            return prompt_initialization + test_data + f' . It was {config.mask_token} .'
    elif task_name == 'SNLI':
        def sentence_fn(test_data):
            return prompt_initialization + test_data + f' ? {config.mask_token} , ' + test_data
    elif task_name == 'DBPedia':
        def sentence_fn(test_data):
            return prompt_initialization + test_data + f' . It was {config.mask_token} .'
    elif task_name == 'QNLI':
        def sentence_fn(test_data):
            return prompt_initialization + test_data + f' ? {config.mask_token} , ' + test_data
    elif task_name == 'QQP':
        def sentence_fn(test_data):
            return prompt_initialization + test_data + f' ? {config.mask_token} , ' + test_data
    else:
        raise NotImplementedError

    return sentence_fn


device = 'cuda'


# config.tasks.keys()
for task_name in ["QNLI"]:
    # for seed in config.tasks[task_name]:
    for seed in [8]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sentence_fn = sentence_fn_factory(task_name)
        embedding_and_attention_mask_fn = lambda x, y: (x, y)
        hidden_states_and_attention_mask_fn = lambda i, x, y: (x, y)
        predictions = torch.tensor([], device=device)
        prompt_embedding = torch.load(f"./results/{task_name}/8/best_prompt.pt")
        for res in test_api(
                sentence_fn=sentence_fn,
                prompt_embedding=prompt_embedding,
                embedding_and_attention_mask_fn=embedding_and_attention_mask_fn,
                hidden_states_and_attention_mask_fn=hidden_states_and_attention_mask_fn,
                test_data_path=f'./test_datasets/{task_name}/train/encrypted.pth',
                task_name=task_name,
                device=device
        ):
            print(res)

            pred = res.argmax(dim=-1)
            print(pred)
            predictions = torch.cat([predictions, pred])

        if not os.path.exists(f'./predictions/{task_name}'):
            os.makedirs(f'./predictions/{task_name}')
        with open(f'./predictions/{task_name}/{seed}.csv', 'w+') as f:
            wt = csv.writer(f)
            wt.writerow(['', 'pred'])
            wt.writerows(
                torch.stack([torch.arange(predictions.size(0)), predictions.detach().cpu()]).long().T.numpy().tolist())
