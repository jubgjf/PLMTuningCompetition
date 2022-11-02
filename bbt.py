import argparse
import os
import random
import time
import cma
import numpy as np
import torch
from config import config
from dataloader import SST2Loader, SNLILoader, DBPediaLoader, QNLILoader, QQPLoader
from metrics import SST2Metric, SNLIMetric, DBPediaMetric, QNLIMetric, QQPMetric
from forward_api import LMForwardAPI

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", default='SST-2', type=str)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str)
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--seed", default=8, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)
args = parser.parse_args()

task_name = args.task_name
n_prompt_tokens = 50
intrinsic_dim = args.intrinsic_dim
batch_size = 32  # no use
budget = args.budget
bound = args.bound
sigma = args.sigma
# bound = math.sqrt(intrinsic_dim)
# if random_proj == 'normal':
#     bound = math.pow(intrinsic_dim, 0.75)
# elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
#     bound = 100
# else:
#     bound = 5
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path

if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'

# fixed hyper-params
if cat_or_add == 'add':
    init_prompt_path = None
else:
    init_prompt_path = './nli_base_prompt.pt'

args.bbt_version = 'bbt'

# log_dir = './v2_logs'
# fitlog.set_log_dir(log_dir)
# fitlog.commit(__file__, fit_msg=save_path)
# fitlog.add_hyper(args)
# fitlog.add_hyper_in_file(__file__)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

tokenizer = config.tokenizer

cache_fn = f"caches/data_{task_name}_{n_prompt_tokens}_{seed}.pt"
DataLoader = {
    'SST-2': SST2Loader,
    'SNLI': SNLILoader,
    'DBPedia': DBPediaLoader,
    'QNLI': QNLILoader,
    'QQP': QQPLoader
}

data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(['train', 'dev'],
                                                                                                  seed)

train_data, dev_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('dev')

for ds in [train_data, dev_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)
print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])

train_data = {
    'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
    'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
    'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
    'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
}
dev_data = {
    'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
    'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
    'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
    'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
}

model_forward_api = LMForwardAPI(
    train_data, dev_data,
    SST2Metric, SNLIMetric, DBPediaMetric, QNLIMetric, QQPMetric,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    loss_type=loss_type,
    init_prompt_path=init_prompt_path,
    device=device,
    inference_framework=inference_framework,
    onnx_model_path=onnx_model_path,
    cat_or_add=cat_or_add,
    intrinsic_dim=intrinsic_dim,
    sigma=sigma,
    print_every=print_every,
    eval_every=eval_every,
    parallel=parallel
)

cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'maxiter': budget if parallel else budget // popsize,
    'verbose': -1,
}
if bound > 0:
    cma_opts['bounds'] = [-1 * bound, 1 * bound]
es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)
print('Population Size: {}'.format(es.popsize))
print('{} Evaluation.'.format('Parallel' if parallel else 'Serial'))
if parallel:
    # expand training data to a larger batch for parallel evaluation
    train_data['input_ids'] = train_data['input_ids'].repeat(es.popsize, 1)
    train_data['attention_mask'] = train_data['attention_mask'].repeat(es.popsize, 1)
    train_data['mask_pos'] = train_data['mask_pos'].repeat(es.popsize)
    train_data['labels'] = train_data['labels'].repeat(es.popsize)

# opt = cma.CMAOptions()
start_time = time.time()
while not es.stop():
    solutions = es.ask()
    if parallel:
        fitnesses = model_forward_api.eval(solutions)
    else:
        fitnesses = [model_forward_api.eval(x) for x in solutions]
    es.tell(solutions, fitnesses)
    # es.logger.add()  # write data to disc to be plotted
    # es.disp()
end_time = time.time()
print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
if not os.path.exists(f'./results/{task_name}/{seed}'):
    os.makedirs(f'./results/{task_name}/{seed}')

torch.save(model_forward_api.linear(torch.tensor(model_forward_api.best_prompt, dtype=torch.float32)),
           f=f'./results/{task_name}/{seed}/best.pt')

# fitlog.finish()
