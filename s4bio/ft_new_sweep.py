import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import yaml
import math

import utils as U
from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm
from models.s4_ss.S4_SS2 import S4Model_FT2
from models.s4_ss.S4_SS import S4Model_FT
from models.s4_ss.S4_SS4 import S4Model_FT4
from models.s4_ss.S4_SS5 import S4Model_FT5
from models.s4_ss.S4_SS6 import S4Model_FT6
from utils import select_possible_augs
from copy import deepcopy
from SAM.sam import SAM


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

from pathlib import Path

import requests
from tqdm import tqdm
import wandb
import json
from typing import Any,Optional
_DATASETS_DIRECTORY = Path('./data')


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# Dataset
parser.add_argument('--dataset', default='FS_esc50', choices=['mnist', 'cifar10','speechcommand','esc50','FS_esc50'], type=str, help='Dataset')
parser.add_argument('--num_class', default=50, type=int, help='Number of classes in the dataset')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
# Model
parser.add_argument('--n_layers', default=6, type=int, help='Number of layers')
parser.add_argument('--d_model', default=256, type=int, help='Model dimension')#128
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--wandb_api_key', default='67265bb3f10a02ce2167c5006180fd57e2598daa', type=str, help='Wandb API key')
parser.add_argument('--output_dir', required=True, type=str, help='Output directory')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
parser.add_argument('--ways', default=5, type=int, help='Number of classes in a task')
parser.add_argument('--shots', default=5, type=int, help='Number of samples per class in a task')
parser.add_argument('--pt_path', type=str, help='Path to the pre-trained model')
parser.add_argument("--project", type=str, help="Wandb project name")
parser.add_argument("--run_name", type=str, help="Wandb run name")


args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["WANDB_API_KEY"] = args.wandb_api_key

# fix all seeds and random process
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
GENERATOR_SEED = torch.Generator().manual_seed(args.seed)

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=GENERATOR_SEED,
    )
    return train, val

def preprocess_setup_train():
        funcs = []
        # if self.opt.strongAugment:
        #funcs += [U.random_scale(1.25)]
#needs repair
        funcs += [U.padding(30250 // 2),
                  U.random_crop(30225),
                  U.normalize(32768.0)]
                 #U.multi_crop(30250,ncrops)]
        return funcs

def preprocess_setup_test():
        funcs = []
        # if self.opt.strongAugment:
        #funcs += [U.random_scale(1.25)]

        funcs += [U.padding(30250 // 2),
                    U.center_crop(30225),
                    U.normalize(32768.0)]
                     #U.multi_crop(30250,ncrops)]
        return funcs



def download(
    url: str,
    dst: Path,
    chunk_size: int = 1024,
    verbose: bool = True,
) -> Path:
    """Download a file from a ``url`` to ``dst``.

    Args:
        url (str): URL of the file to download
        dst (Path): download destination. If a directory, the filename
            will be determined using the URL.
        chunk_size (int): size of "chunks" to use when streaming the data
        verbose (bool): if ``True`` display a progress bar

    Returns:
        dst (Path): the path to the downloaded file

    """
    if dst.is_dir():
        dst = dst.joinpath(Path(url).name)

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0)) or None
    with dst.open("wb") as file:
        with tqdm(
            desc=f"Downloading {Path(url).name}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
            disable=total is None or not verbose,
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                pbar.update(size)
    return dst

class SequenceDataset:
    NAME: Optional[str] = None
    SAVE_NAME: Optional[str] = None
    class_names: Optional[list[str | int]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def root_dir(self) -> Path:
        """Directory where data is stored."""
        name = self.SAVE_NAME or self.NAME
        if not isinstance(name, str):
            raise TypeError("`NAME` not set")

        path = _DATASETS_DIRECTORY.expanduser().joinpath(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def all_classes(self) -> list[str | int]:
        """Names of all classes in the dataset."""
        if self.class_names:
            return self.class_names
        else:
            raise AttributeError("Class names not set")

    @property
    def n_classes(self) -> int:
        """Number of class_names in the dataset."""
        return len(self.all_classes)

    @property
    def channels(self) -> int:
        """Channels in the data, as returned by the dataset."""
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data in the dataset."""
        raise NotImplementedError()


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print(f'==> Preparing {args.dataset} data..')

def build_dataset(args,ways,shots,task_id):
    if args.dataset == 'FS_esc50':
        dataset = np.load('./wav20few_all.npz',allow_pickle=True)
        train_sounds = []
        train_labels = []
        test_sounds = []
        test_labels = []
        for i in range(task_id * ways,(task_id+1)*ways):
            sounds = dataset['class{}'.format(i)].item()['sounds']
            labels = dataset['class{}'.format(i)].item()['labels']
            mask = torch.tensor(range(40))
            num = shots
            ran_num = torch.randperm(40)[:num]
            test_mask = torch.ones_like(torch.tensor(labels),dtype=torch.bool)
            test_mask[ran_num] = False 
            train_mask = ~test_mask
            train_index = mask[train_mask]
            test_index = mask[test_mask]
            for j in range(len(train_index)):#train_sounds has 50
                train_sounds.append(sounds[train_index[j]])
                train_labels.append(labels[train_index[j]] % ways)
            for k in range(len(test_index)):#test_sounds has 150
                test_sounds.append(sounds[test_index[k]])
                test_labels.append(labels[test_index[k]] % ways)

        return train_sounds,train_labels,test_sounds,test_labels
    

def get_itemtrain(batch_size,train_sounds,train_labels):
        trainsounds = []
        trainlabels = []
        indices = torch.randint(0, len(train_sounds), (batch_size,))
        for i in range(batch_size): 
            sound,target = train_sounds[indices[i]],train_labels[indices[i]]

            func = preprocess_setup_train()
            for f in func:
                sound = f(sound)
            aug,_ = select_possible_augs()
            sound1 = sound.reshape(1,1,len(sound))

            sound1 = torch.from_numpy(sound1)
            sound1 = sound1.float()

            sound1 = aug(sound1)

            sound1 = sound1.squeeze(0).T


            trainsounds.append(sound1)
            trainlabels.append(target)
        trainsounds = torch.stack(trainsounds,dim=0)
        trainlabels = torch.tensor(trainlabels)

        return trainsounds,trainlabels

def get_itemtest(batch_size,test_sounds,test_labels, start_idx):
        testsounds = []
        testlabels = []
        batch_size = batch_size if start_idx + batch_size < len(test_sounds) else len(test_sounds) - start_idx
        for j in range(batch_size):
            sound,target = test_sounds[start_idx+j],test_labels[start_idx+j]
            func = preprocess_setup_test()
            for f in func:
                sound = f(sound)
            sound1 = sound.reshape(1,1,len(sound))
            sound1 = torch.from_numpy(sound1)
            sound1 = sound1.float()
            sound1 = sound1.squeeze(0).T

            testsounds.append(sound1)
            testlabels.append(target)

        testsounds = torch.stack(testsounds,dim=0)
        testlabels = torch.tensor(testlabels)

        return testsounds,testlabels

# def build_dataset(args, ways, shots, task_id, batch_size):
#     if args.dataset == 'FS_esc50':
#         dataset = np.load('./wav20few_all.npz',allow_pickle=True)
#         train_sounds = []
#         train_labels = []
#         test_sounds = []
#         test_labels = []
#         for i in range(task_id * ways, (task_id + 1) * ways):
#             sounds = dataset['class{}'.format(i)].item()['sounds']
#             labels = dataset['class{}'.format(i)].item()['labels']
#             mask = torch.tensor(range(40))
#             num = shots * 2
#             ran_num = torch.randperm(40)[:num]
#             test_mask = torch.ones_like(torch.tensor(labels),dtype=torch.bool)
#             test_mask[ran_num] = False 
#             train_mask = ~test_mask
#             train_index = mask[train_mask]
#             test_index = mask[test_mask]
#             for j in range(len(train_index)):
#                 train_sounds.append(sounds[train_index[j]])
#                 train_labels.append(labels[train_index[j]])
#             for k in range(len(test_index)):
#                 test_sounds.append(sounds[test_index[k]])
#                 test_labels.append(labels[test_index[k]])
#         train_da = []
#         test_da = []
#         ncropst = 10
#         ncrops = 10
#         for i in range(0,len(train_sounds)):
#             sound,target = train_sounds[i],train_labels[i]

#             target = target - task_id * ways # ensure the target is in the range of ways

#             func = preprocess_setup(ncrops)
#             for f in func:
#                 sound = f(sound)
#             sound = torch.from_numpy(sound)
#             sound = sound.float()

#             s = sound.unsqueeze(-1)
#             for j in range(ncropst):
#                 train_da.append((s[j],target))

#         for i in range(0,len(test_sounds)):
#             sound,target = test_sounds[i],test_labels[i]

#             target = target - task_id * ways # ensure the target is in the range of ways


#             func = preprocess_setup(ncropst)
#             for f in func:
#                 sound = f(sound)
#             sound = torch.from_numpy(sound)
#             sound = sound.float()
#             s = sound.unsqueeze(-1)
#             for j in range(ncropst):
#                 test_da.append((s[j],target))

#         trainset, valset = split_train_val(train_da,val_split=0.5)
#         testset = test_da



#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, generator=GENERATOR_SEED, collate_fn=collate_fn)
#     valloader = torch.utils.data.DataLoader(
#         valset, batch_size=32, shuffle=False, num_workers=args.num_workers, generator=GENERATOR_SEED)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=32, shuffle=False, num_workers=args.num_workers, generator=GENERATOR_SEED)
    
#     return trainloader, valloader, testloader


def build_model(args, lr, dropout, device):
    print('==> Building model..')
    model = S4Model_FT6(
        pt_path=args.pt_path,
        d_input=1,
        d_output=args.ways,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout= dropout,
        prenorm=args.prenorm,
        lr = lr,
        dropout_fn=dropout_fn,
    )

    model = model.to(device)

    return model

def build_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=lr, weight_decay=weight_decay)
    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min = 1e-7)
    milsetone_neg = epochs // 4
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs - milsetone_neg], gamma=0.1)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

# load yaml config
config = yaml.load(open('sweep_config/ft_config_new.yaml'), Loader=yaml.FullLoader)


def collate_fn(batch):
    aug, _ = select_possible_augs()
    data = [aug(item[0]) for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data, dim=0)
    target = torch.tensor(target)
    return data, target

# def ema_update(model, ori_model, alpha=0.9):
#     for param, ori_param in zip(model.parameters(), ori_model.parameters()):
#         param.data = (1-alpha) * param.data + alpha * ori_param.data

def train_epoch(config, model, train_sounds, train_labels, criterion, optimizer, scheduler, device):
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx in tqdm(range(config.train_steps)):

        inputs, targets = get_itemtrain(config.batch_size,train_sounds,train_labels)
        
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.first_step(zero_grad=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += targets.shape[0]
        correct += predicted.eq(targets).sum().item()

    scheduler.step()

    train_loss = train_loss/(batch_idx+1)
    train_acc = 100.*correct/total


    
    print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}')

    return train_loss, train_acc

def eval(config, model, test_sounds, test_labels, criterion, device, run_name, test_id, task_id, best_acc, checkpoint=False):

    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        
        for batch_idx in tqdm(range(int(math.ceil(len(test_labels)/config.batch_size)))):

            inputs, targets = get_itemtest(config.batch_size,test_sounds,test_labels,batch_idx*config.batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            predicted = outputs.argmax(dim = 1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    eval_loss = eval_loss/(batch_idx+1)
    eval_acc = 100.*correct/total

    # Save checkpoint.
    if checkpoint:

        if eval_acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': eval_acc,
            }
            out_dir = f'{args.output_dir}/{run_name}/{test_id}/{task_id}'
            os.makedirs(out_dir, exist_ok=True)
            torch.save(state, f'{out_dir}/ckpt.pth')
            best_acc = eval_acc

    return eval_loss, eval_acc




def train(args, config, device):
    # Initialize a new wandb run
    criterion = nn.CrossEntropyLoss()
    test_id = 0
    with wandb.init(config=config):
        print(f'==> Starting test {test_id}..')
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        wandb.log({'n_layers': args.n_layers, 'd_model': args.d_model, 'prenorm': args.prenorm, 'ways': args.ways, 'shots': args.shots, 'seed': args.seed, 'dataset': args.dataset, 'pt_path': args.pt_path, 'test_id': test_id})

        config = wandb.config
        average_test_acc = 0
        num_tasks = args.num_class // args.ways

        for task_id in range(num_tasks):
            print(f'Task ID: {task_id}')
            best_acc = 0

            train_sounds,train_labels,test_sounds,test_labels = build_dataset(args, args.ways, args.shots, task_id)

            model = build_model(args, config.learning_rate, config.dropout, device)
            optimizer, scheduler = build_optimizer(model, config.learning_rate, config.weight_decay, config.epochs)

            for epoch in range(config.epochs):
                print(f'Epoch: {epoch}')
                train_loss, train_acc = train_epoch(config, model, train_sounds, train_labels, criterion, optimizer, scheduler, device)

                wandb.log({
                    'Train Loss': train_loss,
                    'Train Acc': train_acc,
                })

            
                test_loss, test_acc = eval(config, model, test_sounds, test_labels, criterion, device, epoch, test_id, task_id, best_acc, checkpoint=True)
                print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f}')

            wandb.log({
                'Test Loss': test_loss,
                'Test Acc': test_acc,
                'Task ID': task_id,
            })

            average_test_acc += test_acc
        
        average_test_acc = average_test_acc / num_tasks
        print(f'Average Test Accuracy: {average_test_acc:.3f}')

        wandb.log({
            'Average Test Accuracy': average_test_acc,
        })

        test_id += 1

swep_id = wandb.sweep(config, project=args.project)
wandb.agent(swep_id, function=lambda: train(args, config, device))



