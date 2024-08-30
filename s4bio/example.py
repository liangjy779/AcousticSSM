'''
Train an S4 model on sequential CIFAR10 / sequential MNIST with PyTorch for demonstration purposes.
This code borrows heavily from https://github.com/kuangliu/pytorch-cifar.

This file only depends on the standalone S4 layer
available in /models/s4/

* Train standard sequential CIFAR:
    python -m example
* Train sequential CIFAR grayscale:
    python -m example --grayscale
* Train MNIST:
    python -m example --dataset mnist --d_model 256 --weight_decay 0.0

The `S4Model` class defined in this file provides a simple backbone to train S4 models.
This backbone is a good starting point for many problems, although some tasks (especially generation)
may require using other backbones.

The default CIFAR10 model trained by this file should get
89+% accuracy on the CIFAR10 test set in 80 epochs.

Each epoch takes approximately 7m20s on a T4 GPU (will be much faster on V100 / A100).
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchaudio
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
from torch.nn import functional as F
from torchaudio.datasets import SPEECHCOMMANDS as _SpeechCommands  # noqa
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

import utils as U
from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from models.sinc_model import SincConv_fast
from tqdm.auto import tqdm
import wandb



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
import json
from typing import Any,Optional
_DATASETS_DIRECTORY = Path('./data')
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

class SpeechCommands(SequenceDataset, _SpeechCommands):
    NAME: str = "SPEECH_COMMANDS"
    SEGMENT_SIZE: int = 16_000
    class_names: list[str] = [
        "bed",
        "cat",
        "down",
        "five",
        "forward",
        "go",
        "house",
        "left",
        "marvin",
        "no",
        "on",
        "right",
        "sheila",
        "tree",
        "up",
        "visual",
        "yes",
        "backward",
        "bird",
        "dog",
        "eight",
        "follow",
        "four",
        "happy",
        "learn",
        "nine",
        "off",
        "one",
        "seven",
        "six",
        "stop",
        "three",
        "two",
        "wow",
        "zero",
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(root=self.root_dir, download=True,**kwargs)

        self.label_ids = {l: e for e, l in enumerate(self.all_classes)}
        self._walker = [
            i for i in self._walker if Path(i).parent.name in self.all_classes
        ]

    def _pad(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] == self.SEGMENT_SIZE:
            return y
        elif y.shape[-1] < self.SEGMENT_SIZE:
            return F.pad(y, pad=(0, self.SEGMENT_SIZE - y.shape[-1]))
        else:
            raise IndexError(f"Invalid shape {y.shape}")

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        y, _, label, *_ = super().__getitem__(item)
        y = self._pad(y.squeeze(0))
        y = y.unsqueeze(1)
        return y,self.label_ids[label]
        #return self._pad(y.squeeze(0)), self.label_ids[label]

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.SEGMENT_SIZE,)  # noqa


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.9, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=50, type=float, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='testesc50', choices=['mnist', 'cifar10','speechcommand','esc50','testesc50'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=6, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')#128
parser.add_argument('--dropout', default=0.05, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--wandb_api_key', default='67265bb3f10a02ce2167c5006180fd57e2598daa', type=str, help='Wandb API key')

args = parser.parse_args()
os.environ["WANDB_API_KEY"] = args.wandb_api_key
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

wandb.init(
    # set the wandb project where this run will be logged
    project="s4d",
    entity="bioacoustics",
    name="testing",
    # track hyperparameters and run metadata
    config={
        "dataset": args.dataset,
        "grayscale": args.grayscale,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "prenorm": args.prenorm,
    },
    
)

# Data
print(f'==> Preparing {args.dataset} data..')

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(420),
    )
    return train, val

def preprocess_setup(ncrops):
        funcs = []
        # if self.opt.strongAugment:
        funcs += [U.random_scale(1.25)]
#needs repair
        funcs += [U.padding(30225 // 2),#change from 30225 to 30250
                  #U.random_crop(30225),
                  U.normalize(32768.0),
                 U.multi_crop(30225,ncrops)]
        return funcs
    
if args.dataset == 'cifar10':

    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            #transforms.Normalize(),
            transforms.Lambda(lambda x: x.view(1, 1024).t())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3 if not args.grayscale else 1
    d_output = 10

elif args.dataset == 'mnist':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
    ])
    transform_train = transform_test = transform

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)

    d_input = 1
    d_output = 10
elif args.dataset == 'speechcommand':
    speechdata = SpeechCommands()
    trainset,valset = split_train_val(speechdata,val_split=0.1)
    #_,valset = split_train_val(speechdata,val_split=0.1)
    _,testset = split_train_val(speechdata,val_split=0.2)
    d_input = 1
    d_output = 35#there are 35 labels in speechcommands data
elif args.dataset == 'esc50':
    dataset = np.load('./wav20.npz',allow_pickle=True);
    train_sounds= []
    train_labels=[]
    for i in range(1,6):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        train_sounds.extend(sounds)
        train_labels.extend(labels)
    data = [(train_sounds[i],train_labels[i]) for i in range(0,len(train_sounds))];
    sounds = []
    labels = [];#thes two store sounds & labels after preprocess
    da = []
    ncrops=5
    for  i in range(0,len(train_sounds)):
        sound,target = data[i]
        func = preprocess_setup(ncrops)
        for f in func:
            sound = f(sound)
        label = np.zeros((ncrops,50))
        label[:,target] = 1
        
        #sound.reshape((len(sound),1))
        #transform = transforms.ToTensor()
        #sound = transform(sound)
        #mix with randomised data
        sound = torch.from_numpy(sound)
        sound = sound.float()
        
        s = sound.unsqueeze(-1)
        
        sounds.extend(s)
        #print(sounds[0].shape)
        labels.extend(label)
        for j in range(ncrops):
            #da.append((s[j],label[j]))
            da.append((s[j],target))
   #需要解决------------------ trainset,_ = split_train_val(speechdata,val_split=0.1)
    trainset,valset = split_train_val(da,val_split=0.1)
    #_,valset = split_train_val(da,val_split=0.1)
    _,testset = split_train_val(da,val_split=0.2)
    d_input = 1
    d_output = 50
elif args.dataset == 'testesc50':
    dataset = np.load('/home/jlia/bioacoustics/S4acoustic/s4bio/wav20few.npz',allow_pickle=True)
    train_sounds = []
    train_labels = []
    test_sounds = []
    test_labels = []
    for i in range(0,5):
        sounds = dataset['class{}'.format(i)].item()['sounds']
        labels = dataset['class{}'.format(i)].item()['labels']
        #mask = torch.ones_like(torch.tensor(labels),dtype=torch.bool)
        mask = torch.tensor(range(40))
        num = 5
        ran_num = torch.randperm(40)[:num]
        test_mask = torch.ones_like(torch.tensor(labels),dtype=torch.bool)
        test_mask[ran_num] = False 
        train_mask = ~test_mask
        train_index = mask[train_mask]
        test_index = mask[test_mask]
        for j in range(len(train_index)):
            train_sounds.append(sounds[train_index[j]])
            train_labels.append(labels[train_index[j]])
        for k in range(len(test_index)):
            test_sounds.append(sounds[test_index[k]])
            test_labels.append(labels[test_index[k]])
    train_da = []
    test_da = []
    ncropst=10
    ncrops = 10
    STD=0
    for i in range(0,len(train_sounds)):
        sound,target = train_sounds[i],train_labels[i]
        func = preprocess_setup(ncropst)
        for f in func:
            sound = f(sound)
        label = np.ones((ncropst,1)) * target
        STD += np.std(sound)
        sound = torch.from_numpy(sound)
        sound = sound.float()

        s = sound.unsqueeze(-1)
        for j in range(ncropst):
            train_da.append((s[j],target))
    STD /= len(train_sounds)
    for i in range(0,len(test_sounds)):
        sound,target = test_sounds[i],test_labels[i]
        func = preprocess_setup(ncrops)
        for f in func:
            sound = f(sound)
        label = np.ones((ncropst,1)) * target
        STD += np.std(sound)
        sound = torch.from_numpy(sound)
        sound = sound.float()
        s = sound.unsqueeze(-1)
        for j in range(ncrops):
            test_da.append((s[j],target))
    trainset,_ = split_train_val(train_da,val_split=0.0)
    testset,valset = split_train_val(test_da,val_split=0.2)
    d_input = 1
    d_output = 5
        
        
        
#elif args.dataset == 'speechcommands':
#    transform = 
#else: raise NotImplementedError

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=32, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=args.num_workers)

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        #self.encoder = nn.Linear(d_input, d_model)
        self.sinc = SincConv_fast(out_channels=16,kernel_size=251,sample_rate=30225)
        # conv1, bn1 = self.make_layers(1, 8, (1, 9), (1, 2));
        # conv2, bn2 = self.make_layers(8, 64, (1, 5), (1, 2));
        conv1,bn1 = self.make_layers(16,64,(1,15),(1,8))
        conv2,bn2 = self.make_layers(64,128,(1,6),(1,2))
        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.ReLU(),\
            conv2, bn2, nn.ReLU(),\
            #nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                #S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model)) 
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        #x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        B = x.shape[0]
        L = x.shape[1]
        x = x.transpose(1,2)
        #x = x.reshape(B,1,1,L)
        x = self.sinc(x)#sincnet waveform
        x = x.reshape(B,16,1,-1)
        x = self.sfeb(x) 
        x = x.squeeze(2)
        #x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def setup_optimizer(model, lr, weight_decay, epochs):
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
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )
    scheduler.step()

    return train_loss/(batch_idx+1), 100.*correct/total


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        return acc

pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))

    train_loss, train_acc = train()
    val_acc = eval(epoch, valloader, checkpoint=True)
    test_acc = eval(epoch, testloader)

    wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc})
    
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")

