import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torchaudio.datasets import SPEECHCOMMANDS as _SpeechCommands  # noqa
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

import utils as U
from tqdm.auto import tqdm
from utils import *
from utils_data import *
from models.s4_ss.S4_SS4 import S4Model_SS4
from models.s4_ss.Resnet import S4Model_Resnet
#from models.s4_ss.Resnet0 import S4Model_Resnet0
from copy import deepcopy
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
#function for process dataset
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


parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning')
# Optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=200, type=int, help='Training epochs')
parser.add_argument('--train_steps', default=250, type=int, help='Training steps')
# Dataset
parser.add_argument('--dataset', default='esc50', choices=['mnist', 'cifar10','speechcommand','esc50','testesc50','bio'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader        
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=8, type=int, help='Number of layers')
parser.add_argument('--d_model', default=1, type=int, help='Model dimension')#128
parser.add_argument('--dropout', default=0.01, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--wandb_api_key', required=True, type=str, help='Wandb API key')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--output_dir',  type=str, help='Output directory')

# loss scale
parser.add_argument('--nce_scale', default=0.1, type=float, help='Loss scale')
parser.add_argument('--mae_scale', default=1.0, type=float, help='Loss scale')
parser.add_argument('--cont_scale', default=0.1, type=float, help='Loss scale')
parser.add_argument('--mse_mfcc_scale', default=5e-4, type=float, help='Loss scale')
parser.add_argument('--mse_lps_scale', default=2e-4, type=float, help='Loss scale')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["WANDB_API_KEY"] = args.wandb_api_key

run = wandb.init(project='s4_pt', config=args)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

def preprocess_setup():
        funcs = []
        # if self.opt.strongAugment:
        funcs += [U.random_scale(1.25)]
#needs repair
        funcs += [U.padding(30250 // 2),#has been 30225 before
                  U.random_crop(30250),
                  U.normalize(32768.0)]
                 #U.random_crop(30250)
                 
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
elif args.dataset == 'bio':
    dataset = np.load('/home/jlia/bioacoustics/S4Bioacoustics/s4bio/data/biowav.npz',allow_pickle=True)
    train_sounds = []
    for i in range(0,6):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        train_sounds.extend(sounds)
    data = [(train_sounds[i]) for i in range(0,len(train_sounds))]
elif args.dataset == 'esc50':
    #dataset = np.load('./wav20.npz',allow_pickle=True)
    #dataset=np.load('./wav20.npz',allow_pickle=True)
    dataset = np.load('./wav20.npz',allow_pickle=True)
    train_sounds= []
    train_labels=[]
    for i in range(1,6):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        
        train_sounds.extend(sounds)
        
    #sdata = [(train_sounds[i],train_labels[i]) for i in range(0,len(train_sounds))]
    d_input = 1
    d_output = 50
def collate_fn(batch):
    sounds = torch.stack([item[0]for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return sounds,targets


# Model
print('==> Building model..')
#d_model=args.d_model,
model = S4Model_Resnet(
    d_input=1,
    pt = True,
    d_output=5,
    d_model=512,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
    lr = args.lr,
    dropout_fn=dropout_fn,
)
model = model.to(device)

if device == 'cuda':
    cudnn.benchmark = True


class Trainer:
    def __init__(self, model, train_sounds, args):
        self.model = model
        self.train_sounds = train_sounds
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.L1Loss()
        self.nce = nn.BCEWithLogitsLoss()
        self.cont = nn.CrossEntropyLoss()

        self.optimizer, self.scheduler = self.setup_optimizer(
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs
        )

    def setup_optimizer(self, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(self.model.parameters())

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

    def next_half_mask(self, audio_batch):
        # mask the first/last half of the audio

        batch_size, num_samples, _ = audio_batch.shape

        # Create a mask array
        masks = torch.ones_like(audio_batch, dtype=torch.bool)

        # Apply random masks to each sample in the batch
        for i in range(batch_size):
            mask_indices = torch.randperm(num_samples)[:num_samples // 2]

            masks[i, mask_indices] = False

            # Apply zero masks
            audio_batch[i, mask_indices] = 0

        return audio_batch

    def mask_audio_batch(self, audio_batch, mask_percentage, std_dev):
        # Ensure the mask percentage is between 0 and 100
        if mask_percentage < 0 or mask_percentage > 100:
            raise ValueError("Mask percentage must be between 0 and 100.")

        # Ensure standard deviation is non-negative
        if std_dev < 0:
            raise ValueError("Standard deviation must be non-negative.")

        batch_size, num_samples, _ = audio_batch.shape #(B,L,1)

        # Calculate the number of samples to mask per audio sample
        num_samples_to_mask = int(num_samples * (mask_percentage / 100))

        # Create a mask array
        masks = torch.ones_like(audio_batch, dtype=torch.bool)

        # Values for replacing masked elements
        random_values = torch.randn_like(audio_batch) * std_dev
        random_values = torch.clamp(random_values, min=-1, max=1)

        mask_indices_tensor = []

        # Apply random masks to each sample in the batch
        for i in range(batch_size):
            mask_indices = torch.randperm(num_samples)[:num_samples_to_mask]#pick up randomised indices for masking

            # Decide the type of masking for each element
            # 50% zeros, 50% random Gaussian values
            zero_mask = torch.rand(num_samples_to_mask) < 0.5
            random_mask = ~zero_mask

            masks[i, mask_indices[zero_mask]] = False
            masks[i, mask_indices[random_mask]] = False

            # Apply zero masks
            audio_batch[i, mask_indices[zero_mask]] = 0

            # Apply random value masks
            audio_batch[i, mask_indices[random_mask]] = random_values[i, mask_indices[random_mask]]

            mask_indices_tensor.append(mask_indices)

        # return audio batch and mask index
        mask_indices_tensor = torch.stack(mask_indices_tensor, dim=0)
        return audio_batch, mask_indices_tensor

    def get_masked_tensor(self, tensor, mask_idx):
        gt_tensor = torch.zeros_like(mask_idx.unsqueeze(-1), dtype = torch.float32)
        for i in range(tensor.size(0)):
            gt_tensor[i] = tensor[i, mask_idx[i]]

        return gt_tensor

    def multi_hot(self, labels, num_classes):
        """
        Convert a list of labels to a multi-hot tensor.
        """
        batch_one_hot = torch.zeros(len(labels), num_classes)

        for i, label in enumerate(labels):
            batch_one_hot[i, label] = 1

        return batch_one_hot

    def get_item(self, batchidx):
        sounds = []
        positive = []
        indice = torch.randperm(len(self.train_sounds)-1)[:self.args.batch_size]
        STD = 0
        for i in range(self.args.batch_size):
            sound = self.train_sounds[indice[i]]
            func = preprocess_setup()
            sound1 = sound
            sound2 = sound
            for f in func:
                sound1 = f(sound1)
            for f in func:
                sound2 = f(sound2)
            sound1 = torch.from_numpy(sound1)
            sound1 = sound1.float()
            sound1 = sound1.unsqueeze(-1)#shape[30250,1]
            sound2 = torch.from_numpy(sound2)
            sound2 = sound2.float()
            sound2 = sound2.unsqueeze(-1)#shape[30250,1]

            sounds.append(sound1)
            positive.append(sound2)
            STD += np.std(sound)
        sounds = torch.stack(sounds,dim=0)#shape(8,30250,1)
        positive = torch.stack(positive,dim=0)#shape(8,30250,1)
        s = sounds.squeeze(-1)
        p = positive.squeeze(-1)
        s = batch_to_log_mel_spec_plus_stft(s)#shape(batchsize,1or3,128,60)
        p = batch_to_log_mel_spec_plus_stft(p)#same shape above
        STD = STD/self.args.batch_size
        return s,p,STD

    def compute_features(self, inputs):
        mfcc_inputs, lps_inputs = [], []
        mf = MFCCProcessor1()
        lps = LPS()

        for s in inputs:
            s = s.squeeze(1).to(self.device)
            mfcc_inputs.append(mf(s))
            lps_inputs.append(lps(s))

        mfcc_inputs = torch.stack(mfcc_inputs, dim=0).to(self.device)
        lps_inputs = torch.stack(lps_inputs, dim=0).to(self.device) # B, 1025, 189
       # lps_inputs = lps(inputs.squeeze(-1)).to(device)
        return mfcc_inputs, lps_inputs

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        temperature = 0.2
        for batch_idx in tqdm(range(self.args.train_steps)):
            sounds,positive,STD = self.get_item(batch_idx)
            inputs = torch.cat([sounds,positive],dim=0)
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            #masked_inputs,mask_idx = self.mask_audio_batch(deepcopy(inputs), 30, STD)

            #mfcc_inputs, lps_inputs = self.compute_features(inputs)
            #outputs, pred_mask,x_mfcc,x_lps,cont_output = self.model(masked_inputs)
            cont_output = self.model(inputs)
            cont_output /= temperature
            sparse_labels = torch.arange(sounds.shape[0]).to(device)

            #mae_loss = self.criterion(outputs, inputs)
            #mse_mfcc_loss = self.criterion(x_mfcc,mfcc_inputs)
            #mse_lps_loss = self.criterion(x_lps,lps_inputs)
            cont_loss = self.cont(cont_output,sparse_labels)
            #nce_targets = self.multi_hot(mask_idx, num_classes=inputs.size(1)).to(self.device)
            #nce_loss = self.nce(pred_mask, nce_targets)

            #loss = args.mae_scale * mae_loss + args.nce_scale * nce_loss + args.mse_mfcc_scale * mse_mfcc_loss + args.mse_lps_scale * mse_lps_loss + args.cont_scale * cont_loss
            loss = cont_loss 
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            print('Batch Idx: (%d/%d) | Loss: %.9f' %
                (batch_idx, self.args.train_steps, train_loss/(batch_idx+1))
            )
            # print('mae_loss: %d  cont_loss: %d  mse_mfcc_loss: %d  mse_lps_loss: %d' %
            #       (mae_loss,cont_loss,mse_mfcc_loss,mse_lps_loss)
            #      )
            print(f'cont loss: {cont_loss}')
            wandb.log({
                'cont_loss':cont_loss,
                'loss':loss,
            })
            # wandb.log({
            #     'mae_loss': mae_loss,
            #     'nce_loss': nce_loss,
            #     'cont_loss': cont_loss,
            #     'mse_mfcc_loss': mse_mfcc_loss,
            #     'mse_lps_loss': mse_lps_loss,
            #     'loss': loss,
            # })

        self.scheduler.step()
        os.makedirs(self.args.output_dir, exist_ok=True)
        if epoch % 10 == 0:
            state = {
                'model': self.model.state_dict(),
                'loss': train_loss/(batch_idx+1),
                'epoch': epoch,
            }

            torch.save(state, f'{self.args.output_dir}/ckpt_{epoch}.pth')

# Assuming model and train_sounds are already defined

trainer = Trainer(model, train_sounds, args)
# Then call train and eval as needed
for epoch in tqdm(range(args.epochs)):
    trainer.train(epoch)
