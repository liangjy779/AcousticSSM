import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Binomial
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm
import numpy as np
import json
import os


class NeuralBlock(nn.Module):

    def __init__(self, name='NeuralBlock'):
        super().__init__()
        self.name = name

	# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    def describe_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print('-' * 10)
        print(self)
        print('Num params: ', pp)
        print('-' * 10)
        return pp

class Model(NeuralBlock):

    def __init__(self, max_ckpts=5, name='BaseModel'):
        super().__init__()
        self.name = name
        self.optim = None
        self.max_ckpts = max_ckpts

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name

        if not hasattr(self, 'saver') and saver is None:
            self.saver = Saver(self, save_path,
                               optimizer=self.optim,
                               prefix=model_name + '-',
                               max_ckpts=self.max_ckpts)

        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            # save with specific saver
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, 
                                   optimizer=self.optim,
                                   prefix=self.name + '-',
                                   max_ckpts=self.max_ckpts)
            self.saver.load_weights()
        else:
            print('Loading ckpt from ckpt: ', save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False, verbose=True):
        # tmp saver
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last, verbose=verbose)


    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_total_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def describe_params(self):
        pp = 0
        if hasattr(self, 'blocks'):
            for b in self.blocks:
                p = b.describe_params()
                pp += p
        else:
            print('Warning: did not find a list of blocks...')
            print('Just printing all params calculation.')
        total_params = self.get_total_params()
        print('{} total params: {}'.format(self.name,
                                           total_params))
        return total_params
    
class MLPBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, din=0, dout=0, context=1, 
                 tie_context_weights=False, name='MLPBlock',
                 ratio_fixed=None, range_fixed=None, 
                 dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        self.ninp = ninp
        self.fmaps = fmaps
        self.tie_context_weights = tie_context_weights
        assert context % 2 != 0, context
        if tie_context_weights:
            self.W = nn.Conv1d(ninp, fmaps, 1)
            self.pool = nn.AvgPool1d(kernel_size=context, stride=1,
                                      padding=context//2, count_include_pad=False)
        else:
            self.W = nn.Conv1d(ninp, fmaps, context, padding=context//2)

        self.din = PatternedDropout(emb_size=emb_size, p=din, 
                                    dropout_mode=dropin_mode,
                                    range_fixed=range_fixed,
                                    ratio_fixed=ratio_fixed,
                                    drop_whole_channels=drop_channels)
        self.act = nn.PReLU(fmaps)
        self.dout = nn.Dropout(dout)

    def forward(self, x, device=None):
        if self.tie_context_weights:
            return self.dout(self.act(self.pool(self.W(self.din(x)))))
        return self.dout(self.act(self.W(self.din(x))))
class MLPMinion(Model):

    def __init__(self, num_inputs,
                 num_outputs,
                 dropout, dropout_time=0.0,hidden_size=256,
                 dropin=0.0,
                 hidden_layers=2,
                 context=1,
                 tie_context_weights=False,
                 skip=True,
                 loss=None,
                 loss_weight=1.,
                 keys=None,
                 augment=False,
                 r=1, 
                 name='MLPMinion',
                 ratio_fixed=None, range_fixed=None, 
                 dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        # Implemented with Conv1d layers to not
        # transpose anything in time, such that
        # frontend and minions are attached very simply
        self.num_inputs = num_inputs
        assert context % 2 != 0, context
        self.context = context
        self.tie_context_weights = tie_context_weights
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        # r frames predicted at once in the output
        self.r = r
        # multiplies number of output dims
        self.num_outputs = num_outputs * r
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for hi in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp,
                                        hidden_size,
                                        din=dropin,
                                        dout=dropout,
                                        context=context,
                                        tie_context_weights=tie_context_weights,
                                        emb_size=emb_size, 
                                        dropin_mode=dropin_mode,
                                        range_fixed=range_fixed,
                                        ratio_fixed=ratio_fixed,
                                        drop_channels=drop_channels))
            ninp = hidden_size
            # in case context has been assigned,
            # it is overwritten to 1
            context = 1
        self.W = nn.Conv1d(ninp, self.num_outputs, context,
                           padding=context//2)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        
        if self.dropout_time > 0 and self.context > 1:
            mask=(torch.FloatTensor(x.shape[0],x.shape[2]).to('cuda').uniform_() > self.dropout_time).float().unsqueeze(1)
            x=x*mask

        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y