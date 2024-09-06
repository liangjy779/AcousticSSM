"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D_bi(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        #y = y.transpose(-1,-2)#convert to B,L,H
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
# class S4BiLayer(nn.Module):
#     def __init__(self,d_model,d_state=64,dropout=0.0,transposed=True,**kernel_args):
#         super().__init__()
#         self.hidden_size = d_model
#         self.intermediate_size = 3*d_model
#         #self.max_seq_length = config.max_position_embeddings
#         #self.pre_norm = config.pre_norm
#         #self.decode = config.decode
#         self.LayerNorm = nn.LayerNorm(self.hidden_size)
#         #ssm layers
#         self.fs4 = S4D_bi(d_model,dropout=dropout,transposed=transposed,**kernel_args)
#         self.bs4 = S4D_bi(d_model,dropout=dropout,transposed=transposed,**kernel_args)
#         #dense layers
#         self.dv = nn.Linear(self.hidden_size,self.intermediate_size)
#         self.du_forward = nn.Linear(self.hidden_size,self.hidden_size)
#         self.du_backward = nn.Linear(self.hidden_size,self.hidden_size)
#         self.duc_forward = nn.Linear(self.hidden_size,self.hidden_size)
#         self.duc_backward = nn.Linear(self.hidden_size,self.hidden_size)
#         self.dol = nn.Linear(self.hidden_size,self.intermediate_size)
#         self.do = nn.Linear(self.intermediate_size,self.hidden_size)

#     def forward(self,hidden_states):
#         hidden_residual = hidden_states
#         hidden_states = self.LayerNorm(hidden_states)# hidden_states with (shape (B,L,H))
#         #gating
#         v = nn.functional.gelu(self.dv(hidden_states))#(B,L,H)->(B,L,intermediate_size)
#         u_forward = nn.functional.gelu(self.du_forward(hidden_states))#shape(B,L,H)->(B,L,H)
#         #make the flipping toward the second dimension (time dimension) 
#         u_backward = nn.functional.gelu(self.du_backward(torch.flip(hidden_states,dims=[1])))#(B,L,H)->(B,L,H)
        
#         #s4layers
#         fs4_output = self.fs4(u_forward)#(B,L,H)
#         bs4_output = self.bs4(u_backward)#(B,L,H)
#         #instead of sum, we use multiplication
#         uc_forward = self.duc_forward(fs4_output[0])#(B,L,H)
#         uc_backward = torch.flip(self.duc_backward(bs4_output[0]), dims=[1])#(B,L,H)
#         hidden_states = self.do(nn.functional.gelu(self.dol(uc_forward * uc_backward)) * v)#output with shape(B,L,H)
#         hidden_states = hidden_residual + hidden_states
#         return hidden_states