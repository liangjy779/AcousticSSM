import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from src.models.nn import DropoutNd
# from transformers.activations import ACT2FN
# from transformers.utils import (
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
# )
# from transformers.modeling_outputs import (
#     BaseModelOutputWithNoAttention,
#     BaseModelOutputWithPooling,
#     MaskedLMOutput,
#     MultipleChoiceModelOutput,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutput,
#     TokenClassifierOutput,
# )
# from transformers.modeling_utils import PreTrainedModel, SequenceSummary
# from transformers.utils import logging
# from .configuration_bigs import BiGSConfig

from einops import repeat
from torch.linalg import eigh
_c2r = torch.view_as_real
_r2c = torch.view_as_complex





def log_step_initializer(H=1024, dt_min=0.01, dt_max=1):
    # Generate dt
    log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
    ) + math.log(dt_min)
    return log_dt

try:
    import pykeops #use this to solve kernel computation issue
    from pykeops.torch import Genred
    has_pykeops = True
    print("Pykeops installation found.")

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
        return tensors
    def cauchy_keops(v, z, w):
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'

        cauchy_mult = Genred(
            f'ComplexDivide({expr_num}, {expr_denom})',
            [
                'v = Vj(2)',
                'z = Vi(2)',
                'w = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2*cauchy_mult(v, z, w, backend='GPU')
        return _r2c(r)

    def log_vandermonde_keops(v, x, L):
        expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'v = Vj(2)',
                'x = Vj(2)',
                'l = Vi(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend='GPU')
        return _r2c(r).real

    def log_vandermonde_transpose_keops(u, v, x, L):
        """
        u: ... H L
        v: ... H N
        x: ... H N
        Returns: ... H N

        V = Vandermonde(a, L) : (H N L)
        contract_L(V * u * v)
        """
        expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
        vandermonde_mult = Genred(
            expr,
            [
                'u = Vj(2)',
                'v = Vi(2)',
                'x = Vi(2)',
                'l = Vj(2)',
            ],
            reduction_op='Sum',
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend='GPU')
        return _r2c(r)

except ImportError:
    has_pykeops = False
    print("Switch to torch vandermonde kernel.")


#S4 kernel module
class S4DKernel(nn.Module):
    #dt_min,dt_max -> initialize log steps
    def __init__(self,d_model,N=64,use_pykeops_kernel=False,dt_min=0.01,dt_max=0.1,lr=None):
        super().__init__()
        H = d_model
        log_dt = log_step_initializer(H,dt_min,dt_max)
        C = torch.randn(H,N//2,dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt",log_dt,lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
        
        # log_step = log_step_initializer(1,dt_min,dt_max)
        # self.C = nn.Parameter(torch.normal(0,0.5**0.5,(N,2)))
        # A_re = -0.5 * torch.ones(N)
        # A_im = math.pi * torch.arange(N)

        # self.register_parameter("log_step",nn.Parameter(log_step))
        # self.register_parameter("A_re",nn.Parameter(A_re))
        # self.register_parameter("A_im",nn.Parameter(A_im))
        self.use_pykeops_kernel = use_pykeops_kernel

    def forward(self,L):
        #return (...H,L) where H is the number of channels(default=1)
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        C = C * (torch.exp(dtA)-1.) / A
        if has_pykeops and self.use_pykeops_kernel:
            K = log_vandermonde_keops(C, dtA, L)
        else:
            K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L) 
            K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
         # Materialize parameters
        # dt = torch.exp(self.log_step)  # (H)
        # A = torch.clamp(self.A_re, None, -1e-4) + 1j * self.A_im
        # C = (self.C[..., 0] + 1j * self.C[..., 1]).unsqueeze(0)

        # # Vandermonde multiplication
        # dtA = A * dt.unsqueeze(-1) # (H N)
        # C = C * (torch.exp(dtA)-1.) / A

        # if has_pykeops and self.use_pykeops_kernel:
        #     K = log_vandermonde_keops(C, dtA, L)
        # else:
        #     K = dtA.unsqueeze(-1) * torch.arange(L, device=dtA.device)  # (H N L)
        #     K = torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        
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

class S4dLayer(nn.Module):
    def __init__(self,d_model,d_state=64,dropout=0.0,transposed=True,**kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.dropout = dropout
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))
        #SSM kernel
        self.kernel = S4DKernel(self.h,N=self.n,**kernel_args)
        #pointwise
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self,u):
        #input shape and output shape is (B,L,H)
        
        u = u.transpose(-1,-2)#convert into (B,H,L)
        L = u.size(-1)
        #compute SSM kernel
        k = self.kernel(L=L) #(H,L)
        #convolution
        k_f = torch.fft.rfft(k,n=2*L) #(H L)
        u_f = torch.fft.rfft(u,n=2*L) #(B H L)
        y = torch.fft.irfft(u_f*k_f,n=2*L)[..., :L] #(B H L)
        y = y + u*self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        #convert back to B,L,H
        y = y.transpose(-1,-2)
        return y,None
# class S4dLayer(nn.Module):
#     def __init__(self,config):
#         super.__init__()
#         self.N = config.num_ssm
#         self.D = nn.Parameter(torch.randn(1))
#         self.kernel = S4DKernel(N=self.N,use_pykeops_kernel=config.use_pykeops_kernel)
        
#     def forward(self,u):
#         #input shape and output shape is (B,L,H)
#         #convert into (B,H,L)
#         u = u.transpose(-1,-2)
#         L = u.size(-1)
#         #compute SSM kernel
#         k = self.kernel(L=L) #(H,L)
#         #convolution
#         k_f = torch.fft.rfft(k,n=2*L) #(H L)
#         u_f = torch.fft.rfft(u,n=2*L) #(B H L)
#         y = torch.fft.irfft(u_f*k_f,n=2*L)[..., :L] #(B H L)
#         y = y + u*self.D.unsqueeze(-1)
#         #convert back to B,H,L
#         y = y.transpose(-1,-2)
#         return y

class aBiGSLayer(nn.Module):
    def __init__(self,d_model,d_state=64,dropout=0.0,transposed=True,**kernel_args):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = 3*d_model
        #self.max_seq_length = config.max_position_embeddings
        #self.pre_norm = config.pre_norm
        #self.decode = config.decode
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        #ssm layers
        self.fs4 = S4dLayer(d_model,dropout=dropout,transposed=transposed,**kernel_args)
        self.bs4 = S4dLayer(d_model,dropout=dropout,transposed=transposed,**kernel_args)
        #dense layers
        self.dv1 = nn.Linear(self.hidden_size,self.intermediate_size)
        self.dv2 = nn.Linear(self.hidden_size,self.intermediate_size)

        self.du_forward = nn.Linear(self.hidden_size,self.hidden_size)
        self.du_backward = nn.Linear(self.hidden_size,self.hidden_size)
        self.duc_forward = nn.Linear(self.hidden_size,self.intermediate_size)
        self.duc_backward = nn.Linear(self.hidden_size,self.intermediate_size)

        self.do1 = nn.Linear(self.intermediate_size,self.hidden_size)
        self.do2 = nn.Linear(self.intermediate_size,self.hidden_size)

    def forward(self,hidden_states):
        hidden_residual = hidden_states
        hidden_states = self.LayerNorm(hidden_states)# hidden_states with (shape (B,L,H))
        #gating
        v_forward = nn.functional.gelu(self.dv1(hidden_states))#(B,L,H)->(B,L,intermediate_size)
        v_backward = nn.functional.gelu(self.dv2(hidden_states))#(B,L,H)->(B,L,intermediate_size)

        u_forward = nn.functional.gelu(self.du_forward(hidden_states))#shape(B,L,H)->(B,L,H)
        #make the flipping toward the second dimension (time dimension) 
        u_backward = nn.functional.gelu(self.du_backward(torch.flip(hidden_states,dims=[1])))#(B,L,H)->(B,L,H)
        
        #s4layers
        fs4_output = self.fs4(u_forward)#(B,L,H)
        bs4_output = self.bs4(u_backward)#(B,L,H)
        #instead of sum, we use multiplication
        uc_forward = self.duc_forward(fs4_output[0])#(B,L,intermediate_size)
        uc_backward = torch.flip(self.duc_backward(bs4_output[0]), dims=[1])#(B,L,intermediate_size)

        out_forward = self.do1(nn.functional.gelu(uc_forward * v_forward))#(B,L,H)
        out_backward = self.do2(nn.functional.gelu(uc_backward * v_backward))#(B,L,H)

        hidden_states = hidden_residual + out_forward + out_backward

        return hidden_states

# class BiGSLayer(nn.Module):
#     def __init__(self,d_model,d_state=64,dropout=0.0,transposed=True,**kernel_args):
#         super.__init__()
#         self.num_ssm = config.num_ssm
#         self.max_seq_length = config.max_position_embeddings
#         self.pre_norm = config.pre_norm
#         self.decode = config.decode
#         self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
#         #ssm layers
#         self.fs4 = S4dLayer(config)
#         self.bs4 = S4dLayer(config)
#         #dense layers
#         self.dv = nn.Linear(config.hidden_size,config.intermediate_size)
#         self.du_forward = nn.Linear(config.hidden_size,config.hidden_size)
#         self.du_backward = nn.Linear(config.hidden_size,config.hidden_size)
#         self.duc_forward = nn.Linear(config.hidden_size,config.hidden_size)
#         self.duc_backward = nn.Linear(config.hidden_size,config.hidden_size)
#         self.dol = nn.Linear(config.hidden_size,config.intermediate_size)
#         self.do = nn.Linear(config.intermediate_size,config.hidden_size)

#     def __call__(self,hidden_states):
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
#         uc_forward = self.duc_forward(fs4_output)#(B,L,H)
#         uc_backward = torch.flip(self.duc_backward(bs4_output), dims=[1])#(B,L,H)
#         hidden_states = self.do(nn.functional.gelu(self.dol(uc_forward * uc_backward)) * v)#output with shape(B,L,H)
#         hidden_states = hidden_residual + hidden_states
#         return hidden_states