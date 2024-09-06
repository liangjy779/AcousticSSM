#S4_SS model used with just sincnet S4d model
import torch.nn as nn
from models.s4.s4d import S4D
import torch
import os
import sys
import torchsummary
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'models'))
from models.sinc_model import SincConv_fast
class S4Model_SS2(nn.Module):

    def __init__(
        self,
        d_input,
        pt,
        lr = 0.001,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        dropout_fn=nn.Dropout,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.sinc = SincConv_fast(out_channels=16,kernel_size=251,sample_rate=30225)
  
        conv1, bn1 = self.make_layers(16, 64, (1, 15), (1, 5))#kernel width, stride
        conv2, bn2 = self.make_layers(64, 64, (1, 11), (1, 2))
        conv3, bn3 = self.make_layers(64, 128,(1, 11),(1, 1))
        conv4, bn4 = self.make_layers(128, 128,(1, 10),(1, 2))
        conv5, bn5 = self.make_layers(128, 128,(1, 11),(1, 1))
        conv6, bn6 = self.make_layers(128, 128,(1, 11),(1, 2))
        self.sfeb = nn.Sequential(
            #Start: Filter bank
            conv1, bn1, nn.PReLU(),\
            conv2, bn2, nn.PReLU(),\
            conv3, bn3, nn.PReLU(),\
            conv4, bn4, nn.PReLU(),\
            conv5, bn5, nn.PReLU(),\
            conv6, bn6, nn.PReLU(),\
            #nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )
        deconv1,debn1 = self.make_deconvolution(128,128,(1,11),(1,2))
        deconv2,debn2 = self.make_deconvolution(128,128,(1,11),(1,1))
        deconv3,debn3 = self.make_deconvolution(128,128,(1,10),(1,2))
        deconv4,debn4 = self.make_deconvolution(128,64,(1,11),(1,1))  
        deconv5,debn5 = self.make_deconvolution(64,64,(1,11),(1,2))
        deconv6,debn6 = self.make_deconvolution(64,16,(1,15),(1,5))
        self.deconvfeb = nn.Sequential(
            deconv1,debn1,nn.PReLU(),\
            deconv2,debn2,nn.PReLU(),\
            deconv3,debn3,nn.PReLU(),\
            deconv4,debn4,nn.PReLU(),\
            deconv5,debn5,nn.PReLU(),\
            deconv6,debn6,nn.PReLU(),\
        )
        self.deconv_sinc = nn.ConvTranspose2d(16,1,(1,251),(1,1))
        #self.encoder = nn.Linear(d_input, d_model)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                #S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model)) 
            self.dropouts.append(dropout_fn(dropout))

        self.nce_head = nn.Sequential(
            nn.Linear(734, 128),
            nn.GELU(),
            nn.Linear(128, 30225)
        )
        self.pt = pt

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        #x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        B = x.shape[0]
        L = x.shape[1]
        x = x.transpose(1,2)
        x = self.sinc(x)
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

        

        if self.pt:
            #x with shape(16,H,750)(B,H,L)
            x2 = x.mean(dim=1)#(B,L)(16,750)(16,734)
            x = x.unsqueeze(2)#x shape is (B,H,1,L)(16,128,1,750)
            x = self.deconvfeb(x)#shape [16,64,1,29975]
            x = self.deconv_sinc(x)#(B,H,1,L)[16,1,1,30225]
            
            x = x.squeeze(2)
            x = x.transpose(-1, -2)#(B,L,H)(16,30225,1)

            
            y_hat = self.nce_head(x2)#shape(16,30225)
            return x, y_hat
       
        return x
    
    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn
    def make_deconvolution(self,in_channels,out_channels,kernel_size,stride=(1,1),padding=0,bias=False):
        conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn
        
class S4Model_FT2(nn.Module):

    def __init__(
        self,
        pt_path,
        d_input,
        lr = 0.001,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        dropout_fn=nn.Dropout,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.ss = S4Model_SS2(d_input, pt=None, lr=lr, d_output=d_output, d_model=d_model, n_layers=n_layers, dropout=dropout, prenorm=prenorm, dropout_fn=dropout_fn)
        self.ss.load_state_dict(torch.load(pt_path)['model'])
        #self.ss.ss_head = nn.Identity()
        # freeze the weights
        # for param in self.ss.parameters():
        #     param.requires_grad = True

        #self.ss.eval() 

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def freeze_ss(self):
        for param in self.ss.parameters():
            param.requires_grad = False
    
    def unfreeze_ss(self):
        for param in self.ss.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """


        x = self.ss(x)#.squeeze(-1)
        #print(x.shape)
        x = x.mean(dim=2)
        #print(x.shape)
        x = self.decoder(x)

        return x