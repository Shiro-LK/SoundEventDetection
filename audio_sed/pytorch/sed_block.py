import torch
from torch import nn
import numpy as np 
from torch.cuda.amp import  autocast
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union

## -- model -- ##


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
    
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        #self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        #init_bn(self.bn_att)
         
    def forward(self, x): # (bs, hidden size, step/channel)
        att =   torch.clamp(self.att(x), -10, 10) #  torch.tanh(self.att(x)) #
        norm_att = torch.softmax(att, dim=-1) # apply softmax on step dim
        cla = self.nonlinear_transform(self.cla(x)) 
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'softmax':
            return torch.softmax(x, dim=-1)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)   
        elif self.activation == 'swish':
            return x * torch.sigmoid(x)
        elif self.activation == 'elu': 
            return F.elu(x)
        else:
            raise("Attention Block activation function not accepted. Use only [linear, sigmoid, softmax]")
        
class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention , self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h_i):
        u_i = self.W(h_i)
        a_i =  F.softmax(u_i, dim=1) 
        v = (a_i * h_i).sum(1)   
        return v, a_i

class Transpose(nn.Module):
    def __init__(self, axis1, axis2):
        super(Transpose , self).__init__()
        self.axis1 = axis1 
        self.axis2 = axis2

    def forward(self, inputs):
        return torch.transpose(inputs, self.axis1, self.axis2)

class SED_Block(nn.Module):
    def __init__(self, num_classes:List[int], in_features:int, hidden_size:int = 1024, activation:int = "sigmoid",
                drop_rate:List[int] = [0.5, 0.5], apply_attention="step"):
        """Apply SED block on output of a backbone.
        """
        super(SED_Block, self).__init__()
        assert apply_attention in ["step", "channel"]
        self.apply_attention = apply_attention
        self.num_classes = num_classes
        #self.dropout = nn.Dropout(drop_rate[0])
        self.blocks = nn.ModuleList()
        for num_class in self.num_classes:
            # 1 ) att on step => (bs, channel, step) = > transpose => (bs, step, channel) = > linear => (bs, step, hidden size)
            # => transpose => (bs, hidden size, step) => attblock on hstep => (bs , num_classes, step )
            # 2 ) att on channels => (bs, channel, step) = > linear => (bs, channel, hidden size)
            # => transpose => (bs, hidden size, channel) => attblock on channel => (bs , num_classes, channel  ) => (bs , num_classes,   )
            self.blocks.append(nn.Sequential(nn.Dropout(drop_rate[0]),
                                            Transpose(1,2) if self.apply_attention == "step" else nn.Identity(), 
                                            nn.Linear(in_features, hidden_size),
                                            nn.ReLU(),
                                            Transpose(1,2) ,
                                            nn.Dropout(drop_rate[1]),
                                            AttBlock(hidden_size,  num_class, activation=activation, temperature=1.)))

    def forward(self, inputs):
        """
            inputs: (Batch size, channels, steps, freq)
            @return: List[num_classes] each element of the list following this format : 
                    {'clipwise':(BATCH SIZE, n), 'segmentwise':(BATCH SIZE, step, n),
                    'norm_att':(BATCH SIZE, step)}
        """
        outputs = []
        x = torch.mean(inputs, dim=3) # Average freq features
    
        # smooth channels
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1) # pool on steps
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        for block in self.blocks:
            (clipwise_output, norm_att, segmentwise_output) = block(x)
            segmentwise_output = segmentwise_output.transpose(1, 2) # batch size, steps, channels
            outputs.append({"clipwise":clipwise_output, "segmentwise":segmentwise_output, 
            "norm_att":norm_att})
        return outputs
        

