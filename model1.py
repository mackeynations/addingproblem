import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import regularizer

class RNN(torch.nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.sequence_length = args.seq_length
        self.batchsize = args.batch_size
        self.nlayers = args.nlayers
        self.nhid = args.nhid
        self.weight_decay = args.weight_decay
        self.regtype = int(args.regtype)
        self.device = args.device
        self.embed_dim = 2
        self.moduli = args.regularizer
        
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=args.nhid,
                                nonlinearity=args.activation,
                                bias=True,
                                num_layers = self.nlayers,
                                batch_first = True)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(args.nhid*args.seq_length, 10, bias=True)
        self.decoder2 = torch.nn.Linear(10, 1, bias=True)
        
        self.reg = None
        self.get_regularizer(args)
        self.regtype = args.regtype
        
        #self.init_weights()
        
    def forward(self, x):
        hidden = self.init_hidden(self.batchsize)
        out, _ = self.RNN(x, hidden)
        #print(out.shape)
        decoded = self.decoder(out.reshape(self.batchsize, -1))
        decoded=self.decoder2(F.relu(decoded))
        return decoded
        
        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    def get_regularizer(self, args):
        reg = regularizer.regularizer(args)
        self.reg = reg.reg
        
    def regularizer(self):
        return torch.mean(self.reg*torch.abs(self.rnn.weight_hh_l0)**int(self.regtype))
    
    def init_weights(self):
        self.RNN.weight_hh_l0.data = torch.eye(self.nhid)
        self.RNN.bias_hh_l0.data = torch.zeros(self.nhid)
        nn.init.normal_(self.RNN.weight_ih_l0, mean=0, std=.001)
        nn.init.normal_(self.RNN.bias_ih_l0, mean=0, std=.001)
        nn.init.normal_(self.decoder.weight, mean=0, std=.001)
        nn.init.normal_(self.decoder.bias, mean=0, std=.001)