import torch
import torch.nn as nn
import numpy as np
import random
import regularizer

class RNN(torch.nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.sequence_length = args.sequence_length
        self.batchsize = args.batchsize
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
                                batch_first = True)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(args.nhid, 1, bias=True)
        
        self.reg = None
        self.get_regularizer(args)
        self.regtype = args.regtype
        
    def forward(self, x):
        hidden = self.init_hidden(self.batchsize)
        out, _ = self.RNN(x, hidden)
        decoded = self.decoder(out[:,-1,:])
        return decoded
        
        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    def get_regularizer(self, args):
        reg = regularizer.regularizer(args)
        self.reg = reg.reg
        
    def regularizer(self):
        return torch.mean(self.reg*torch.abs(self.rnn.weight_hh_l0)**int(self.regtype))