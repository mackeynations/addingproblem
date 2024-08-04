import torch
import numpy as np
import random

class TrajectoryGenerator(object):
    """ This class creates the input data for the model: pairs of trajectories at different times, and with shuffled coords """
    def __init__(self, args, distr='uniform'):
        self.batchsize = args.batch_size
        self.seq_length = args.seq_length
        self.distr = distr
        
        
        
    def generate_trajectory(self):
        if self.distr == 'normal':
            velos = .15*torch.randn(self.batchsize, self.seq_length)
        elif self.distr == 'uniform':
            velos = torch.rand(self.batchsize, self.seq_length)
            
        mask = torch.zeros(self.batchsize, self.seq_length)
        mask[:,:2] = torch.ones(self.batchsize, 2)
        permutations = torch.rand(self.batchsize, self.seq_length).argsort(dim=1)
        permuted_tensor = mask.gather(1, permutations)
        return torch.stack((velos, permuted_tensor), dim=2), torch.sum(velos*permuted_tensor, dim=1)
        
    
    def get_generator(self):
        """ Returns a generator which yields training data """
        while True:
            # Generate original and shuffled trajectory
            traj = self.generate_trajectory()
            
            # Calculate relative (true north) cumulative positions along the two trajectories
            #tn_cum = torch.cumsum(tn_traj, 1)
            yield traj