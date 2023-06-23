"""
Vertically partitioned SplitNN implementation

Clients own vertically partitioned data
Server owns the labels
"""


import syft as sy
import torch
from torch import nn, Tensor
import torch.distributed as dist
import numpy as np
import random
import time

from src.utils.fagin_utils import split_samples_by_class, get_kth_dist, digamma, get_sorted_distances

hook = sy.TorchHook(torch)
DEFAULT_METHOD = "zeros"
MEAN_DELAY = 0
STD_DELAY = 0
PROBABILITY_OF_TESTING = 0.5
PROBABILITY_OF_PICKING = 0.5
METHOD = 'RANDOM'

class RandomDiscreteSplitNN:
    def __init__(self, models, server, data_owners, optimizers, dist_data, k, n_selected, padding_method=DEFAULT_METHOD):
        self.models = models
        self.server = server
        self.data_owners = data_owners
        self.optimizers = optimizers
        self.selected = {}
        self.selected[server.id] = True
        for owner in data_owners:
            self.selected[owner.id] = True

        self.PADDING_METHOD = padding_method
        self.latest = {}
        self.means = {}
        self.wei = {}
        self.counters = {}
        for owner in self.data_owners:
            self.counters[owner.id] = 0

        self.classes = None
        self.k = k
        self.n_selected = n_selected
        self.dist_data = dist_data

        self.Q = 1
        self.N = len(self.dist_data)
        self.Nq = {}
        self.mq = {}
        self.seed_count = 1
        
    def generate_data(self, owner, remote_outputs):
        res = None
        if self.PADDING_METHOD == "latest":
            if not owner.id in self.latest:
                res = torch.zeros([64, 64]).send(self.server)
            else:
                res = self.latest[owner.id]
        elif self.PADDING_METHOD == "mean": 
            if not owner.id in self.means:
                res = torch.zeros([64, 64]).send(self.server)
            else:
                res = self.means[owner.id]
        elif self.PADDING_METHOD == "wei": 
            if not owner.id in self.means:
                res = torch.zeros([64, 64]).send(self.server)
            else:
                res = self.wei[owner.id]
        elif self.PADDING_METHOD == "zeros":
            res = torch.zeros([64, 64]).send(self.server)
        else:
            raise Exception("Padding method not supported")
        return res

    def predict(self, data_pointer):        
        #outputs that is moved to server and subjected to concatenate for server input
        remote_outputs = []
        delays = []
        
        #iterate over each client and pass thier inputs to respective model segment and send outputs to server
        missing = []
        counter = 0
        for owner in self.data_owners:
            if self.selected[owner.id]:
                remote_outputs.append(
                    self.models[owner.id](data_pointer[owner.id].reshape([-1, 7*28])).move(self.server)
                )
                delays.append(max(random.gauss(MEAN_DELAY, STD_DELAY), 0))

                # latest padding update
                self.latest[owner.id] = Tensor.copy(remote_outputs[-1])

                # mean padding update
                self.counters[owner.id] += 1
                if not owner.id in self.means:
                    self.means[owner.id] = remote_outputs[-1]
                else:
                    self.means[owner.id] = torch.div(torch.add(torch.mul(self.means[owner.id], self.counters[owner.id]-1), remote_outputs[-1]), float(self.counters[owner.id]))
                # wei padding update
                if not owner.id in self.wei:
                    self.wei[owner.id] = remote_outputs[-1]
                else:
                    self.wei[owner.id] = torch.add(torch.mul(self.means[owner.id], 0.5), torch.div(remote_outputs[-1], 2))
            else:
                missing.append(counter)
            counter += 1

        for miss_index in missing:
            remote_outputs.insert(
                miss_index, self.generate_data(self.data_owners[miss_index], remote_outputs)
            )
        
        # wait for all outputs to arrive
        time.sleep(max(delays))
        
        #concat outputs from all clients at server's location
        server_input = torch.cat(remote_outputs, 1)
        
        #pass concatenated output from server's model segment
        pred = self.models["server"](server_input)
        
        return pred

    def train(self, data_pointer, target):

        #make grads zero
        for opt in self.optimizers:
            opt[0].zero_grad()
        
        #predict the output
        pred = self.predict(data_pointer)
        
        #calculate loss
        criterion = nn.NLLLoss()
        loss = criterion(pred, target.reshape(-1, 64)[0])
        
        #backpropagate
        loss.backward()
        
        #optimization step
        for opt in self.optimizers:
            if self.selected[opt[1].id]:
                opt[0].step()
            
        return loss.detach().get()
    
    def eval(self, data_pointer, target):
        pred = self.predict(data_pointer)
        
        #calculate loss
        criterion = nn.NLLLoss()
        loss = criterion(pred, target.reshape(-1, 64)[0])
        
        return loss.detach().get()

    
    def group_testing(self):
        self.seed_count += 1
        random.seed(self.seed_count)
        for own in self.data_owners:
            self.selected[own.id] = False
        if METHOD == 'RANDOM':
            while(sum([self.selected[ow] for ow in self.selected]) != self.n_selected + 1):
                for own in self.data_owners:
                    if random.random() < PROBABILITY_OF_PICKING:
                        self.selected[own.id] = True
                    else:
                        self.selected[own.id] = False
        else:
            self.selected[self.data_owners[0].id] = False
            self.selected[self.data_owners[1].id] = True
            self.selected[self.data_owners[2].id] = True
            self.selected[self.data_owners[3].id] = False
        return
    
    def set_lr(self, optimizer):
        self.optimizers = optimizer