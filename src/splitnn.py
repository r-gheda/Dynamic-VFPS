"""
Vertically partitioned SplitNN implementation

Clients own vertically partitioned data
Server owns the labels
"""


import syft as sy
import torch
from torch import nn
import torch.distributed as dist
import numpy as np

from src.utils.fagin_utils import split_samples_by_class, get_kth_dist, digamma

hook = sy.TorchHook(torch)


class SplitNN:
    def __init__(self, models, server, data_owners, optimizers):
        self.models = models
        self.server = server
        self.data_owners = data_owners
        self.optimizers = optimizers
        self.selected = {}
        self.selected[server.id] = True
        for owner in data_owners:
            self.selected[owner.id] = True
        self.selected[data_owners[0].id] = False

        self.rank = None
        self.n_features = None
        self.world_size = None 
        self.k = None
        self.data = None

        self.classes = None

    def predict(self, data_pointer):
            
        #individual client's output upto their respective cut layer
        client_output = {}
        
        #outputs that is moved to server and subjected to concatenate for server input
        remote_outputs = []
        
        #iterate over each client and pass thier inputs to respective model segment and send outputs to server
        for owner in self.data_owners:
            if self.selected[owner.id]:
                remote_outputs.append(
                    self.models[owner.id](data_pointer[owner.id].reshape([-1, 14*28])).move(self.server)
                )
            else:
                remote_outputs.append(torch.zeros([64, 64]).send(self.server))
        
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

    def knn_mi_estimator(self, distributed_data, k):
        class_data = split_samples_by_class(distributed_data)
        aggregate_distances = {}
        distances = {}
    
        id1 = 0
        for data_ptr, target in distributed_data:
            id2 = 0
            for data_ptr2, target2 in distributed_data:
                remote_partials = []
                for owner in self.data_owners:
                    if (owner, id1, id2) in distances:
                        part_dist = distances[(owner, id1, id2)]
                    else:
                        part_dist = torch.cdist(data_ptr[owner.id], data_ptr2[owner.id])
                    distances[(owner, id1, id2)] = part_dist
                    remote_partials.append(part_dist.move(self.server))
                aggregate_distances[(id1, id2)] = 0
                for rp in remote_partials:
                    aggregate_distances[(id1, id2)] += torch.sum(rp)
                id2 += 1
            id1 += 1

        mi = 0
        id1 = 0
        for data_ptr, target in distributed_data:
            kth_nearest = get_kth_dist(id1, class_data[target], aggregate_distances, k)
            m = [0 for i in range(len(distributed_data)) if aggregate_distances[(id1, i)] < kth_nearest]
            mi += digamma(len(distributed_data)) + digamma(len(class_data)) + digamma(k) - digamma(len(m))
            id1 += 1
        return mi / len(distributed_data)