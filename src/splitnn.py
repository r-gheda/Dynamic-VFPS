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
import random
import time

from src.utils.fagin_utils import split_samples_by_class, get_kth_dist, digamma

hook = sy.TorchHook(torch)
DEFAULT_METHOD = "zeros"
MEAN_DELAY = 1
STD_DELAY = 3

class SplitNN:
    def __init__(self, models, server, data_owners, optimizers, padding_method=DEFAULT_METHOD):
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
        self.counters = {}
        for owner in self.data_owners:
            self.counters[owner.id] = 0

        self.classes = None
        self.k = 1
        
    def generate_data(self, owner, remote_outputs):
        res = None
        if self.PADDING_METHOD == "latest":
            if owner.id in self.latest:
                self.PADDING_METHOD = DEFAULT_METHOD
            res = self.latest[owner.id]
        elif self.PADDING_METHOD == "mean": 
            if owner.id in self.means:
                self.PADDING_METHOD = DEFAULT_METHOD
            res = self.means[owner.id]
        elif self.PADDING_METHOD == "zeros":
            res = torch.zeros([64, 64])
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
                    self.models[owner.id](data_pointer[owner.id].reshape([-1, 14*28])).move(self.server)
                )
                delays.append(max(random.gauss(MEAN_DELAY, STD_DELAY), 0))

                # latest padding update
                self.latest[owner.id] = remote_outputs[-1]

                # mean padding update
                self.counters[owner.id] += 1
                if not owner.id in self.means:
                    self.means[owner.id] = remote_outputs[-1]
                else:
                    self.means[owner.id] = torch.div(torch.add(torch.mul(self.means[owner.id], self.counters[owner.id]-1), remote_outputs[-1]), float(self.counters[owner.id]))
            
            else:
                missing.append(counter)
            counter += 1

        for miss_index in missing:
            remote_outputs.insert(
                miss_index, self.generate_data(self.data_owners[miss_index], remote_outputs).send(self.server)
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

    def knn_mi_estimator(self, distributed_data):
        class_data = split_samples_by_class(distributed_data)
        aggregate_distances = {}
        distances = {}
    
        id1 = 0
        for data_ptr, target in distributed_data:
            id2 = 0
            for data_ptr2, _ in distributed_data:
                remote_partials = []
                delays = []
                for owner in self.data_owners:
                    if not owner.id in data_ptr:
                        continue
                    if (owner, id1, id2) in distances:
                        part_dist = distances[(owner, id1, id2)]
                    else:
                        part_dist = torch.cdist(data_ptr[owner.id], data_ptr2[owner.id])
                    distances[(owner, id1, id2)] = part_dist
                    remote_partials.append(part_dist.move(self.server))
                    delays.append(max(random.gauss(MEAN_DELAY, STD_DELAY), 0))
                # wait for all partials to arrive
                time.sleep(max(delays))
                aggregate_distances[(id1, id2)] = 0
                for rp in remote_partials:
                    aggregate_distances[(id1, id2)] += torch.sum(rp)
                id2 += 1
            id1 += 1

        mi = 0
        id1 = 0
        for data_ptr, target in distributed_data:
            kth_nearest = get_kth_dist(id1, class_data[target], aggregate_distances, self.k)
            m = [0 for i in range(len(distributed_data)) if aggregate_distances[(id1, i)] < kth_nearest]
            mi += digamma(len(distributed_data)) + digamma(len(class_data)) + digamma(self.k) - digamma(len(m))
            id1 += 1
        return mi / len(distributed_data)
    
    def group_testing(self, distributed_data, k, n_tests=100):
        scores = self.get_scores(distributed_data, n_tests)

        for _ in range(k):
            max_owner = max(scores, key=scores.get)
            self.selected[max_owner] = True
            scores.pop(max_owner)
        
        for owner in scores:
            self.selected[owner] = False
        
        return

    def get_scores(self, distributed_data, n_tests=100):
        self.scores = {}
        for _ in range(n_tests):
            # random select from self.data_owners
            test_instance = self.test_gen()
            
            distributed_data_split = []
            for data_ptr, target in distributed_data:
                distributed_data_split.append( (data_ptr.copy(), target) )

            for owner in self.data_owners:
                if not owner in test_instance:
                    for data_ptr, _ in distributed_data_split:
                        data_ptr.pop(owner.id)
            
            mi = self.knn_mi_estimator(distributed_data_split)
            for owner in test_instance:
                if owner not in self.scores:
                    self.scores[owner] = 0
                self.scores[owner] += mi
        
        self.scores = {k: v / n_tests for k, v in self.scores.items()}
        return self.scores
    
    def test_gen(self, p=0.5):
        # random generate a test based on selection probability p
        test_list = []

        while len(test_list) < 1: # empty test is not allowed
            for owner in self.data_owners:
                if np.random.rand() < p:
                    test_list.append(owner)

        return test_list