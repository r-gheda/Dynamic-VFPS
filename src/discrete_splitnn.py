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

from src.utils.fagin_utils import split_samples_by_class, get_kth_dist, digamma, get_sorted_distances

hook = sy.TorchHook(torch)
DEFAULT_METHOD = "zeros"
MEAN_DELAY = 0
STD_DELAY = 0
PROBABILITY_OF_TESTING = 0.5

class DiscreteSplitNN:
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
        
    def generate_data(self, owner, remote_outputs):
        res = None
        if self.PADDING_METHOD == "latest":
            if not owner.id in self.latest:
                self.PADDING_METHOD = DEFAULT_METHOD
            res = self.latest[owner.id]
        elif self.PADDING_METHOD == "mean": 
            if not owner.id in self.means:
                self.PADDING_METHOD = DEFAULT_METHOD
            res = self.means[owner.id]
        elif self.PADDING_METHOD == "zeros":
            res = torch.zeros([1, 32])
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
        loss = criterion(pred, target.reshape(-1, 1)[0])
        
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
        loss = criterion(pred, target.reshape(-1, 1)[0])
        
        return loss.detach().get()

    def knn_mi_estimator(self, distributed_subdata):
        self.class_data = self.dist_data.split_samples_by_class(distributed_subdata)
        aggregate_distances = {}
        distances = {}
    
        for id1, data_ptr, target in distributed_subdata:
            for id2, data_ptr2, _ in distributed_subdata:
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
                    self.local_scores[owner.id] = [part_dist]
                    remote_partials.append(part_dist.move(self.server))
                    delays.append(max(random.gauss(MEAN_DELAY, STD_DELAY), 0))
                # wait for all partial distances to arrive
                time.sleep(max(delays))
                aggregate_distances[id1, id2] = 0
                for rp in remote_partials:
                    aggregate_distances[id1, id2] += torch.sum(rp)

        mi = 0
        for id1, data_ptr, target in distributed_subdata:
            self.Nq[target.item()] = len(self.class_data[target.item()])
            sorted_distances = get_sorted_distances(id1, self.class_data[target.item()], aggregate_distances)
           
            self.mq[id1] = len([0 for id2, _, _ in distributed_subdata if (
                len(sorted_distances) > 0
            ) and (
                aggregate_distances[id1, id2] < sorted_distances[min(self.k, len(sorted_distances)-1)]
            )])
            if self.mq[id1] > 0:
                mi += digamma(self.N) - digamma(self.Nq[target.item()]) + digamma(self.k) - digamma(self.mq[id1])

        return mi / self.Q
    
    def group_testing(self, n_tests=100):
        scores = self.get_scores(n_tests)

        for _ in range(self.n_selected):
            max_owner = max(scores, key=scores.get)
            self.selected[max_owner] = True
            scores.pop(max_owner)
        
        for owner in scores:
            self.selected[owner] = False
        
        return

    def get_scores(self, n_tests=100):
        self.scores = {}
        estimate_subdata = self.dist_data.generate_estimate_subdata()
        self.local_scores = {}
        for _ in range(n_tests):
            # random select from self.data_owners
            test_instance = self.test_gen(PROBABILITY_OF_TESTING)
            
            distributed_data_split = []
            for id, data_ptr, target in estimate_subdata:
                distributed_data_split.append( (id, data_ptr.copy(), target) )

            for owner in self.data_owners:
                if not owner in test_instance:
                    for _, data_ptr, _ in distributed_data_split:
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