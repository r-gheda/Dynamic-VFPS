"""
Vertically partitioned SplitNN implementation

Clients own vertically partitioned data
Server owns the labels
"""

DEFAULT_METHOD = "zeros"
MEAN_DELAY = 0
STD_DELAY = 0
PROBABILITY_OF_TESTING = 0.5


import syft as sy
import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import random
import time

from src.utils.fagin_utils import split_samples_by_class, get_kth_dist, digamma, get_sorted_distances

hook = sy.TorchHook(torch)

class SplitNN:
    def __init__(self, models, server, data_owners, optimizers, distributed_data, k=1, n_selected=1, padding_method=DEFAULT_METHOD):
        self.models = models
        self.server = server
        self.data_owners = data_owners
        self.optimizers = optimizers
        self.selected = {}
        self.selected[server.id] = True
        self.dist_data = distributed_data
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

        self.Q = 1
        self.N = len(self.dist_data)
        self.Nq = {}
        self.mq = {}
        
    def generate_data(self, owner, remote_outputs):
        res = None
        if self.PADDING_METHOD == "latest":
            if not owner.id in self.latest:
                self.PADDING_METHOD = DEFAULT_METHOD
            else:
                res = self.latest[owner.id]
        if self.PADDING_METHOD == "mean": 
            if not owner.id in self.means:
                self.PADDING_METHOD = DEFAULT_METHOD
            else:
                res = self.means[owner.id]
        if self.PADDING_METHOD == "zeros":
            res = torch.zeros([1, 32])
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
        if len(delays) > 0:
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

        for id1, data_ptr, _ in distributed_subdata:
            for id2, data_ptr2, _ in distributed_subdata:
                if not (id1, id2) in self.aggregated_distances:
                    self.aggregated_distances[(id1, id2)] = 0
                    self.dist_counters[(id1, id2)] = 0
                self.aggregated_distances[(id1, id2)] = (self.dist_counters[(id1, id2)]*self.aggregated_distances[(id1, id2)] + aggregate_distances[id1, id2] ) / (self.dist_counters[(id1, id2)] + 1)
                self.dist_counters[(id1, id2)] += 1

        return mi / self.Q
    
    def group_testing(self, n_tests=10):
        self.sorted_scores = []
        self.aggregated_distances = {}
        self.dist_counters = {}
        self.N = len(self.dist_data.distributed_subdata)
        self.get_scores(n_tests)
        
        scores = {}
        
        for ow in self.scores:
            scores[ow] = self.scores[ow]
            self.scores[ow] = self.scores[ow]

        scores_length = len(scores)
        for idx in range(scores_length):
            max_owner = max(scores, key=scores.get)
            if idx < self.n_selected:
                self.selected[max_owner] = True
            else:
                self.selected[max_owner] = False
            self.sorted_scores.append(scores[max_owner])
            scores.pop(max_owner)

        self.sorted_distances = {}
        for id1, _, target in self.dist_data.distributed_subdata:
            self.sorted_distances[id1] = get_sorted_distances(id1, self.class_data[target.item()], self.aggregated_distances)       
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
    
    def is_group_testing_needed(self, removed, added):
        if len(removed) == len(added) == 0:
            return False
        for owner in self.data_owners:
            if not owner.id in self.scores:
                continue
            for id1, data_ptr, _ in removed:
                
                self.scores[owner.id] += 1 / self.Q
                for id2, _, _ in self.dist_data.distributed_subdata:
                    if (id1, id2) in self.aggregated_distances and id1 in self.sorted_distances and id2 in self.sorted_distances and len(self.sorted_distances[id2]) > 1 and self.aggregated_distances[(id1, id2)] < self.sorted_distances[id1][self.k-1]:
                        if self.selected[owner.id]:
                            a = self.sorted_distances[id2][min(self.k,   len(self.sorted_distances[id2])-1)]
                            b = self.sorted_distances[id2][min(self.k-1, len(self.sorted_distances[id2])-2)]
                            c =  self.sorted_distances[id2][-1]*self.Q / self.N
                            self.scores[owner.id] -= (a - b) / c
            for _ in range(len(added)):
                self.scores[owner.id] -= 1 / self.Q
        
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        # return true if in the top k scores, there is a non selected owner
        for owner in self.data_owners:
            if(owner.id in self.scores) and (
                (
                    self.scores[owner.id] > sorted_scores[self.n_selected - 1][1] and not self.selected[owner.id]) or (
                    self.scores[owner.id] <= sorted_scores[self.n_selected - 1][1] and self.selected[owner.id]
                )
            ):
                return True
        
        #print(sorted_scores)
        return False