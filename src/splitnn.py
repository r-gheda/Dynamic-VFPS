"""
Vertically partitioned SplitNN implementation

Clients own vertically partitioned data
Server owns the labels
"""


import syft as sy
import torch
from torch import nn

hook = sy.TorchHook(torch)


class SplitNN:
    def __init__(self, models, server, data_owners, optimizers):
        self.models = models
        self.server = server
        self.data_owners = data_owners
        self.optimizers = optimizers

    def predict(self, data_pointer):
            
        #individual client's output upto their respective cut layer
        client_output = {}
        
        #outputs that is moved to server and subjected to concatenate for server input
        remote_outputs = []
        
        #iterate over each client and pass thier inputs to respective model segment and send outputs to server
        for owner in self.data_owners:
            client_output[owner.id] = self.models[owner.id](data_pointer[owner.id].reshape([-1, 14*28]))
            remote_outputs.append(
                client_output[owner.id].move(self.server)
            )
        
        #concat outputs from all clients at server's location
        server_input = torch.cat(remote_outputs, 1)
        
        #pass concatenated output from server's model segment
        pred = self.models["server"](server_input)
        
        return pred

    def train(self, data_pointer, target):

        #make grads zero
        for opt in self.optimizers:
            opt.zero_grad()
        
        #predict the output
        pred = self.predict(data_pointer)
        
        #calculate loss
        criterion = nn.NLLLoss()
        loss = criterion(pred, target.reshape(-1, 64)[0])
        
        #backpropagate
        loss.backward()
        
        #optimization step
        for opt in self.optimizers:
            opt.step()
            
        return loss.detach().get()

