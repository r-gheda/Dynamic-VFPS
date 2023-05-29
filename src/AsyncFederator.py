import syft as sy
import torch
from torch import nn
import asyncio
from random import random

hook = sy.TorchHook(torch)


class AsyncFederator:
    '''
    k = number of perticipants to be selected
    '''
    def __init__(self, models, optimizers, clients, k = 0):
        if k <= 0:
            k = len(clients)
        
        self.models = models
        self.optimizers = optimizers
        self.clients = clients
        self.selected_clients = []
        self.received = 0

        self.data = []
        self.remote_tensors = []

    async def epoch(self):
        for x in self.examples:
            self.received = 0
            for c in self.clients:
                self.send_message(c)
            
            while (self.received < len(self.clients)):
                True

            self.forward(x) ## do local computation (TO DO)
            self.backward() ## do backward propagation (TO DO)

    
    async def increase_received(self):
        self.received += 1
        return
        
    async def send_message(self, client):
        client.forward()
        return

    def forward(self, x):
        data = []
        remote_tensors = []

        data.append(models[0](x))

        if data[-1].location == models[1].location:
            remote_tensors.append(data[-1].detach().requires_grad_())
        else:
            remote_tensors.append(
                data[-1].detach().move(models[1].location).requires_grad_()
            )

        i = 1
        while i < (len(models) - 1):
            data.append(models[i](remote_tensors[-1]))

            if data[-1].location == models[i + 1].location:
                remote_tensors.append(data[-1].detach().requires_grad_())
            else:
                remote_tensors.append(
                    data[-1].detach().move(models[i + 1].location).requires_grad_()
                )

            i += 1

        data.append(models[i](remote_tensors[-1]))

        self.data = data
        self.remote_tensors = remote_tensors

        return data[-1]

    def backward(self):
        data = self.data
        remote_tensors = self.remote_tensors

        i = len(models) - 2
        while i > -1:
            if remote_tensors[i].location == data[i].location:
                grads = remote_tensors[i].grad.copy()
            else:
                grads = remote_tensors[i].grad.copy().move(data[i].location)

            data[i].backward(grads)
            i -= 1

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()
