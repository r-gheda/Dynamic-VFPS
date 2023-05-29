import syft as sy
import torch
from torch import nn
import asyncio
from random import random

hook = sy.TorchHook(torch)

class AsyncClient:
    
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        self.selected = True
        self.federator = None

    def set_federator(self, fed):
        self.federator = fed

    def set_selected(self,val):
        self.selected(val)

    async def forward(self, x):
        if(not self.selected):
            self.federator.increase_received() 
            return
        await asyncio.sleep(random.)
        ## do computation (TO DO)

        self.federator.increase_received()

    def backward(self):
        ## do backward prop (TO DO)
        pass
