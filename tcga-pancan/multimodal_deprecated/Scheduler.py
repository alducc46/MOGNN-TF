from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch
import pandas as pd
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, CosineAnnealingLR, SequentialLR, LRScheduler

class OmicScheduler(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_name: str, total_epochs: int):
        self.scheduler_name = scheduler_name
        self.total_epochs = total_epochs
        self.optimizer = optimizer
        self.scheduler = self._get_scheduler(scheduler_name, optimizer, total_epochs)

    def _get_scheduler(self, scheduler_name: str, optimizer: torch.optim.Optimizer, total_epochs: int) -> LRScheduler:
        if scheduler_name == 'constant':
            return ConstantLR(optimizer, factor=1.0, total_iters=total_epochs)
        elif scheduler_name == 'step':
            step_size = 12
            gamma = 0.1
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'exponential':
            gamma = 0.9
            return ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = total_epochs
            return CosineAnnealingLR(optimizer, T_max=T_max)
# Inserire più scheduler se necessario
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' is not implemented.")
            
    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    def state_dict(self):
        return self.scheduler.state_dict()
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)