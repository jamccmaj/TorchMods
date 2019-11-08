#! /usr/bin/env python3

from abc import ABC, abstractmethod
from torch.nn import Module


class DefaultModel(ABC, Module):
    def __init__(self, *args, model_name="Default"):
        self.model_name = model_name

        self.device = 'cpu'
        self.loss = None
        self.built = False

    def build(self, output_dir='./'):
        pass

    @abstractmethod
    def fit_batch(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError
