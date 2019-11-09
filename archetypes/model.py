#! /usr/bin/env python3

import socket
from abc import ABC, abstractmethod
from datetime import datetime

import torch
from torch.nn import Module


def loss_function_producer(loss_functions):
    """
        loss_functions is a python (ordered) dict of
        function names (keys) and functions (values),
        which each take a tuple of two tensors as input,
        e.g. (prediction, target).

        MSE = torch.nn.MSELoss(reduction='mean')
        KL = torch.distributions.kl_divergence
        loss_functions = {
            'mean_squared_error': MSE,
            'kl_divergence': KL
        }

        loss_func = loss_function_producer(loss_functions)

        # call it like this
        loss = loss_func(((recon, target), (post, prior))).sum()
        loss.backward()
        optimizer.step()

    """
    def loss_function(inputs):
        loss_dict = {}
        for f, (i, o) in zip(loss_functions.keys(), inputs):
            loss_dict[f] = loss_functions[f](i, o)
        return loss_dict

    return loss_function


class GenericModel(ABC, Module):
    def __init__(self, *args, model_name="Default", **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.device = 'cpu'
        self.loss = None
        self.built = False

    def build(
        self, losses_dict, optimizer=torch.optim.Adam,
        learning_rate=1e-3, try_cuda=True, cuda_device='0'
    ):

        if try_cuda:
            self.use_cuda = torch.cuda.is_available()

        self.device = torch.device(
            f"cuda:{cuda_device}" if self.use_cuda else "cpu"
        )

        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.benchmark = True

        # losses_dict is kept for tensorboard logging
        self.losses_dict = losses_dict
        self.loss_function = loss_function_producer(losses_dict)

        self.opt = optimizer(self.parameters(), lr=learning_rate)

        self.to(self.device)
        self.built = True

    @abstractmethod
    def fit_batch(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError
