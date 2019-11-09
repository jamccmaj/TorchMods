#! /usr/bin/env python3

import socket
from abc import ABC, abstractmethod
from datetime import datetime
from collections import OrderedDict

import torch
from torch.nn import Module, Sequential


def flatten(_list):
    return [
        item for sublist in _list for item in sublist
    ]


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
    def __init__(
        self, model_pieces, *args,
        model_name="Default", **kwargs
    ):
        super().__init__(*args, **kwargs)

        if len(model_pieces) == 1:
            key = list(model_pieces.keys())[0]
            self.pieces = Sequential(model_pieces[key])

        else:
            self.pieces = Sequential()
            for key in model_pieces:
                self.pieces.add_module(
                    key, Sequential(model_pieces[key])
                )

        self.model_name = model_name
        self.device = 'cpu'
        self.use_cuda = False
        self.loss = None
        self.tensorboard_writer = None
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

    # @abstractmethod
    # def fit_batch(self):
    #     raise NotImplementedError

    # @abstractmethod
    # def fit(self):
    #     raise NotImplementedError

    def forward(self, inputs):
        pieces = {}
        current_inputs = inputs

        for key in self.pieces:
            pieces[key] = self.model_pieces(current_inputs)
            current_inputs = pieces[key]

        return pieces

    def log_to_tensorboard(self, params_dict, count):
        """
            params_dict is a set of key, value pairs, where
            the key is the top-level heading and the value is
            a dict containing key, value pairs corresponding
            to names (key) and scalars (value).

            Such an input dict looks like this:

            params_dict = {
                'loss': {
                    'mse': 23.234,
                    'kl': 4.453
                },

                'cov_matrix': {
                    'trace': 8.564,
                    'det': 1.234
                }
            }

            These will be logged under separate subheadings
            in the tensorboard scalar window.

        """

        if not(self.tensorboard_writer):
            return

        for key in params_dict:
            self.writer.add_scalars(
                key, params_dict[key], count
            )
