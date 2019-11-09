#! /usr/bin/env python3

import sys
from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import MnistDataset
from archetypes.model import GenericModel


import torch
from torch.nn import Linear, Sigmoid, LeakyReLU

sigmoid = Sigmoid()
leaky_relu = LeakyReLU()
mse = torch.nn.MSELoss(reduction='mean')

bs = 64
shuf = True
nepochs = 1

data_home = "/home/jamc/Data/MNIST_data"
image_fn = f"{data_home}/train-images-idx3-ubyte.gz"
label_fn = f"{data_home}/train-labels-idx1-ubyte.gz"

dataset = MnistDataset(image_fn, label_fn, shape=(-1,))

training = torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuf
)

model_dict = OrderedDict(
    (
        ('Hidden_Layer_1', Linear(dataset.images.shape[-1], 128)),
        ('Activation_Layer_1', leaky_relu),
        ('Hidden_Layer_2', Linear(128, 10)),
        ('Sigmoid_Output', sigmoid)

    )
)

for i in range(nepochs):
    for j, (x, y) in enumerate(training):
        # print(j, x, y)
        break

model = GenericModel(
    {"classifier": model_dict}, model_name="Classifier"
)

# print(model.model_pieces)

model.build({'mse': mse}, try_cuda=False)
