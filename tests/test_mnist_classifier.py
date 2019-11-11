#! /usr/bin/env python3

from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import MnistDataset
from archetypes.model import GenericModel
from archetypes.classifier import MnistClassifier


import torch
from torch.nn import Linear, Sigmoid, LeakyReLU

sigmoid = Sigmoid()
leaky_relu = LeakyReLU()
mse = torch.nn.MSELoss(reduction='sum')

bs = 512
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

model = MnistClassifier(
    {"classifier": model_dict}, model_name="Classifier"
)

model.build({'mse': mse}, try_cuda=False)

model.fit(training, nepochs=10)
