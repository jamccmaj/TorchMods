#! /usr/bin/env python3

from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import MnistDataset
from archetypes.autoencoder import Autoencoder


import torch
from torch.nn import Linear, Sigmoid, LeakyReLU

sigmoid = Sigmoid()
leaky_relu = LeakyReLU()
mse = torch.nn.MSELoss(reduction='sum')

bs = 256
shuf = True
nepochs = 1

data_home = "/home/jamc/Data/MNIST_data"
image_fn = f"{data_home}/train-images-idx3-ubyte.gz"
label_fn = f"{data_home}/train-labels-idx1-ubyte.gz"

dataset = MnistDataset(image_fn, label_fn, shape=(-1,))

training = torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuf
)

encoder = OrderedDict(
    (
        ('Hidden_Layer_1', Linear(dataset.images.shape[-1], 256)),
        ('Activation_Layer_1', leaky_relu),
        ('Hidden_Layer_2', Linear(256, 8)),
        ('Sigmoid_Encoding', leaky_relu)
    )
)

decoder = OrderedDict(
    (
        ('Hidden_Layer_1', Linear(8, 256)),
        ('Activation_Layer_1', leaky_relu),
        ('Hidden_Layer_2', Linear(256, dataset.images.shape[-1])),
        ('Sigmoid_Output', sigmoid)
    )
)

model_dict = {
    "encoder": encoder, "decoder": decoder
}

model = Autoencoder(
    model_dict, model_name="Classifier"
)

model.build({'mse': mse}, try_cuda=False)

model.fit(training, nepochs=10)
