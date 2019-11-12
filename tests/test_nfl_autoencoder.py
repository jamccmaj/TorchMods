#! /usr/bin/env python3

from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import NflRushingDataset
from archetypes.autoencoder import Autoencoder


import torch
from torch.nn import Linear, Sigmoid, LeakyReLU

sigmoid = Sigmoid()
leaky_relu = LeakyReLU()
mse = torch.nn.MSELoss(reduction='sum')
# bce = torch.nn.BCELoss(reduction='sum')

bs = 512
shuf = True
nepochs = 100

data_home = "/home/jamc/Data/MNIST_data"
data_home = "/mnt/Data/dev/NflBigData/data/"

image_fn = f"{data_home}/influence_images_dir_standardized.npy"
label_fn = f"{data_home}/influence_images_yards_per_image.npy"

dataset = NflRushingDataset(image_fn, label_fn, shape=(-1,))

training = torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuf
)

encoder = OrderedDict(
    (
        ('Hidden_Layer_1', Linear(dataset.images.shape[-1], 4)),
        ('Activation_Layer_1', leaky_relu),
        ('Hidden_Layer_2', Linear(4, 2)),
        ('Sigmoid_Encoding', leaky_relu)
    )
)

decoder = OrderedDict(
    (
        ('Hidden_Layer_1', Linear(2, 4)),
        ('Activation_Layer_1', leaky_relu),
        ('Hidden_Layer_2', Linear(4, dataset.images.shape[-1])),
        ('Sigmoid_Output', sigmoid)
    )
)

model_dict = {
    "encoder": encoder, "decoder": decoder
}

model = Autoencoder(
    model_dict, model_name="Classifier"
)

# model.build({'bce': bce}, try_cuda=False, learning_rate=1e-5)
model.build({'mse': mse}, try_cuda=False, learning_rate=1e-3)

model.fit(training, nepochs=nepochs)
