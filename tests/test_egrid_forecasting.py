#! /usr/bin/env python3

from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import ElectricGridPredict
from utensils.special import LstmAllHidden, LstmCellOnly
from utensils.special import ExpandAndRepeatOutput

from archetypes.uber import Forecaster

import torch
from torch.nn import Linear, Tanh

tanh = Tanh()
mse = torch.nn.MSELoss(reduction='sum')

bs = 256
shuf = True
nepochs = 100
ts = 1

data_home = "/home/jamc/.datasets/electricity_grid"
data_fn = f"{data_home}/LD2011_2014_.npy"
latent_fn = f"2019_11_12_LD2011_2014_latent_rep_with_metadata.npy"
latent_fn = f"{data_home}/{latent_fn}"


dataset = ElectricGridPredict(latent_fn, data_fn, timesteps=ts)

# scaling for tanh input
dataset.targets = 2 * dataset.targets - 1.0

training = torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuf
)

inter_dim_1 = 128
inter_dim_2 = 256
latent_dim = dataset.latent_w_meta.shape[-1]
output_dim = dataset.targets.shape[-1]

forecaster = OrderedDict(
    (
        ('linear_layer_1', Linear(latent_dim, inter_dim_1)),
        ('tanh_layer_1', tanh),
        ('linear_layer_2', Linear(inter_dim_1, inter_dim_2)),
        ('tanh_layer_2', tanh),
        ('linear_layer_3', Linear(inter_dim_2, output_dim)),
        ('tanh_layer_3', tanh),
    )
)

model_dict = {
    "forecaster": forecaster
}

model = Forecaster(
    model_dict, model_name="UberLatentForecast"
)

model.build({'mse': mse}, try_cuda=True)

model.fit(training, nepochs=nepochs)
