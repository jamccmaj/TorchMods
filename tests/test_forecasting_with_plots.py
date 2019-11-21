#! /usr/bin/env python3

from collections import OrderedDict

from context import archetypes
from context import utensils

from utensils.datasets import ElectricGridPredict
from utensils.special import LstmAllHidden, LstmCellOnly
from utensils.special import ExpandAndRepeatOutput

from archetypes.uber import ForecastEmbedding, Forecaster

import torch
from torch.nn import Linear, Tanh

import numpy as np

from matplotlib import pyplot as plt

tanh = Tanh()
mse = torch.nn.MSELoss(reduction='sum')

bs = 256
shuf = False
nepochs = 100
embed_ts = 50
ts = 1

data_home = "/home/jamc/.datasets/electricity_grid"
data_fn = f"{data_home}/LD2011_2014_.npy"
latent_fn = f"2019_11_21_LD2011_2014_latent_rep_with_metadata.npy"
latent_fn = f"{data_home}/{latent_fn}"
dataset = ElectricGridPredict(latent_fn, data_fn, timesteps=ts)


num_clients = 370
inter_dim = 128
latent_dim = 16

encoder = OrderedDict(
    (
        ('lstm_enc_1', LstmAllHidden(
            num_clients, inter_dim, batch_first=True
        )),
        ('lstm_enc_2', LstmCellOnly(
            inter_dim, latent_dim, batch_first=True
        )),
    )
)

decoder = OrderedDict(
    (
        ('repeater', ExpandAndRepeatOutput(1, embed_ts)),
        ('lstm_dec_1', LstmAllHidden(
            latent_dim, inter_dim, batch_first=True
        )),
        ('lstm_enc_2', LstmAllHidden(
            inter_dim, num_clients, batch_first=True
        ))
    )
)

embed_dict = {
    "encoder": encoder, "decoder": decoder
}

embedder = ForecastEmbedding(
    embed_dict, model_name="UberLatentEmbed"
)

embedder.build({'mse': mse}, try_cuda=True)

inter_dim_1 = 128
inter_dim_2 = 256
latent_fc_dim = latent_dim + 32
output_dim = 370

forecaster = OrderedDict(
    (
        ('linear_layer_1', Linear(latent_fc_dim, inter_dim_1)),
        ('tanh_layer_1', tanh),
        ('linear_layer_2', Linear(inter_dim_1, inter_dim_2)),
        ('tanh_layer_2', tanh),
        ('linear_layer_3', Linear(inter_dim_2, output_dim)),
        ('tanh_layer_3', tanh),
    )
)

forecast_dict = {
    "forecaster": forecaster
}

forecaster = Forecaster(
    forecast_dict, model_name="UberLatentForecast"
)

forecaster.build({'mse': mse}, try_cuda=True)

embedder_states_fn = 'uber_enc_dec_embedding.ckp'
forecaster_states_fn = 'uber_embedding_forecast.ckp'

embedder_states = torch.load(
    embedder_states_fn, map_location=torch.device('cpu')
)

embedder.load_state_dict(embedder_states['model_state_dict'])
embedder.opt.load_state_dict(
    embedder_states['optimizer_state_dict']
)

embed_func = embedder.pieces.encoder
decode_func = embedder.pieces.decoder

forecaster_states = torch.load(
    forecaster_states_fn, map_location=torch.device('cpu')
)

forecaster.load_state_dict(forecaster_states['model_state_dict'])
forecaster.opt.load_state_dict(
    forecaster_states['optimizer_state_dict']
)


# data preparations
datadir = "/home/jamc/.datasets/electricity_grid/"
data_fn = f"{datadir}/LD2011_2014_.npy"
meta_fn = f"{datadir}/2019_11_12_LD2011_2014_metadata.npy"

data = np.load(data_fn).astype(np.float32)
data /= data.max(axis=0)
data = 2 * data - 1.0
md = np.load(meta_fn)

# start forecasting
start_pt = 45000
# predict 25 hours
num_ts_to_predict = 1000

data_seed = torch.tensor(
    data[start_pt:start_pt+embed_ts, :]
).reshape(1, embed_ts, -1)

md_seed = torch.tensor(md[start_pt]).reshape(1, -1)

seed_embed = embed_func(data_seed).unsqueeze(0)
fc_in = torch.cat((seed_embed, md_seed), 1)
prediction = forecaster(fc_in)['forecaster']

curr_step_edata = data_seed
curr_step_emeta = md_seed
outputs = []

for x in range(num_ts_to_predict):
    curr_step_embed = embed_func(curr_step_edata).unsqueeze(0)
    fc_in = torch.cat(
        (curr_step_embed, curr_step_emeta), 1
    )
    prediction = forecaster(fc_in)['forecaster'].unsqueeze(0)
    outputs.append(prediction)
    curr_step_edata = torch.cat(
        (curr_step_edata[:, 1:, :], prediction), 1
    )
    curr_step_emeta = torch.tensor(
        md[start_pt+x]
    ).reshape(1, -1)

all_predictions = torch.cat(
    outputs, 1
).squeeze().cpu().detach().numpy()

a = start_pt + embed_ts
b = start_pt + embed_ts + num_ts_to_predict
all_originals = data[a:b, :]

num_of_clients_to_plot = 15

fig, ax = plt.subplots(num_of_clients_to_plot, sharex=True)
begin_client = 140

for y in range(num_of_clients_to_plot):
    ax[y].plot(
        all_originals[:, begin_client+y], color='blue'
    )
    ax[y].plot(
        all_predictions[:, begin_client+y], color='red'
    )

fig.show()
