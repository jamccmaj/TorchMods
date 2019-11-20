#! /usr/bin/env python

from archetypes.model import GenericModel
from utensils.errors import NotBuiltError


class ForecastEmbedding(GenericModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_batch(self, input_batch):
        self.opt.zero_grad()
        batch_input, batch_target = (
            x.to(self.device) for x in input_batch
        )
        outputs_dict = self.forward(batch_input)
        keys = list(outputs_dict.keys())
        predictions = outputs_dict[keys[-1]]
        loss_params = ((predictions, batch_target),)
        losses = self.loss_function(loss_params)
        self.loss = sum(losses.values())
        self.loss.backward()
        self.opt.step()
        return predictions, losses

    def fit(
        self, inputs, nepochs=1, logint=100,
        continue_counter=0, continue_epoch=0
    ):

        if not(self.built):
            msg = "run build method on model before fitting!"
            raise NotBuiltError(msg)

        counter = continue_counter
        nepochs += continue_epoch

        for i in range(continue_epoch, nepochs):
            self.train()
            for j, batch in enumerate(inputs):
                predictions, losses = self.fit_batch(batch)

                if j % logint == 0:
                    outmsg = f"[Epoch {i}, Batch {j} Loss] "
                    outmsg += f"Total: {round(self.loss.item(), 3)}"
                    for k in losses:
                        outmsg += f" {k}: " + str(
                            round(losses[k].item(), 3)
                        )
                    print(outmsg)
                counter += 1


class Forecaster(GenericModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_batch(self, input_batch):
        self.opt.zero_grad()
        batch_input, batch_target = (
            x.to(self.device) for x in input_batch
        )
        outputs_dict = self.forward(batch_input)
        keys = list(outputs_dict.keys())
        predictions = outputs_dict[keys[-1]]
        loss_params = ((predictions, batch_target),)
        losses = self.loss_function(loss_params)
        self.loss = sum(losses.values())
        self.loss.backward()
        self.opt.step()
        return predictions, losses

    def fit(
        self, inputs, nepochs=1, logint=100,
        continue_counter=0, continue_epoch=0
    ):

        if not(self.built):
            msg = "run build method on model before fitting!"
            raise NotBuiltError(msg)

        counter = continue_counter
        nepochs += continue_epoch

        for i in range(continue_epoch, nepochs):
            self.train()
            for j, batch in enumerate(inputs):
                predictions, losses = self.fit_batch(batch)

                if j % logint == 0:
                    outmsg = f"[Epoch {i}, Batch {j} Loss] "
                    outmsg += f"Total: {round(self.loss.item(), 3)}"
                    for k in losses:
                        outmsg += f" {k}: " + str(
                            round(losses[k].item(), 3)
                        )
                    print(outmsg)
                counter += 1
