#! /usr/bin/env python3

class LstmAllHidden(Module):
    def __init__(
        self, input_size, hidden_size, *args,
        num_layers=1, bias=True, batch_first=True,
        dropout=0.0, bidirectional=False, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, inputs):
        self.output, (
            self.hidden_state, self.cell_state
        ) = self.lstm(inputs)
        return self.output


class LstmCellStateOnly(Module):
    def __init__(
        self, input_size, hidden_size, *args,
        num_layers=1, bias=True, batch_first=True,
        dropout=0.0, bidirectional=False, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, inputs):
        self.output, (
            self.hidden_state, self.cell_state
        ) = self.lstm(inputs)
        return self.cell_state.squeeze()
