from typing import List

from torch.nn import Module, Sequential

from torchrec.model.layer.Dense import Dense


class MLP(Module):
    def __init__(self,
                 input_units: int,
                 hidden_units_list: List[int],
                 activation: str,
                 dropout: float,
                 ):
        super().__init__()
        self.mlp = Sequential()
        pre_units = input_units
        for index, hidden_units in enumerate(hidden_units_list):
            self.mlp.add_module(f"dense_{index}", Dense(pre_units, hidden_units, activation, dropout))
            pre_units = hidden_units

    def forward(self, x):
        return self.mlp(x)
