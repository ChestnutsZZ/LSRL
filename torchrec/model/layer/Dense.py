from torch.nn import Module, Linear, Dropout, ReLU


class Dense(Module):
    def __init__(self,
                 input_units: int,
                 output_units: int,
                 activation: str,
                 dropout: float
                 ):
        super().__init__()
        self.linear = Linear(input_units, output_units)
        # todo 其他激活函数支持
        if activation == "relu":
            self.activation = ReLU()
        else:
            self.activation = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
