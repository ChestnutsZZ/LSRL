import torch
from torch import Tensor
from torch.nn import Embedding, Linear
from typing import Dict

from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.model.IValueRLModel import IQNet
from torchrec.model.LSRL import LSRL
from torchrec.model.layer.Dense import Dense
from torchrec.model.layer.MLP import MLP


# V0版本：(p-GRU + n-GRU + MLP) + MLP + Linear


class LRLQNet(IQNet):
    def __init__(self,
                 weight_file: str,
                 uid_column: CategoricalColumnWithIdentity,
                 iid_column: CategoricalColumnWithIdentity,
                 pos_state_len_column: CategoricalColumnWithIdentity,
                 pos_state_column: CategoricalColumnWithIdentity,
                 pos_next_state_len_column: CategoricalColumnWithIdentity,
                 pos_next_state_column: CategoricalColumnWithIdentity,
                 neg_state_len_column: CategoricalColumnWithIdentity,
                 neg_state_column: CategoricalColumnWithIdentity,
                 neg_next_state_len_column: CategoricalColumnWithIdentity,
                 neg_next_state_column: CategoricalColumnWithIdentity,
                 rl_sample_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 hidden_size: int,
                 ):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.uid_column = uid_column
        self.iid_column = iid_column
        self.pos_state_len_column = pos_state_len_column
        self.pos_state_column = pos_state_column
        self.pos_next_state_len_column = pos_next_state_len_column
        self.pos_next_state_column = pos_next_state_column
        self.neg_state_len_column = neg_state_len_column
        self.neg_state_column = neg_state_column
        self.neg_next_state_len_column = neg_next_state_len_column
        self.neg_next_state_column = neg_next_state_column
        self.rl_sample_column = rl_sample_column
        self.weight_file = weight_file

        self.i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)

        self.u_embeddings = Embedding(self.uid_column.category_num, self.emb_size)
        self.long_mlp = MLP(self.emb_size * 2, [self.emb_size] * 3, activation="relu", dropout=0.2)

        self.mlp = Dense(self.emb_size * 1, self.emb_size, activation="relu", dropout=0.2)
        self.prediction = Linear(self.emb_size, 1, bias=False)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # B * N
        if len(i_ids.shape) == 1:
            i_ids = i_ids.unsqueeze(-1)
        i_vectors: Tensor = self.i_embeddings(i_ids)  # B * N * E

        sample_n = i_ids.shape[1]

        # long
        u_ids: Tensor = self.uid_column.get_feature_data(data)  # B
        u_vectors: Tensor = self.u_embeddings(u_ids)  # B * E
        u_vectors: Tensor = u_vectors.unsqueeze(1).repeat(1, sample_n, 1)  # B * N * E
        long_mlp_input = torch.cat([u_vectors, i_vectors], dim=-1)  # B * N * 2E
        long_vectors = self.long_mlp(long_mlp_input)  # B * N * E

        prediction = self.prediction(self.mlp(long_vectors)).squeeze(-1)

        if prediction.shape[1] == 1:
            prediction = prediction.squeeze(-1)

        return prediction

    def next_forward(self, data: Dict[str, Tensor]) -> Tensor:
        i_ids: Tensor = self.rl_sample_column.get_feature_data(data)  # B * N
        i_vectors: Tensor = self.i_embeddings(i_ids)  # B * N * E

        sample_n = i_ids.shape[1]

        # long
        u_ids: Tensor = self.uid_column.get_feature_data(data)  # B
        u_vectors: Tensor = self.u_embeddings(u_ids)  # B * E
        u_vectors: Tensor = u_vectors.unsqueeze(1).repeat(1, sample_n, 1)  # B * N * E
        long_mlp_input = torch.cat([u_vectors, i_vectors], dim=-1)  # B * N * 2E
        long_vectors = self.long_mlp(long_mlp_input)  # B * N * E

        prediction = self.prediction(self.mlp(long_vectors)).squeeze(-1)

        return prediction

    def load_pretrain_embedding(self) -> None:
        # weight = torch.load(self.weight_file)
        # self.i_embeddings.from_pretrained(embeddings=weight["i_embeddings.weight"], freeze=True)
        # self.u_embeddings.from_pretrained(embeddings=weight["u_embeddings.weight"], freeze=True)
        pass


class LRL(LSRL):
    pass
