import torch
from torch import Tensor
from torch.nn import Embedding, GRU, Linear
from typing import Dict

from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.model.IValueRLModel import IQNet
from torchrec.model.LSRL import LSRL
from torchrec.model.layer.Dense import Dense
from torchrec.model.layer.MLP import MLP


# 只有正反馈序列


class SRLPQNet(IQNet):
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

        self.pos_rnn = GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.pos_mlp = MLP(self.hidden_size + self.emb_size, [self.emb_size] * 3, activation="relu", dropout=0.2)

        self.mlp = Dense(self.emb_size * 1, self.emb_size, activation="relu", dropout=0.2)
        self.prediction = Linear(self.emb_size, 1, bias=False)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # B * N
        if len(i_ids.shape) == 1:
            i_ids = i_ids.unsqueeze(-1)
        i_vectors: Tensor = self.i_embeddings(i_ids)  # B * N * E

        sample_n = i_ids.shape[1]

        pos_state_ids: Tensor = self.pos_state_column.get_feature_data(data)  # B * S
        pos_state_len: Tensor = self.pos_state_len_column.get_feature_data(data)  # B
        pos_state_vectors: Tensor = self.i_embeddings(pos_state_ids)  # B * S * E

        sort_pos_state_lengths, sort_pos_idx = torch.topk(pos_state_len, k=len(pos_state_len))  # noqa
        sort_pos_state_vectors = pos_state_vectors.index_select(dim=0, index=sort_pos_idx)
        pos_state_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_pos_state_vectors, sort_pos_state_lengths, batch_first=True)

        pos_output, pos_hidden = self.pos_rnn(pos_state_packed, None)

        pos_mlp_input = torch.cat([pos_hidden[-1].unsqueeze(1).repeat(1, sample_n, 1), i_vectors], dim=-1)

        # Unsort
        sort_pos_mlp_vector = self.pos_mlp(pos_mlp_input)  # B * E
        unsort_pos_idx = torch.topk(sort_pos_idx, k=len(pos_state_len), largest=False)[1]  # noqa
        pos_mlp_vector: Tensor = sort_pos_mlp_vector.index_select(dim=0, index=unsort_pos_idx)  # B * E

        prediction = self.prediction(self.mlp(pos_mlp_vector)).squeeze(-1)

        if prediction.shape[1] == 1:
            prediction = prediction.squeeze(-1)

        return prediction

    def next_forward(self, data: Dict[str, Tensor]) -> Tensor:
        i_ids: Tensor = self.rl_sample_column.get_feature_data(data)  # B * N
        i_vectors: Tensor = self.i_embeddings(i_ids)  # B * N * E

        sample_n = i_ids.shape[1]

        pos_state_ids: Tensor = self.pos_next_state_column.get_feature_data(data)  # B * S
        pos_state_len: Tensor = self.pos_next_state_len_column.get_feature_data(data)  # B
        pos_state_vectors: Tensor = self.i_embeddings(pos_state_ids)  # B * S * E

        sort_pos_state_lengths, sort_pos_idx = torch.topk(pos_state_len, k=len(pos_state_len))  # noqa
        sort_pos_state_vectors = pos_state_vectors.index_select(dim=0, index=sort_pos_idx)
        pos_state_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_pos_state_vectors, sort_pos_state_lengths, batch_first=True)

        pos_output, pos_hidden = self.pos_rnn(pos_state_packed, None)

        pos_mlp_input = torch.cat([pos_hidden[-1].unsqueeze(1).repeat(1, sample_n, 1), i_vectors], dim=-1)

        # Unsort
        sort_pos_mlp_vector = self.pos_mlp(pos_mlp_input)  # B * E
        unsort_pos_idx = torch.topk(sort_pos_idx, k=len(pos_state_len), largest=False)[1]  # noqa
        pos_mlp_vector: Tensor = sort_pos_mlp_vector.index_select(dim=0, index=unsort_pos_idx)  # B * E

        prediction = self.prediction(self.mlp(pos_mlp_vector)).squeeze(-1)

        return prediction

    def load_pretrain_embedding(self) -> None:
        # weight = torch.load(self.weight_file)
        # self.i_embeddings.from_pretrained(embeddings=weight["i_embeddings.weight"], freeze=True)
        # self.u_embeddings.from_pretrained(embeddings=weight["u_embeddings.weight"], freeze=True)
        pass


class SRLP(LSRL):
    pass
