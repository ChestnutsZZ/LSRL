import torch
from torch import Tensor
from torch.nn import Embedding, GRU, Linear
from typing import Dict, Any, List

from torchrec.feature_column import CategoricalColumnWithIdentity
from torchrec.model.IValueRLModel import IValueRLModel, IQNet
from torchrec.utils.argument import ArgumentDescription


class DQNQNet(IQNet):
    def __init__(self,
                 weight_file: str,
                 iid_column: CategoricalColumnWithIdentity,
                 state_len_column: CategoricalColumnWithIdentity,
                 state_column: CategoricalColumnWithIdentity,
                 next_state_len_column: CategoricalColumnWithIdentity,
                 next_state_column: CategoricalColumnWithIdentity,
                 rl_sample_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 hidden_size: int,
                 ):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.iid_column = iid_column
        self.state_len_column = state_len_column
        self.state_column = state_column
        self.next_state_len_column = next_state_len_column
        self.next_state_column = next_state_column
        self.rl_sample_column = rl_sample_column
        self.weight_file = weight_file

        self.i_embedding = Embedding(self.iid_column.category_num, self.emb_size)
        self.rnn = GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = Linear(self.hidden_size, self.emb_size)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # B or B * N
        state_ids: Tensor = self.state_column.get_feature_data(data)  # B * S
        state_len: Tensor = self.state_len_column.get_feature_data(data)  # B

        i_vectors: Tensor = self.i_embedding(i_ids)  # B * E or B * N * E
        state_vectors: Tensor = self.i_embedding(state_ids)  # B * S * E

        sort_state_lengths, sort_idx = torch.topk(state_len, k=len(state_len))  # noqa
        sort_state_vectors = state_vectors.index_select(dim=0, index=sort_idx)
        state_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_state_vectors, sort_state_lengths, batch_first=True)

        output, hidden = self.rnn(state_packed, None)  # noqa

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])  # B * E
        unsort_idx = torch.topk(sort_idx, k=len(state_len), largest=False)[1]  # noqa
        rnn_vector: Tensor = sort_rnn_vector.index_select(dim=0, index=unsort_idx)  # B * E

        if len(i_ids.shape) == 1:
            prediction = (rnn_vector * i_vectors).sum(-1)  # B
        else:
            prediction = (rnn_vector.unsqueeze(1) * i_vectors).sum(-1)  # B * N

        return prediction

    def next_forward(self, data: Dict[str, Tensor]) -> Tensor:
        next_i_ids: Tensor = self.rl_sample_column.get_feature_data(data)  # B * N
        next_state_ids: Tensor = self.next_state_column.get_feature_data(data)  # B * S
        next_state_len: Tensor = self.next_state_len_column.get_feature_data(data)  # B

        next_i_vectors: Tensor = self.i_embedding(next_i_ids)  # B * N * E
        next_state_vectors: Tensor = self.i_embedding(next_state_ids)  # B * S * E

        sort_next_state_lengths, sort_idx = torch.topk(next_state_len, k=len(next_state_len))  # noqa
        sort_next_state_vectors = next_state_vectors.index_select(dim=0, index=sort_idx)
        next_state_packed = torch.nn.utils.rnn.pack_padded_sequence(
            sort_next_state_vectors, sort_next_state_lengths, batch_first=True)

        output, hidden = self.rnn(next_state_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])  # B * E
        unsort_idx = torch.topk(sort_idx, k=len(next_state_len), largest=False)[1]  # noqa
        rnn_vector: Tensor = sort_rnn_vector.index_select(dim=0, index=unsort_idx)  # B * E

        prediction = (rnn_vector.unsqueeze(1) * next_i_vectors).sum(-1)  # B * N

        return prediction

    def load_pretrain_embedding(self) -> None:
        # weight = torch.load(self.weight_file)["i_embeddings.weight"]
        # self.i_embedding.from_pretrained(embeddings=weight, freeze=True)
        pass


class DQN(IValueRLModel):
    def forward(self, data: Dict[str, Tensor]):
        main_q: Tensor = self.eval_net(data)
        with torch.no_grad():
            reward: Tensor = self.reward_column.get_feature_data(data).float()
            target_q: Tensor = reward + self.gamma * torch.max(self.target_net.next_forward(data), dim=-1)[0]
        return main_q, target_q

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass
