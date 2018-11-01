import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGram(nn.Module):

    def __init__(self, emb_size: int, emb_dimension: int):
        super(SkipGram, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        init_range = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -init_range, init_range)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v) -> torch.LongTensor:
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word: Dict, file_name: str) -> None:
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        logging.info("Saving {} word vectors of dimension {}".format(len(id2word),
                                                                     self.emb_dimension))
        with open(file_name, 'w') as f:
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write("{} {}\n".format(w, e))
