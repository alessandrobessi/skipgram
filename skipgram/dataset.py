from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset


class Word2VecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.input_file_name, encoding="utf8")

    def __len__(self) -> int:
        return self.data.sentences_count

    def __getitem__(self,
                    idx: int) -> List[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]:
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[
                                    self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.get_negatives(5)) for i, u in enumerate(word_ids)
                            for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches: List) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        # noinspection PyUnresolvedReferences
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
