import os
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datareader import DataReader
from .dataset import Word2VecDataset
from .model import SkipGram


class Word2VecTrainer:
    def __init__(self,
                 input_file: str,
                 output_file: str,
                 emb_dimension: int = 100,
                 batch_size: int = 8,
                 window_size: int = 5,
                 iterations: int = 3,
                 initial_lr: float = 0.001,
                 min_count: int = 1):

        self.data = DataReader(input_file, min_count)
        dataset = Word2VecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=os.cpu_count(),
                                     collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGram(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self) -> None:

        optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for iteration in range(self.iterations):
            logging.info("Iteration: {}".format(iteration + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        logging.info("Loss: {}".format(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
