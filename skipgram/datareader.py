import logging
import numpy as np
from typing import List, Union


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_file_name: str, min_count: int):

        self.negatives = list()
        self.discards = list()
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.input_file_name = input_file_name
        self.read_words(min_count)
        self.init_table_negatives()
        self.init_table_discards()

    def read_words(self, min_count: int) -> None:
        word_frequency = dict()
        for line in open(self.input_file_name, encoding='utf8'):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1e6 == 0:
                            logging.info("Read {}M words".format(int(self.token_count / 1e6)))

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        logging.info("Total embeddings: {}".format(len(self.word2id)))

    def init_table_discards(self) -> None:
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def init_table_negatives(self) -> None:
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, size: int) -> Union[List, np.array]:
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response
