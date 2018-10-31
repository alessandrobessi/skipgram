from skipgram.train import Word2VecTrainer

if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file='LICENSE', output_file='out.vec')
    w2v.train()
