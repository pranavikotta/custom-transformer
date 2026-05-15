'''
dataset.py
Defines a class which is a PyTorch Dataset for training language models. 
It takes a list of token IDs and a block length, and provides input-target 
pairs for training. The input is a sequence of token IDs, and the target is 
the same sequence shifted by one position, allowing the model to learn to 
predict the next token in a sequence.
'''
import torch

class LanguageModelDataset:
    def __init__(self, token_ids, block_length=128):
        self.token_ids = torch.LongTensor(token_ids)
        self.block_length = block_length
    def __len__(self):
        return max(0, len(self.token_ids) - self.block_length)
    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx+self.block_length+1]
        input = chunk[:-1]
        target = chunk[1:]
        return input, target
    