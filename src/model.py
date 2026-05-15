'''
model.py
'''
import tokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from layers import Block

VOCAB_SIZE = len(tokenizer.vocab)
EMBEDDING_DIM = 128
embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #create matrix of shape (max_len, emb_size); each row corresponds to a position and each column to a dimension of the embedding
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0,max_len).reshape(max_len, 1) #shape (max_len, 1)
        pos_embedding = torch.zeros(max_len, emb_size) #shape (max_len, emb_size)
        pos_embedding[:, 0::2] = torch.sin(pos*den) #apply sine to even dimensions
        pos_embedding[:, 1::2] = torch.cos(pos*den) #apply cosine to odd dimensions
        pos_embedding = pos_embedding.unsqueeze(0) #shape (1, max_len, emb_size)
        self.dropout = nn.Dropout(dropout) #to mitigate overfitting
        self.register_buffer('pos_embedding', pos_embedding) #register as buffer so it gets moved to GPU with the model
    def forward(self, token_embedding: torch.Tensor):
        #apply positional encodings to input token embeddings
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    
class TransformerLanguageModel(nn.Module):
    def __init__(self, dropout, block_size, vocab_size=VOCAB_SIZE, emb_size=EMBEDDING_DIM):
        super(TransformerLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout, max_len=block_size)
        self.block_size = block_size

        #create a mask to prevent attention from seeing future tokens
        mask = torch.tril(torch.ones(block_size, block_size))
        causal_mask = torch.zeros(block_size, block_size)
        causal_mask = causal_mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('causal_mask', causal_mask) #registered as buffer, not trainable parameter

        #create multiple instances of multi-head attention and feedforward layers for the transformer blocks
        block = Block(emb_size, 4, 4*emb_size)
        self.blocks = nn.ModuleList([Block(emb_size, 4, 4*emb_size) for _ in range(4)]) #stack 4 blocks for deeper model
        self.lm_head = nn.Linear(emb_size, vocab_size) #final output layer to project back to vocab size for predictions
        self.ln_f = nn.LayerNorm(emb_size) #final layer norm for stability

    def forward(self, idx):
        x = self.token_embedding(idx) #shape (seq_len, batch_size, emb_size) -> converts integers (ids) into high-dimensional vectors
        x = self.positional_encoding(x) #shape (seq_len, batch_size, emb_size) -> adds positional information to the token embeddings
        #sequentially pass through each transformer block, applying the same causal mask to prevent attention to future tokens
        for block in self.blocks:
            x = block(x, self.causal_mask) #shape (seq_len, batch_size, emb_size)
        x = self.ln_f(x) #shape (seq_len, batch_size, emb_size) -> final normalization for stability
        logits = self.lm_head(x) #shape (seq_len, batch_size, vocab_size)
        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx