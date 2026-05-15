"""
layers.py
This module defines components of the transformer focused on context,
namely the relationship between tokens in a sequence. The self-attention
mechanism allows the model to weigh the importance of different tokens when
processing a sequence, while the multi-head attention allows the model to
capture different types of relationships between tokens by using multiple
attention heads in parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import mask

class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super(SelfAttentionHead, self).__init__()
        self.d_model = d_model #dimension of the model
        self.headsize = head_size #size of each attention head
        self.query = nn.Linear(d_model, head_size, bias=False) #linear layer to project input to query space
        self.key = nn.Linear(d_model, head_size, bias=False) #linear layer to project input to key space
        self.value = nn.Linear(d_model, head_size, bias=False) #linear layer to project input to value space
    def forward(self, x, mask):
        #inputs shape: (batch_size, seq_len, d_model)
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #scaled dot product to check similairty between q and k; scale by sqrt of head size to prevent large values and small gradients
        attention_weights = torch.matmul(q, k.transpose(-2,-1))/(math.sqrt(self.d_model))
        #apply causal mask
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask[:T, :T] == float("-inf"), float("-inf"))
        #softmax for probabilities and matmul with v to get weighted sum of values
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.heads = nn.ModuleList([SelfAttentionHead(d_model, self.head_size) for _ in range(num_heads)])
        #linear layer to project concatenated head outputs from d_model back to d_model
        self.output_projection = nn.Linear(d_model, d_model)
    def forward(self, x, mask):
        #run heads in parallel and concatenate their outputs along the feature dimension
        head_outputs = [head(x, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(concatenated) #project back to d_model
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        gate = F.silu(self.w1(x)) #gated activation to allow model to learn which features to pass through
        value = self.w2(x) #second linear layer for the feedforward network
        x = self.w3(gate*value) #project back to d_model for residual connection
        return F.dropout(x, p=0.1, training=self.training) #dropout for regularization

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        #layer normalization to stabilize training and improve convergence
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, input, mask):
        #residual connections x+f(x), to help with gradient flow and prevent vanishing gradients; output of each sublayer is added to its input
        x = input + self.attention(self.ln1(input), mask)
        output = x + self.feed_forward(self.ln2(x))
        return output