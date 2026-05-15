import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import TransformerLanguageModel
from dataset import LanguageModelDataset
import kagglehub
import os

# 1. Download & Prep Data
path = kagglehub.dataset_download("thedevastator/tinystories-narrative-classification")
csv_file = os.path.join(path, "tinystories.csv")
df = pd.read_csv(csv_file)

# Take a subset for manageable training
text = " ".join(df['text'].astype(str).tolist()[:15000]) 

# 2. Tokenizer (Character-level for this demo)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Prepare Tensors & Dataset Objects
all_token_ids = encode(text)
n = int(0.9 * len(all_token_ids))
train_ids = all_token_ids[:n]
val_ids = all_token_ids[n:]

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64  
block_size = 128 
learning_rate = 3e-4
epochs = 1 # With 15k stories, 1 epoch is usually plenty for a demo loop

# Initialize Dataset and DataLoaders
train_ds = LanguageModelDataset(train_ids, block_length=block_size)
val_ds = LanguageModelDataset(val_ids, block_length=block_size)

# DataLoader handles the batching and shuffling for us
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

# 4. Initialize Model
model = TransformerLanguageModel(dropout=0.1, block_size=block_size, vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Training Loop
model.train()
print(f"Starting training on {device}...")

for epoch in range(epochs):
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward Pass
        logits = model(x_batch) # Shape: (B, T, Vocab_Size)
        B, T, C = logits.shape
        
        # Flatten the batch and sequence dimensions for CrossEntropy
        loss = F.cross_entropy(logits.view(B*T, C), y_batch.view(B*T))

        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

# 6. Final Save & Export
torch.save(model.state_dict(), 'tinystories_model.pth')

# Optional: Quick Inference Test
model.eval()
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with a blank token
    print("\n--- Model Sample Output ---")
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))