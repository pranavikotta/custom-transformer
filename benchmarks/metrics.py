import time
import torch
import torch.nn.functional as F
import math
from collections import Counter

def calculate_metrics(model, tokenizer, val_loader, device, num_batches=10):
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_generated_tokens = []

    # Calculate Perplexity and Top-k Accuracy
    top_1_correct = 0
    top_5_correct = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_batches: break
            x, y = x.to(device), y.to(device)

            logits = model(x) # (B, T, vocab_size)
            B, T, C = logits.shape

            # Reshape for cross_entropy
            logits_flat = logits.view(B*T, C)
            targets_flat = y.view(B*T)

            loss = F.cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

            # Calculate Top-k
            probs = F.softmax(logits_flat, dim=-1)
            _, top_5_indices = torch.topk(probs, 5, dim=-1)

            top_1_correct += (top_5_indices[:, 0] == targets_flat).sum().item()
            top_5_correct += (top_5_indices == targets_flat.unsqueeze(1)).sum().item()
            total_tokens += targets_flat.size(0)

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    top_1_acc = (top_1_correct / total_tokens) * 100
    top_5_acc = (top_5_correct / total_tokens) * 100

    # Compression Ratio
    sample_text = "Once upon a time, a small robot found a battery in the woods."
    char_count = len(sample_text)
    token_count = len(tokenizer.encode(sample_text))
    compression_ratio = char_count / token_count

    # Inference Latency (Tokens Per Second)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    start_time = time.time()
    num_gen_tokens = 50
    _ = model.generate(context, max_new_tokens=num_gen_tokens)
    end_time = time.time()
    tps = num_gen_tokens / (end_time - start_time)

    # Vocabulary Diversity (on a generated sample)
    gen_sample = model.generate(context, max_new_tokens=300)[0].tolist()
    unique_tokens = len(set(gen_sample))
    diversity_ratio = unique_tokens / len(gen_sample)

    print("--- TRANSFORMER PERFORMANCE METRICS ---")
    print(f"1. Perplexity:         {perplexity:.2f} (lower is better)")
    print(f"2. Compression Ratio:  {compression_ratio:.2f}x (chars per token)")
    print(f"3. Top-1 Accuracy:     {top_1_acc:.2f}%")
    print(f"   Top-5 Accuracy:     {top_5_acc:.2f}%")
    print(f"4. Inference Speed:    {tps:.2f} tokens/sec")
    print(f"5. Diversity Ratio:    {diversity_ratio:.2f} (unique/total tokens)")
    print("---------------------------------------")
