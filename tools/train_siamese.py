"""
train_siamese.py — train a small LSTM encoder to embed melody 
interval sequences such that same-piece excerpts are close in 
vector space and different-piece excerpts are far apart.

Architecture: MelodyEncoder (LSTM -> Linear -> L2-normalized embedding)
Loss:         Contrastive loss with MARGIN=1.8
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

EMBEDDING_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
MARGIN = 1.8  # larger margin forces negatives to be genuinely separated


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        a = torch.tensor(pair["a"], dtype=torch.float32)
        b = torch.tensor(pair["b"], dtype=torch.float32)
        label = torch.tensor(pair["label"], dtype=torch.float32)
        return a, b, label


def collate_pairs(batch):
    """Pad variable-length sequences to the longest in the batch."""
    a_seqs, b_seqs, labels = zip(*batch)

    max_len_a = max(len(s) for s in a_seqs)
    max_len_b = max(len(s) for s in b_seqs)

    a_padded = torch.zeros(len(batch), max_len_a)
    b_padded = torch.zeros(len(batch), max_len_b)
    a_lengths = torch.zeros(len(batch), dtype=torch.long)
    b_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, (a, b) in enumerate(zip(a_seqs, b_seqs)):
        a_padded[i, :len(a)] = a
        b_padded[i, :len(b)] = b
        a_lengths[i] = len(a)
        b_lengths[i] = len(b)

    labels = torch.stack(labels)
    return a_padded, a_lengths, b_padded, b_lengths, labels


class MelodyEncoder(nn.Module):
    """
    Single encoder applied to both inputs in a siamese pair.
    LSTM reads the interval sequence step by step; its final 
    hidden state is projected to a fixed-length L2-normalized 
    embedding vector.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, lengths):
        # x: (batch, seq_len) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        embedding = self.fc(hidden.squeeze(0))
        return F.normalize(embedding, p=2, dim=1)


def contrastive_loss(embed_a, embed_b, labels, margin=MARGIN):
    """
    label=1 (same piece): minimize distance between embeddings
    label=0 (different piece): push distance to at least `margin`
    """
    distances = F.pairwise_distance(embed_a, embed_b)
    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)
    return (positive_loss + negative_loss).mean()


def train():
    with open("data/training_pairs.json") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} training pairs")

    dataset = PairDataset(pairs)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairs
    )

    encoder = MelodyEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches = 0

        for a, a_len, b, b_len, labels in loader:
            optimizer.zero_grad()
            embed_a = encoder(a, a_len)
            embed_b = encoder(b, b_len)
            loss = contrastive_loss(embed_a, embed_b, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), "data/melody_encoder.pt")
    print("\nSaved trained encoder to data/melody_encoder.pt")


if __name__ == "__main__":
    train()