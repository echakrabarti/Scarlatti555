import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 64
EMBEDDING_DIM = 32

class MelodyEncoder(nn.Module):
    """
    LSTM encoder that maps a variable-length interval sequence to a 
    fixed-length, L2-normalized embedding vector.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x, lengths):
        x = x.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        embedding = self.fc(hidden.squeeze(0))
        return F.normalize(embedding, p=2, dim=1)