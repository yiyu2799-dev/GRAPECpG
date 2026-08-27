import torch
import torch.nn as nn


class DNAEncoderCNN(nn.Module):
    """Encode a target-centered DNA window into a CpG-site representation."""

    def __init__(
        self,
        dna_window=201,
        vocab_size=5,
        token_dim=4,
        conv1_channels=64,
        conv2_channels=128,
        kernel1=11,
        pool1=4,
        kernel2=3,
        pool2=2,
        hidden_dim=64,
        out_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        if dna_window <= 0 or dna_window % 2 != 1:
            raise ValueError('dna_window must be a positive odd integer.')

        self.dna_window = int(dna_window)
        self.out_dim = int(out_dim)
        self.embed = nn.Embedding(vocab_size, token_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(token_dim, conv1_channels, kernel_size=kernel1),
            nn.ReLU(),
            nn.MaxPool1d(pool1),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=kernel2),
            nn.ReLU(),
            nn.MaxPool1d(pool2),
        )

        conv_out_len = self._conv_output_length(self.dna_window, kernel1, pool1, kernel2, pool2)
        if conv_out_len <= 0:
            raise ValueError(f'dna_window={dna_window} is too short for the CNN settings.')
        self.flatten_dim = conv2_channels * conv_out_len
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _conv_output_length(length, kernel1, pool1, kernel2, pool2):
        length = length - kernel1 + 1
        length = length // pool1
        length = length - kernel2 + 1
        length = length // pool2
        return length

    def forward(self, dna_tokens):
        if dna_tokens.dtype != torch.long:
            dna_tokens = dna_tokens.long()
        x = self.embed(dna_tokens)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
