# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import time
import math
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Initialize weights for Conv1d, Linear, and ConvTranspose1d layers using Xavier normal initialization
def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)  # Initialize weights using Xavier normal initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to zero

# Generate positional encoding for a batch of sequences
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]  # Ensure batch size matches
    # Calculate positional encoding using sine and cosine functions
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])  # Apply sine to even indices
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])  # Apply cosine to odd indices
    return torch.from_numpy(positions_enc).float()  # Convert to PyTorch tensor

# Generate padding masks for sequences with different lengths
def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()  # Convert to list
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)  # Initialize mask
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0  # Set mask values to 0 for valid positions
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()  # Return boolean mask and its complement

# Apply top-k filtering to logits
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)  # Get top-k values and indices
    out = logits.clone()  # Clone logits tensor
    out[out < v[:, [-1]]] = -float('Inf')  # Set values below top-k to negative infinity
    return out

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        # Initialize positional encoding tensor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer('pe', pe)  # Register as buffer

    def forward(self, pos):
        return self.pe[pos]  # Return positional encoding for given positions

# Movement convolutional encoder
class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),  # Convolutional layer
            nn.Dropout(0.2, inplace=True),  # Dropout layer
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),  # Convolutional layer
            nn.Dropout(0.2, inplace=True),  # Dropout layer
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU activation
        )
        self.out_net = nn.Linear(output_size, output_size)  # Output linear layer
        self.main.apply(init_weight)  # Initialize weights
        self.out_net.apply(init_weight)  # Initialize weights

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)  # Permute dimensions
        outputs = self.main(inputs).permute(0, 2, 1)  # Apply convolutional layers and permute back
        return self.out_net(outputs)  # Apply output linear layer

# Movement convolutional decoder
class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),  # Transposed convolutional layer
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),  # Transposed convolutional layer
            nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU activation
        )
        self.out_net = nn.Linear(output_size, output_size)  # Output linear layer
        self.main.apply(init_weight)  # Initialize weights
        self.out_net.apply(init_weight)  # Initialize weights

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)  # Permute dimensions
        outputs = self.main(inputs).permute(0, 2, 1)  # Apply transposed convolutional layers and permute back
        return self.out_net(outputs)  # Apply output linear layer

# Text encoder using bidirectional GRU
class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)  # Positional embedding layer
        self.input_emb = nn.Linear(word_size, hidden_size)  # Input embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)  # Bidirectional GRU layer
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Linear layer
            nn.LayerNorm(hidden_size),  # Layer normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation
            nn.Linear(hidden_size, output_size)  # Output linear layer
        )
        self.input_emb.apply(init_weight)  # Initialize weights
        self.pos_emb.apply(init_weight)  # Initialize weights
        self.output_net.apply(init_weight)  # Initialize weights
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))  # Initialize hidden state

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)  # Apply positional embedding
        inputs = word_embs + pos_embs  # Add positional embeddings to word embeddings
        input_embs = self.input_emb(inputs)  # Apply input embedding
        hidden = self.hidden.repeat(1, num_samples, 1)  # Repeat hidden state for batch size

        cap_lens = cap_lens.data.tolist()  # Convert to list
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)  # Pack padded sequences

        gru_seq, gru_last = self.gru(emb, hidden)  # Apply GRU layer

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)  # Concatenate bidirectional GRU outputs

        return self.output_net(gru_last)  # Apply output network

# Motion encoder using bidirectional GRU
class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)  # Input embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)  # Bidirectional GRU layer
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),  # Linear layer
            nn.LayerNorm(hidden_size),  # Layer normalization
            nn.LeakyReLU(0.2, inplace=True),  # Leaky ReLU activation
            nn.Linear(hidden_size, output_size)  # Output linear layer
        )
        self.input_emb.apply(init_weight)  # Initialize weights
        self.output_net.apply(init_weight)  # Initialize weights
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))  # Initialize hidden state

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)  # Apply input embedding
        hidden = self.hidden.repeat(1, num_samples, 1)  # Repeat hidden state for batch size

        cap_lens = m_lens.data.tolist()  # Convert to list
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)  # Pack padded sequences

        gru_seq, gru_last = self.gru(emb, hidden)  # Apply GRU layer

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)  # Concatenate bidirectional GRU outputs

        return self.output_net(gru_last)  # Apply output network