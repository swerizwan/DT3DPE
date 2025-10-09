# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack

# Function to compute the logarithm with a small epsilon to avoid log(0)
def log(t, eps=1e-20):
    """
    Compute the logarithm of a tensor with a small epsilon to avoid log(0).

    :param t: Tensor to compute the logarithm of
    :param eps: Small epsilon value
    :return: Logarithm of the tensor
    """
    return torch.log(t.clamp(min=eps))

# Function to generate Gumbel noise
def gumbel_noise(t):
    """
    Generate Gumbel noise for a tensor.

    :param t: Tensor to generate noise for
    :return: Gumbel noise tensor
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# Function to perform Gumbel sampling
def gumbel_sample(
    logits,
    temperature=1.,
    stochastic=False,
    dim=-1,
    training=True
):
    """
    Perform Gumbel sampling from logits.

    :param logits: Logits tensor
    :param temperature: Temperature for Gumbel-Softmax
    :param stochastic: Whether to add noise
    :param dim: Dimension to sample along
    :param training: Whether the model is in training mode
    :return: Sampled indices
    """
    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    return ind

# QuantizeEMAReset class for vector quantization with EMA and reset mechanism
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        """
        Initialize the QuantizeEMAReset module.

        :param nb_code: Number of codebook entries
        :param code_dim: Dimension of each codebook entry
        :param args: Arguments containing mu (EMA decay rate)
        """
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu  # EMA decay rate
        self.reset_codebook()

    def reset_codebook(self):
        """
        Reset the codebook and related statistics.
        """
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _tile(self, x):
        """
        Tile the input tensor to match the codebook size.

        :param x: Input tensor
        :return: Tiled tensor
        """
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        """
        Initialize the codebook with the input tensor.

        :param x: Input tensor
        """
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        """
        Quantize the input tensor.

        :param x: Input tensor
        :param sample_codebook_temp: Temperature for sampling
        :return: Quantized indices
        """
        k_w = self.codebook.t()

        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  

        code_idx = gumbel_sample(-distance, dim=-1, temperature=sample_codebook_temp, stochastic=True, training=self.training)

        return code_idx

    def dequantize(self, code_idx):
        """
        Dequantize the indices to get the codebook entries.

        :param code_idx: Quantized indices
        :return: Dequantized tensor
        """
        x = F.embedding(code_idx, self.codebook)
        return x

    def get_codebook_entry(self, indices):
        """
        Get the codebook entries for the given indices.

        :param indices: Indices tensor
        :return: Codebook entries
        """
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        """
        Compute the perplexity of the codebook usage.

        :param code_idx: Quantized indices
        :return: Perplexity
        """
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """
        Update the codebook using EMA.

        :param x: Input tensor
        :param code_idx: Quantized indices
        :return: Perplexity
        """
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) 
        code_count = code_onehot.sum(dim=-1) 

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        """
        Preprocess the input tensor.

        :param x: Input tensor
        :return: Preprocessed tensor
        """
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        """
        Forward pass of the quantization module.

        :param x: Input tensor
        :param return_idx: Whether to return the quantized indices
        :param temperature: Temperature for sampling
        :return: Quantized tensor, indices (if return_idx), commitment loss, perplexity
        """
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach()) 
        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity

# QuantizeEMA class for vector quantization with EMA
class QuantizeEMA(QuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        """
        Update the codebook using EMA.

        :param x: Input tensor
        :param code_idx: Quantized indices
        :return: Perplexity
        """
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) 
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) 
        code_count = code_onehot.sum(dim=-1) 

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity