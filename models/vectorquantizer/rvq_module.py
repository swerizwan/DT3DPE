import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
from models.vectorquantizer.quantization_modules import QuantizeEMAReset, QuantizeEMA

from einops import rearrange, repeat, pack, unpack

# Function to check if a value exists
def exists(val):
    """
    Check if a value is not None.

    :param val: Value to check
    :return: Boolean indicating if the value exists
    """
    return val is not None

# Function to provide a default value if the given value does not exist
def default(val, d):
    """
    Return the given value if it exists, otherwise return the default value.

    :param val: Value to check
    :param d: Default value
    :return: Given value or default value
    """
    return val if exists(val) else d

# Function to round up a number to the nearest multiple
def round_up_multiple(num, mult):
    """
    Round up a number to the nearest multiple.

    :param num: Number to round up
    :param mult: Multiple to round up to
    :return: Rounded up number
    """
    return ceil(num / mult) * mult

# Residual Vector Quantization (ResidualVQ) module
class ResidualVQ(nn.Module):
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        """
        Initialize the ResidualVQ module.

        :param num_quantizers: Number of quantizers
        :param shared_codebook: Whether to share the codebook across quantizers
        :param quantize_dropout_prob: Probability of quantizer dropout
        :param quantize_dropout_cutoff_index: Cutoff index for quantizer dropout
        :param kwargs: Additional keyword arguments for the quantization layers
        """
        super().__init__()

        self.num_quantizers = num_quantizers

        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

    @property
    def codebooks(self):
        """
        Get the codebooks from all quantization layers.

        :return: Stacked codebooks tensor
        """
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks 

    def get_codes_from_indices(self, indices): 
        """
        Get the codes from the given indices.

        :param indices: Indices tensor
        :return: Codes tensor
        """
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b=batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d=codebooks.shape[-1])

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) 
        all_codes = codebooks.gather(2, gather_indices) 

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes

    def get_codebook_entry(self, indices):
        """
        Get the codebook entry for the given indices.

        :param indices: Indices tensor
        :return: Codebook entry tensor
        """
        all_codes = self.get_codes_from_indices(indices)
        latent = torch.sum(all_codes, dim=0) 
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None, force_dropout_index=-1):
        """
        Forward pass of the ResidualVQ module.

        :param x: Input tensor
        :param return_all_codes: Whether to return all codes
        :param sample_codebook_temp: Temperature for sampling the codebook
        :param force_dropout_index: Force dropout at a specific index
        :return: Quantized output, indices, loss, perplexity (and all codes if return_all_codes is True)
        """
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) 
            null_indices_shape = [x.shape[0], x.shape[-1]] 
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]] 
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp) 

            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)

        return ret

    def quantize(self, x, return_latent=False):
        """
        Quantize the input tensor.

        :param x: Input tensor
        :param return_latent: Whether to return the latent representation
        :return: Quantized indices (and latent representation if return_latent is True)
        """
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) 

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)

            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx