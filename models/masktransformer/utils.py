# Import necessary libraries
import torch
import torch.nn.functional as F
import math
from einops import rearrange

# Function to create a mask based on sequence lengths
def lengths_to_mask(lengths, max_len):
    """
    Generate a mask tensor where each row corresponds to a sequence length.
    Entries are True for positions within the sequence length and False otherwise.

    :param lengths: Tensor of sequence lengths (batch_size,)
    :param max_len: Maximum sequence length
    :return: Mask tensor (batch_size, max_len)
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask 

# Function to create a padding mask for a sequence
def get_pad_mask_idx(seq, pad_idx):
    """
    Generate a mask tensor where entries are True for non-padding elements.

    :param seq: Input sequence tensor (batch_size, seq_len)
    :param pad_idx: Padding index
    :return: Mask tensor (batch_size, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(1)

# Function to create a subsequent mask for transformer attention
def get_subsequent_mask(seq):
    """
    Generate a mask tensor to prevent attention to subsequent positions.

    :param seq: Input sequence tensor (batch_size, seq_len)
    :return: Subsequent mask tensor (batch_size, seq_len, seq_len)
    """
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)

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

# Decorator to switch model to evaluation mode and back
def eval_decorator(fn):
    """
    Decorator to ensure a model is in evaluation mode during the function call.

    :param fn: Function to decorate
    :return: Decorated function
    """
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# Function to normalize a tensor using L2 norm
def l2norm(t):
    """
    Normalize a tensor using L2 norm.

    :param t: Tensor to normalize
    :return: Normalized tensor
    """
    return F.normalize(t, dim=-1)

# Function to create a mask subset based on probability
def get_mask_subset_prob(mask, prob):
    """
    Generate a subset mask based on the given probability.

    :param mask: Original mask tensor
    :param prob: Probability for each element to be included in the subset
    :return: Subset mask tensor
    """
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

# Function to create a mask for special tokens
def get_mask_special_tokens(ids, special_ids):
    """
    Generate a mask tensor for special tokens.

    :param ids: Input sequence tensor (batch_size, seq_len)
    :param special_ids: List of special token IDs
    :return: Mask tensor (batch_size, seq_len)
    """
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids == special_id)
    return mask

# Function to get an activation function by name
def _get_activation_fn(activation):
    """
    Get an activation function by name.

    :param activation: Name of the activation function ('relu' or 'gelu')
    :return: Activation function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# Function to generate a uniform random tensor
def uniform(shape, device=None):
    """
    Generate a uniform random tensor.

    :param shape: Shape of the tensor
    :param device: Device to place the tensor on
    :return: Uniform random tensor
    """
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

# Function to generate a mask tensor with a given probability
def prob_mask_like(shape, prob, device=None):
    """
    Generate a mask tensor with a given probability.

    :param shape: Shape of the tensor
    :param prob: Probability for each element to be True
    :param device: Device to place the tensor on
    :return: Mask tensor
    """
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

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

# Function to sample from a tensor using Gumbel-Softmax
def gumbel_sample(t, temperature=1., dim=1):
    """
    Sample from a tensor using Gumbel-Softmax.

    :param t: Tensor to sample from
    :param temperature: Temperature for the Gumbel-Softmax
    :param dim: Dimension to sample along
    :return: Sampled indices
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

# Function to apply top-k filtering to a tensor
def top_k(logits, thres=0.9, dim=1):
    """
    Apply top-k filtering to a tensor.

    :param logits: Tensor to filter
    :param thres: Threshold for top-k filtering
    :param dim: Dimension to filter along
    :return: Filtered tensor
    """
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs

# Function to generate a cosine schedule
def cosine_schedule(t):
    """
    Generate a cosine schedule.

    :param t: Input tensor
    :return: Cosine schedule tensor
    """
    return torch.cos(t * math.pi * 0.5)

# Function to scale a cosine schedule
def scale_cosine_schedule(t, scale):
    """
    Scale a cosine schedule.

    :param t: Input tensor
    :param scale: Scale factor
    :return: Scaled cosine schedule tensor
    """
    return torch.clip(scale * torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# Function to generate a quantization schedule
def q_schedule(bs, low, high, device):
    """
    Generate a quantization schedule.

    :param bs: Batch size
    :param low: Lower bound
    :param high: Upper bound
    :param device: Device to place the tensor on
    :return: Quantization schedule tensor
    """
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

# Function to calculate performance metrics (loss, accuracy)
def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    """
    Calculate performance metrics (loss, accuracy).

    :param pred: Predicted tensor
    :param labels: Ground truth labels
    :param ignore_index: Index to ignore in the loss calculation
    :param smoothing: Smoothing factor for label smoothing
    :param tk: Top-k value for accuracy calculation
    :return: Loss, predicted indices, accuracy
    """
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)

    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc

# Function to calculate cross-entropy loss with optional label smoothing
def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    """
    Calculate cross-entropy loss with optional label smoothing.

    :param pred: Predicted tensor
    :param labels: Ground truth labels
    :param ignore_index: Index to ignore in the loss calculation
    :param smoothing: Smoothing factor for label smoothing
    :return: Loss tensor
    """
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss