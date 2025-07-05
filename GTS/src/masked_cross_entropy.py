# coding: utf-8

import torch
from torch.nn import functional


def sequence_mask(sequence_length, max_len=None):
    """
    Creates a binary mask for each sequence in a batch
    
    Args:
        sequence_length: Tensor of shape (batch_size) containing sequence lengths
        max_len: Maximum sequence length (optional)
        
    Returns:
        Binary mask of shape (batch_size, max_len)
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Calculates cross-entropy loss with masking based on sequence length
    
    Args:
        logits: Tensor of shape (batch_size, max_len, num_classes) containing 
                unnormalized prediction scores
        target: Tensor of shape (batch_size, max_len) containing the true class indices
        length: List/tensor of sequence lengths for each batch element
        
    Returns:
        Average loss value masked by the sequence lengths
    """
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    # Flatten logits and targets for loss calculation
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    target_flat = target.view(-1, 1)
    
    # Calculate per-element losses
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # Reshape losses and apply mask
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    
    # Calculate average loss
    loss = losses.sum() / length.float().sum()
    return loss


def masked_cross_entropy_without_logit(logits, target, length):
    """
    Calculates cross-entropy loss when inputs are already probabilities (not logits)
    
    Args:
        logits: Tensor of shape (batch_size, max_len, num_classes) containing 
                probability distributions
        target: Tensor of shape (batch_size, max_len) containing the true class indices
        length: List/tensor of sequence lengths for each batch element
        
    Returns:
        Average loss value masked by the sequence lengths
    """
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)

    # Flatten inputs and targets
    logits_flat = logits.view(-1, logits.size(-1))
    
    # Add small epsilon for numerical stability
    log_probs_flat = torch.log(logits_flat + 1e-12)
    target_flat = target.view(-1, 1)
    
    # Calculate per-element losses
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # Reshape losses and apply mask
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    
    # Calculate average loss
    loss = losses.sum() / length.float().sum()
    return loss