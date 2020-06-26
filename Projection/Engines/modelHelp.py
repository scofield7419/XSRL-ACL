import torch
import random
from DataUtils.Common import *

torch.manual_seed(seed_num)
random.seed(seed_num)


def prepare_pack_padded_sequence(inputs_words, seq_lengths, device="cpu", descending=True):

    sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long(), descending=descending)
    indices_ = indices.numpy().tolist()

    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]

    return sorted_inputs_words, sorted_seq_lengths.cpu().numpy(), desorted_indices
