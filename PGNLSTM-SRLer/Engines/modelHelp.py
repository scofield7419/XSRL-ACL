import torch
import random
from DataUtils.Common import *

torch.manual_seed(seed_num)
random.seed(seed_num)


def prepare_pack_padded_sequence(inputs_words, pos, dep, dep_head, prd, seq_lengths, device="cpu", descending=True):
    """
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(torch.Tensor(seq_lengths).long(), descending=descending)
    indices_ = indices.numpy().tolist()
    sorted_const = []
    for ind in indices_:
        sorted_const.append(dep_head[ind])

    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    sorted_pos = pos[indices]
    sorted_dep = dep[indices]
    sorted_prd = prd[indices]
    return sorted_inputs_words, sorted_pos, sorted_dep, sorted_const, sorted_prd, sorted_seq_lengths.cpu().numpy(), desorted_indices
