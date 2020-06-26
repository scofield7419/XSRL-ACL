import torch
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance:
    """
        Instance
    """
    def __init__(self):
        self.words = []
        self.labels = []

        self.words_size = 0

        self.words_index = []
        self.label_index = []


