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
        self.pos = []
        self.prd = []
        self.prd_posi = []
        self.labels = []

        self.elmo_token_index = []
        self.elmo_char_index = []

        self.words_size = 0

        self.words_index = []
        self.pos_index = []
        self.prd_index = []
        self.prd_posi_index = []
        self.label_index = []
