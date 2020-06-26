import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from DataUtils.Common import *
from Engines.initialize import *
from Engines.modelHelp import prepare_pack_padded_sequence

torch.manual_seed(seed_num)
random.seed(seed_num)


class BiLSTM(nn.Module):
    """
        BiLSTM
    """

    def __init__(self, **kwargs):
        super(BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        total_dim = 0

        word_paddingId = self.word_paddingId
        word_embed_num = self.embed_num
        word_embed_dim = self.embed_dim
        total_dim += word_embed_dim
        self.word_embedding_dim = word_embed_dim

        self.embed = nn.Embedding(word_embed_num, word_embed_dim, padding_idx=word_paddingId)

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)

        C = self.label_num

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout_emb)


        self.bilstm = nn.LSTM(input_size=total_dim, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)
        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, word, sentence_length):

        word, sentence_length, desorted_indices = prepare_pack_padded_sequence(word, sentence_length,
                                                                               device=self.device)
        x_word = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x_word)

        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[desorted_indices]
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)

        return logit
