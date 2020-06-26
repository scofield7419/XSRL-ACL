import torch
import torch.nn as nn
import random
from Engines.BiLSTM import BiLSTM
from Engines.CRF import CRF
from DataUtils.Common import *

torch.manual_seed(seed_num)
random.seed(seed_num)


class SequenceLabeler(nn.Module):

    def __init__(self, config):
        super(SequenceLabeler, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.word_paddingId = config.word_paddingId

        self.label_num = config.class_num
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pretrain
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight

        # use crf
        self.use_crf = config.use_crf

        # cuda or cpu
        self.device = config.device

        self.target_size = self.label_num if self.use_crf is False else self.label_num + 2

        self.encoder_model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim,
                                    label_num=self.target_size,
                                    word_paddingId=self.word_paddingId,
                                    dropout_emb=self.dropout_emb, dropout=self.dropout,
                                    lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                    pretrained_embed=self.pretrained_embed,
                                    pretrained_weight=self.pretrained_weight,
                                    device=self.device)

        if self.use_crf is True:
            args_crf = dict({'target_size': self.label_num, 'device': self.device})
            self.crf_layer = CRF(**args_crf)

    @staticmethod
    def _conv_filter(str_list):

        int_list = []
        str_list = str_list.split(",")
        for str in str_list:
            int_list.append(int(str))
        return int_list

    def forward(self, word, sentence_length, train=False):

        encoder_output = self.encoder_model(word, sentence_length)
        return encoder_output
