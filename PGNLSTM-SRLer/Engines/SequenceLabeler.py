import torch
import torch.nn as nn
import random
from Engines.Encoder import Encoder
from Engines.CRF import CRF
from DataUtils.Common import *

torch.manual_seed(seed_num)
random.seed(seed_num)


class SequenceLabeler(nn.Module):

    def __init__(self, config, lang2id):
        super(SequenceLabeler, self).__init__()
        self.config = config

        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.word_paddingId = config.word_paddingId

        self.is_predicate_position = config.is_predicate_position

        self.is_predicate = config.is_predicate
        self.prd_embed_num = config.prd_embed_num
        self.prd_embed_dim = config.prd_embed_dim
        self.prd_paddingId = config.prd_paddingId

        self.is_pos = config.is_pos
        self.pos_embed_num = config.pos_embed_num
        self.pos_embed_dim = config.pos_embed_dim
        self.pos_paddingId = config.pos_paddingId

        self.label_num = config.class_num
        self.lang_emb_dim = config.lang_emb_dim

        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout

        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers

        self.use_crf = config.use_crf

        self.device = config.device
        print('self.device', self.device)

        self.target_size = self.label_num if self.use_crf is False else self.label_num + 2

        self.encoder_model = Encoder(embed_num=self.embed_num, embed_dim=self.embed_dim,
                                     lang2id=lang2id, lang_emb_dim=self.lang_emb_dim,
                                     is_predicate_position=self.is_predicate_position,
                                     is_predicate=self.is_predicate, prd_embed_num=self.prd_embed_num,
                                     prd_embed_dim=self.prd_embed_dim, prd_paddingId=self.prd_paddingId,
                                     is_pos=self.is_pos, pos_embed_num=self.pos_embed_num,
                                     pos_embed_dim=self.pos_embed_dim, pos_paddingId=self.pos_paddingId,
                                    weight_file=config.weight_file, options_file=config.options_file, is_elmo=config.is_elmo,
                                     label_num=self.target_size,
                                     elmo_vocab=config.elmo_vocab,
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

    def forward(self, elmo_char_seqs, elmo_word_seqs, word,lang, pos, prd, x_prd_posi, mask, sentence_length, train=False):

        encoder_output = self.encoder_model(elmo_char_seqs, elmo_word_seqs, word, lang, pos, prd, x_prd_posi, mask, sentence_length)
        return encoder_output
