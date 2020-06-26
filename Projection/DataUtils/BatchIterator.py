import torch
import random
import numpy as np
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class BatchFeatures:
    """
    Batch_Features
    """
    def __init__(self):

        self.batch_length = 0
        self.inst = None
        self.word_features = 0
        self.label_features = 0
        self.sentence_length = []
        self.desorted_indices = None

    @staticmethod
    def cuda(features):

        features.word_features = features.word_features.cuda()
        features.label_features = features.label_features.cuda()


class Iterators:
    """
    Iterators
    """
    def __init__(self, batch_size=None, data=None, operator=None, device=None, config=None):
        self.config = config
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.operator = operator
        self.operator_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self):
        """
        :param batch_size:  batch size
        :param data:  data
        :param operator:
        :param config:
        :return:
        """
        assert isinstance(self.data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(self.batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        for id_data in range(len(self.data)):
            self.config.logger.info("*****************    create {} iterator    **************".format(id_data + 1))
            self._convert_word2id(self.data[id_data], self.operator)
            self.features = self._Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                       operator=self.operator, device=self.device)
            self.data_iter.append(self.features)
            self.features = []
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def _convert_word2id(insts, operator):

        for inst in insts:
            for index in range(inst.words_size):
                word = inst.words[index]
                wordId = operator.word_alphabet.from_string(word)
                # if wordID is None:
                if wordId == -1:
                    wordId = operator.word_unkId
                inst.words_index.append(wordId)

                label = inst.labels[index]
                labelId = operator.label_alphabet.from_string(label)
                inst.label_index.append(labelId)


    def _Create_Each_Iterator(self, insts, batch_size, operator, device):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            if len(batch) == batch_size or count_inst == len(insts):
                one_batch = self._Create_Each_Batch(insts=batch, batch_size=batch_size, operator=operator, device=device)
                self.features.append(one_batch)
                batch = []
        self.config.logger.info("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, insts, batch_size, operator, device):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        max_label_size = -1
        sentence_length = []
        for inst in insts:
            sentence_length.append(inst.words_size)
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size

            if len(inst.labels) > max_label_size:
                max_label_size = len(inst.labels)
        assert max_word_size == max_label_size

        batch_word_features = np.zeros((batch_length, max_word_size))
        batch_label_features = np.zeros((batch_length * max_word_size))

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features[id_inst][id_word_index] = operator.word_paddingId

                if id_word_index < len(inst.label_index):
                    batch_label_features[id_inst * max_word_size + id_word_index] = inst.label_index[id_word_index]
                else:
                    batch_label_features[id_inst * max_word_size + id_word_index] = operator.label_paddingId

        batch_word_features = torch.from_numpy(batch_word_features).long()
        batch_label_features = torch.from_numpy(batch_label_features).long()
        
        # batch
        features = BatchFeatures()
        features.batch_length = batch_length
        features.inst = insts
        features.word_features = batch_word_features
        features.label_features = batch_label_features
        features.sentence_length = sentence_length
        features.desorted_indices = None

        if device != cpu_device:
            features.cuda(features)
        return features
