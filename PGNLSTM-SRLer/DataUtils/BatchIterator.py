import torch
from torch.autograd import Variable
import random
import numpy as np
from DataUtils.Common import *

torch.manual_seed(seed_num)
random.seed(seed_num)

class BatchFeatures:

    def __init__(self):
        self.batch_length = 0
        self.inst = None
        self.lang = None
        self.mask = None
        self.word_features = None
        self.pos_features = None
        self.elmo_char_seqs = None
        self.elmo_word_seqs = None
        self.prd_features = None
        self.prd_posi_features = None
        self.label_features = None
        self.sentence_length = []
        self.desorted_indices = None

    @staticmethod
    def cuda(features):
        """
        :param features:
        :return:
        """
        features.word_features = features.word_features.cuda()
        features.elmo_char_seqs = features.elmo_char_seqs.cuda()
        features.elmo_word_seqs = features.elmo_word_seqs.cuda()
        features.label_features = features.label_features.cuda()
        features.pos_features = features.pos_features.cuda()
        features.prd_posi_features = features.prd_posi_features.cuda()
        features.prd_posi_features.requires_grad = False
        features.mask = features.mask.cuda()
        features.mask.requires_grad = False
        features.lang = features.lang.cuda()
        features.lang.requires_grad = False

        features.prd_features = features.prd_features.cuda()


class Iterators:
    """
    Iterators
    """

    def __init__(self, domains=None, batch_size=None, data=None, operator=None, lang2id=None, device=None, config=None):
        self.config = config
        self.domains = domains
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.elmo_toten_vocab = {}
        self.elmo_char_vocab = {paddingkey: k for k in range(262)}
        for id_, token in enumerate(config.elmo_vocab):
            self.elmo_toten_vocab[token] = id_

        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

        self.lang2id = lang2id

        self.w2i_word = config.wordembeddings.w2i
        self.w2i_pos = config.posembeddings.w2i
        self.w2i_prd = config.predembeddings.w2i
        self.w2i_arg = config.argvocab.c2i

        self.word_paddingId = config.word_paddingId
        self.pos_paddingId = config.pos_paddingId
        self.prd_paddingId = config.prd_paddingId
        self.arg_paddingId = config.arg_paddingId

        self.pointers = {ind: dict({sub_name: 0 for sub_name in self.domains}) for ind in range(3)}
        self.lengths = {ind: dict({sub_name: len(self.data[ind][sub_name]) for sub_name in self.domains}) for ind in
                        range(3)}

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
            self._convert_word2id(self.data[id_data])
            self.features = self._Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                       device=self.device)
            self.data_iter.append(self.features)
            self.features = []
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    def reset_flag4trainset(self):
        self.flag = {sub_name: True for sub_name in self.domains}

    def createSetIterator(self, setIndex=0, T_domain=None, isTrain=True):

        self.config.logger.info("*****************    create {train} iterator    **************")
        for domain in self.domains:
            self._convert_word2id(self.data[setIndex][domain])

        batch_size = self.batch_size[setIndex]

        if T_domain == None:
            domain = random.choice(self.domains)

            k = 0
            while not self.flag[domain]:
                domain = random.choice(self.domains)
                k += 1

                if k > len(self.flag):
                    return
        else:
            domain = T_domain

        if isTrain:
            cur_start_pointer = self.pointers[setIndex][domain]

            if (cur_start_pointer % self.lengths[setIndex][domain]) <= (
                        (cur_start_pointer + batch_size) % self.lengths[setIndex][domain]):
                insts = self.data[setIndex][domain][cur_start_pointer % self.lengths[setIndex][domain]:
                (cur_start_pointer + batch_size) % self.lengths[setIndex][domain]]
                self.pointers[setIndex][domain] = (cur_start_pointer + batch_size) % self.lengths[setIndex][domain]

            else:
                insts = self.data[setIndex][domain][
                        cur_start_pointer % self.lengths[setIndex][domain]:self.lengths[setIndex][domain]]
                insts.extend(random.sample(self.data[setIndex][domain], k=(batch_size - len(insts))))
                self.pointers[setIndex][domain] = 0
                self.flag[domain] = False

            one_batch = self._Create_Each_Batch(domain=domain, insts=insts, batch_size=self.batch_size[setIndex],
                                                device=self.device)
            yield one_batch
        else:
            batch = []
            count_inst = 0
            for index, inst in enumerate(self.data[setIndex][domain]):
                batch.append(inst)
                count_inst += 1
                if len(batch) == self.batch_size[setIndex]:
                    one_batch = self._Create_Each_Batch(domain=domain, insts=batch,
                                                        batch_size=self.batch_size[setIndex],
                                                        device=self.device)

                    yield one_batch
                    batch = []
                if count_inst == len(self.data[setIndex]):
                    one_batch = self._Create_Each_Batch(domain=domain, insts=batch, batch_size=len(batch),
                                                        device=self.device)
                    yield one_batch
                    batch = []

    # @staticmethod
    def _convert_word2id(self, insts):
        """
        :param insts:
        :param operator:
        :return:
        """
        for inst in insts:
            for index in range(inst.words_size):

                word = inst.words[index]
                pos_ = inst.pos[index]
                prd_ = inst.prd[index]
                label_ = inst.labels[index]

                if word in self.w2i_word.keys():
                    wordId = self.w2i_word[word]
                else:
                    wordId = self.w2i_word[unkkey]
                inst.words_index.append(wordId)

                if word in self.elmo_toten_vocab.keys():
                    elmowordId = self.elmo_toten_vocab[word]
                else:
                    elmowordId = self.elmo_toten_vocab[elmo_unkkey]
                inst.elmo_token_index.append(elmowordId)


                if pos_ in self.w2i_pos.keys():
                    poslId = self.w2i_pos[pos_]
                else:
                    poslId = self.w2i_pos[unkkey]
                inst.pos_index.append(poslId)


                if prd_ == nullkey:
                    prdId = self.w2i_prd[nullkey]
                else:
                    prdId = self.w2i_prd[prd_]
                inst.prd_index.append(prdId)

                inst.prd_posi_index=inst.prd_posi

                if label_ in self.w2i_arg.keys():
                    labelId = self.w2i_arg[label_]
                else:
                    labelId = self.w2i_arg[unkkey]
                inst.label_index.append(labelId)

    def _Create_Each_Iterator(self, insts, batch_size, device):
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
                one_batch = self._Create_Each_Batch(insts=batch, batch_size=batch_size, device=device)
                self.features.append(one_batch)
                batch = []
        self.config.logger.info("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, domain, insts, batch_size, device):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """

        def sequence_mask(sequence_length, max_len=None):
            if max_len is None:
                max_len = sequence_length.data.max()
            batch_size = sequence_length.size(0)
            seq_range = torch.arange(0, max_len).long()
            seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
            seq_range_expand = Variable(seq_range_expand)
            if sequence_length.is_cuda:
                seq_range_expand = seq_range_expand.cuda()
            seq_length_expand = (sequence_length.unsqueeze(1)
                                 .expand_as(seq_range_expand))
            return seq_range_expand < seq_length_expand

        batch_length = len(insts)

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
        batch_pos_features = np.zeros((batch_length, max_word_size))
        batch_prd_features = np.zeros((batch_length, max_word_size))
        batch_prd_posi_features = np.zeros((batch_length, max_word_size))

        batch_elmo_char_seqs = np.zeros((batch_length, max_word_size, 262))
        batch_elmo_word_seqs = np.zeros((batch_length, max_word_size))

        batch_label_features = np.zeros((batch_length * max_word_size))

        for id_inst in range(batch_length):
            inst = insts[id_inst]

            for id_word_index in range(max_word_size):

                if id_word_index < inst.words_size:
                    batch_elmo_word_seqs[id_inst][id_word_index] = inst.elmo_token_index[id_word_index]
                else:
                    batch_elmo_word_seqs[id_inst][id_word_index] = 0

                if id_word_index < inst.words_size:
                    batch_word_features[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features[id_inst][id_word_index] = self.word_paddingId

                if id_word_index < inst.words_size:
                    batch_pos_features[id_inst][id_word_index] = inst.pos_index[id_word_index]
                else:
                    batch_pos_features[id_inst][id_word_index] = self.pos_paddingId

                if id_word_index < inst.words_size:
                    batch_prd_posi_features[id_inst][id_word_index] = inst.prd_posi_index[id_word_index]
                else:
                    batch_prd_posi_features[id_inst][id_word_index] = 0

                if id_word_index < inst.words_size:
                    batch_prd_features[id_inst][id_word_index] = inst.prd_index[id_word_index]
                else:
                    batch_prd_features[id_inst][id_word_index] = self.prd_paddingId

                if id_word_index < len(inst.label_index):
                    batch_label_features[id_inst * max_word_size + id_word_index] = inst.label_index[id_word_index]
                else:
                    batch_label_features[id_inst * max_word_size + id_word_index] = self.arg_paddingId


        batch_elmo_char_seqs = torch.from_numpy(batch_elmo_char_seqs).long()
        batch_elmo_word_seqs = torch.from_numpy(batch_elmo_word_seqs).long()
        batch_word_features = torch.from_numpy(batch_word_features).long()
        batch_pos_features = torch.from_numpy(batch_pos_features).long()
        batch_prd_features = torch.from_numpy(batch_prd_features).long()
        batch_label_features = torch.from_numpy(batch_label_features).long()
        sentence_length = torch.from_numpy(np.array(sentence_length)).long()

        batch_prd_posi_features = torch.from_numpy(batch_prd_posi_features).long()
        batch_prd_posi_features.requires_grad = False

        # batch
        features = BatchFeatures()
        features.batch_length = batch_length
        features.lang = torch.from_numpy(self.lang2id[domain]).long()
        features.inst = insts
        features.sentence_length = sentence_length
        features.mask = sequence_mask(sentence_length, max_word_size)
        features.word_features = batch_word_features
        features.pos_features = batch_pos_features
        features.prd_features = batch_prd_features
        features.prd_posi_features = batch_prd_posi_features
        features.label_features = batch_label_features
        features.desorted_indices = None

        features.elmo_char_seqs = batch_elmo_char_seqs
        features.elmo_word_seqs = batch_elmo_word_seqs

        if device != cpu_device:
            features.cuda(features)
        return features
