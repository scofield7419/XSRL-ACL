import os
import random

import torch

from DataUtils.Common import *
from DataUtils.Instance import Instance

torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _write_shuffle_inst_to_file(config, insts, path):
        """
        :return:
        """
        w_path = ".".join([path, shuffle])
        if os.path.exists(w_path):
            os.remove(w_path)
        file = open(w_path, encoding="UTF-8", mode="w")
        for id, inst in enumerate(insts):
            for word, po, dps, cot, prd, label in zip(inst.words, inst.pos, inst.dep, inst.dep_head, inst.prd,
                                                      inst.labels):
                file.write("\t".join([word, po, dps, str(cot), prd, label, "\n"]))
            file.write("\n")
        config.logger.info("write shuffle insts to file {}".format(w_path))


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """

    def __init__(self,domains, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        self.config = config
        self.domains = domains
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

    def load(self):
        """
        :return:
        """
        paths = self.path
        assert isinstance(paths, list), "Path Must Be In List"
        for path in paths:
            self.config.logger.info("Data Path {}".format(path))
            self.config.logger.info('')

        for id_data in range(len(paths)):
            this_insts = {sub_name:list() for sub_name in self.domains}
            for sub_name in self.domains:
                sub_data = paths[id_data][sub_name]
                self.config.logger.info("Loading Data From {}".format(sub_data))
                insts = self._Load_Each_Data(path=sub_data)
                this_insts[sub_name] = insts

            self.data_list.append(this_insts)

        if len(self.data_list) == 3:
            self.config.train_cnt = len(self.data_list[0])
            self.config.dev_cnt = len(self.data_list[1])
            self.config.test_cnt = len(self.data_list[2])
            return self.data_list[0], self.data_list[1], self.data_list[2]

    def load_single(self, domain):
        """
        :return:
        """
        paths = self.path
        assert isinstance(paths, list), "Path Must Be In List"
        for path in paths:
            self.config.logger.info("Data Path {}".format(path))
            self.config.logger.info('')

        for id_data in range(len(paths)):
            this_insts = {sub_name:list() for sub_name in self.domains}
            self.config.logger.info("Loading Data From {}".format(paths[id_data]))
            insts = self._Load_Each_Data(path=paths[id_data])
            this_insts[domain] = insts
            self.data_list.append(this_insts)

        if len(self.data_list) == 3:
            self.config.train_cnt = len(self.data_list[0])
            self.config.dev_cnt = len(self.data_list[1])
            self.config.test_cnt = len(self.data_list[2])
            return self.data_list[0], self.data_list[1], self.data_list[2]

    def _Load_Each_Data(self, path=None):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        with open(path, encoding="UTF-8") as f:
            inst = Instance()
            for ind, line in enumerate(f.readlines()):
                if ind == 0: continue
                line = line.strip()
                if line == "" and len(inst.words) != 0:
                    inst.words_size = len(inst.words)
                    insts.append(inst)
                    inst = Instance()
                else:
                    line = line.split()

                    inst.words.append(line[0])
                    inst.pos.append(line[1])
                    inst.prd.append(line[2])
                    inst.prd_posi.append(1 if line[2] != nullkey else 0)
                    inst.labels.append(line[3])
                if len(insts) == self.max_count:
                    break
            if len(inst.words) != 0:
                inst.words_size = len(inst.words)
                insts.append(inst)
        random.shuffle(insts)
        return insts
