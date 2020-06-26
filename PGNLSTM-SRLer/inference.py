import os
import torch
from DataUtils.utils import *
from DataUtils.Common import *


def load_test_model(model, config):
    """
    :param model:  initial model
    :param config:  config
    :return:  loaded model
    """
    if config.t_model is None:
        test_model_dir = config.save_model_dir
        test_model_name = "{}.pt".format(config.model_name)
        test_model_path = os.path.join(test_model_dir, test_model_name)

        if os.path.isfile(test_model_path):
            config.logger.info("load default best model from {}".format(test_model_path))
        else:
            config.logger.info("please specify the pre-trained model")
            exit()
    else:
        test_model_path = config.t_model
        config.logger.info("load user model from {}".format(test_model_path))
    model.load_state_dict(torch.load(test_model_path))
    return model


def load_test_data(train_iter=None, dev_iter=None, test_iter=None, config=None):
    """
    :param train_iter:  train data
    :param dev_iter:  dev data
    :param test_iter:  test data
    :param config:  config
    :return:  data for test
    """
    data, path_source, path_result = None, None, None
    if config.t_data is None:
        config.logger.info("default[test] for model test.")
        data = test_iter
        path_source = ".".join([config.test_file, shuffle])
        path_result = "{}.out".format(path_source)
    elif config.t_data == "train":
        config.logger.info("train data for model test.")
        data = train_iter
        path_source = ".".join([config.train_file, shuffle])
        path_result = "{}.out".format(path_source)
    elif config.t_data == "dev":
        config.logger.info("dev data for model test.")
        data = dev_iter
        path_source = ".".join([config.dev_file, shuffle])
        path_result = "{}.out".format(path_source)
    elif config.t_data == "test":
        config.logger.info("test data for model test.")
        data = test_iter
        path_source = ".".join([config.test_file, shuffle])
        path_result = "{}.out".format(path_source)
    else:
        config.logger.info("Error value --- t_data = {}, must in [None, 'train', 'dev', 'test'].".format(config.t_data))
        exit()
    return data, path_source, path_result


class Inference(object):
    """
        Test Inference
    """
    def __init__(self, model, data, path_source, path_result, alphabet, use_crf, config):
        """
        :param model:  nn model
        :param data:  infer data
        :param path_source:  source data path
        :param path_result:  result data path
        :param alphabet:  alphabet
        :param config:  config
        """
        self.config = config

        config.logger.info("Initialize T_Inference")
        self.model = model
        self.data = data
        self.path_source = path_source
        self.path_result = path_result
        self.alphabet = alphabet
        self.config = config
        self.use_crf = use_crf

    def infer2file(self):
        """
        :return: None
        """

        self.config.logger.info('\n\n')
        self.config.logger.info('=-' * 20)
        self.config.logger.info("infer.....")
        self.model.eval()
        predict_labels = []
        predict_label = []
        all_count = len(self.data)
        now_count = 0
        for data in self.data:
            now_count += 1
            self.config.logger.info("infer with batch number {}/{} .".format(now_count, all_count))
            word, char, mask, sentence_length, tags = self._get_model_args(data)
            logit = self.model(word, char, sentence_length, train=False)
            if self.use_crf is False:
                predict_ids = torch_max(logit)
                for id_batch in range(data.batch_length):
                    inst = data.inst[id_batch]
                    label_ids = predict_ids[id_batch]
                    for id_word in range(inst.words_size):
                        predict_label.append(self.alphabet.label_alphabet.from_id(label_ids[id_word]))
            else:
                path_score, best_paths = self.model.crf_layer(logit, mask)
                for id_batch in range(data.batch_length):
                    inst = data.inst[id_batch]
                    label_ids = best_paths[id_batch].cpu().data.numpy()[:inst.words_size]
                    for i in label_ids:
                        predict_label.append(self.alphabet.label_alphabet.from_id(i))

        self.config.logger.info("\ninfer finished.")
        self.write2file(self.config, result=predict_label, path_source=self.path_source, path_result=self.path_result)

    @staticmethod
    def write2file(config, result, path_source, path_result):
        """
        :param result:
        :param path_source:
        :param path_result:
        :return:
        """
        config.logger.info('\n\n')
        config.logger.info('=-'*20)
        config.logger.info("write result to file {}".format(path_result))
        if os.path.exists(path_source) is False:
            config.logger.info("source data path[path_source] is not exist.")
        if os.path.exists(path_result):
            os.remove(path_result)
        file_out = open(path_result, encoding="UTF-8", mode="w")
        with open(path_source, encoding="UTF-8") as file:
            id = 0
            for line in file.readlines():
                if line == "\n":
                    file_out.write("\n")
                    continue
                line = line.strip().split()
                line.append(result[id])
                id += 1
                file_out.write(" ".join(line) + "\n")
                if id >= len(result):
                    break
        file_out.close()
        config.logger.info("\nfinished.")

    @staticmethod
    def _get_model_args(batch_features):
        """
        :param batch_features:  Batch Instance
        :return:
        """
        word = batch_features.word_features
        char = batch_features.char_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        tags = batch_features.label_features
        return word, char, mask, sentence_length, tags

