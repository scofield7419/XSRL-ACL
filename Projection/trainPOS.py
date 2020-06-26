import argparse
import datetime
import Config.config as configurable
from DataUtils.mainHelp import *
from DataUtils.Alphabet import *
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.eval import Eval, EvalPRF
from DataUtils.Common import *
import logging
import os
import sys
import time
import shutil
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils


# solve default encoding problem
from imp import reload

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)



class Train(object):

    def __init__(self, **kwargs):

        self.config = kwargs["config"]
        self.config.logger.info("Training Start......")
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.use_crf = self.config.use_crf
        self.average_batch = self.config.average_batch
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        self.loss_function = self._loss(learning_algorithm=self.config.learning_algorithm,
                                        label_paddingId=self.config.label_paddingId, use_crf=self.use_crf)
        self.config.logger.info(self.optimizer)
        self.config.logger.info(self.loss_function)
        self.best_score = Best_Result()
        self.train_eval, self.dev_eval, self.test_eval = Eval(), Eval(), Eval()
        self.train_iter_len = len(self.train_iter)

    def _loss(self, learning_algorithm, label_paddingId, use_crf=False):

        if use_crf:
            loss_function = self.model.crf_layer.neg_log_likelihood_loss
            return loss_function
        elif learning_algorithm == "SGD":
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="sum")
            return loss_function
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="mean")
            return loss_function

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):

        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):

        if config.use_lr_decay is True and epoch > config.max_patience and (
                    epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):

        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):

        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch, config):

        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            self.config.logger.info(
                "Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                self.end_of_epoch = epoch
                self.config.logger.info(
                    "\n\nEarly Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                self.save_training_summary()
                exit()

    @staticmethod
    def _get_model_args(batch_features):

        word = batch_features.word_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        tags = batch_features.label_features
        return word, mask, sentence_length, tags

    def _calculate_loss(self, feats, mask, tags):

        if not self.use_crf:
            batch_size, max_len = feats.size(0), feats.size(1)
            lstm_feats = feats.view(batch_size * max_len, -1)
            tags = tags.view(-1)
            return self.loss_function(lstm_feats, tags)
        else:
            loss_value = self.loss_function(feats, mask, tags)
        if self.average_batch:
            batch_size = feats.size(0)
            loss_value /= float(batch_size)
        return loss_value

    def train(self):

        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate
        self.config.logger.info('\n\n')
        self.config.logger.info('=-' * 50)
        self.config.logger.info('batch number: %d' % len(self.train_iter))

        for epoch in range(1, epochs + 1):
            self.config.logger.info("\n\n### Epoch: {}/{} ###".format(epoch, epochs))
            self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            self.config.logger.info("current lr: {}".format(self.optimizer.param_groups[0].get("lr")))
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            self.config.logger.info('=-' * 10)
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                word, mask, sentence_length, tags = self._get_model_args(batch_features)
                logit = self.model(word, sentence_length, train=True)
                loss = self._calculate_loss(logit, mask, tags)
                loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.getAcc(self.train_eval, batch_features, logit, self.config)
                    self.config.logger.info(
                        "batch_count:{} , loss: {:.4f}, [TAG-ACC: {:.4f}%]".format(batch_count + 1,
                                                                                   loss.item(),
                                                                                   self.train_eval.acc()))
            end_time = time.time()
            self.config.logger.info("Train Time {:.3f}".format(end_time - start_time))
            self.config.logger.info('=-' * 10)
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self.config.logger.info('=-' * 10)
            self._model2file(model=self.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch, config=self.config)
            self.config.logger.info('=-' * 15)
        self.save_training_summary()

    def save_training_summary(self):
        self.config.logger.info("Copy the last model ckps to {} as backup.".format(self.config.save_dir))
        shutil.copytree(self.config.save_model_dir,
                        "/".join([self.config.save_dir, self.config.save_model_dir + "_bak"]))

        self.config.logger.info("save the training summary at end of the log file.")
        self.config.logger.info("\n")
        self.config.logger.info("*" * 25)

        par_path = os.path.dirname(self.config.train_file)
        self.config.logger.info("dataset:\n\t %s" % par_path)
        self.config.logger.info("\ttrain set count: %d" % self.config.train_cnt)
        self.config.logger.info("\tdev set count: %d" % self.config.dev_cnt)
        self.config.logger.info("\ttest set count: %d" % self.config.test_cnt)

        self.config.logger.info("*" * 10)
        self.config.logger.info("model:")
        self.config.logger.info(self.model)

        self.config.logger.info("*" * 10)
        self.config.logger.info("training:")
        self.config.logger.info('\tbatch size: %d' % self.config.batch_size)
        self.config.logger.info('\tbatch count: %d' % len(self.train_iter))

        self.config.logger.info("*" * 10)
        self.config.logger.info("best performance:")
        self.config.logger.info("\tend at epoch: %d" % self.end_of_epoch)
        self.config.logger.info("\tbest at epoch: %d" % self.best_score.best_epoch)
        self.config.logger.info("\tdev(%):")
        self.config.logger.info("\t\tprecision, %.5f" % self.best_score.best_dev_p_score)
        self.config.logger.info("\t\trecall, %.5f" % self.best_score.best_dev_r_score)
        self.config.logger.info("\t\tf1, %.5f" % self.best_score.best_dev_f1_score)
        self.config.logger.info("\ttest(%):")
        self.config.logger.info("\t\tprecision, %.5f" % self.best_score.p)
        self.config.logger.info("\t\trecall, %.5f" % self.best_score.r)
        self.config.logger.info("\t\tf1, %.5f" % self.best_score.f)

        self.config.logger.info("*" * 25)

    def eval(self, model, epoch, config):

        self.dev_eval.clear_PRF()
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.dev_eval, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        self.config.logger.info("Dev Time: {:.3f}".format(eval_end_time - eval_start_time))
        self.config.logger.info('=-' * 10)

        self.test_eval.clear_PRF()
        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.test_eval, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        self.config.logger.info("Test Time: {:.3f}".format(eval_end_time - eval_start_time))

    def _model2file(self, model, config, epoch):

        if config.save_model and config.save_all_model:
            save_model_all(model, config, config.save_model_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config, config.save_model_dir, config.model_name, self.best_score)
        else:
            self.config.logger.info()

    def eval_batch(self, data_iter, model, eval_instance, best_score, epoch, config, test=False):

        test_flag = "Test"
        if test is False:  # dev
            test_flag = "Dev"

        model.eval()  # set flag for pytorch
        eval_PRF = EvalPRF()
        gold_labels = []
        predict_labels = []
        for batch_features in data_iter:
            word, mask, sentence_length, tags = self._get_model_args(batch_features)
            logit = model(word, sentence_length, train=False)

            if self.use_crf is False:
                predict_ids = torch_max(logit)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    label_ids = predict_ids[id_batch]
                    predict_label = []
                    for id_word in range(inst.words_size):
                        predict_label.append(config.create_alphabet.label_alphabet.from_id(label_ids[id_word]))
                    gold_labels.append(inst.labels)
                    predict_labels.append(predict_label)
            else:
                path_score, best_paths = model.crf_layer(logit, mask)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    gold_labels.append(inst.labels)
                    label_ids = best_paths[id_batch].cpu().data.numpy()[:inst.words_size]
                    label = []
                    for i in label_ids:
                        # self.config.logger.info("\n", i)
                        label.append(config.create_alphabet.label_alphabet.from_id(int(i)))
                    predict_labels.append(label)

        for p_label, g_label in zip(predict_labels, gold_labels):
            eval_PRF.evalPRF(predict_labels=p_label, gold_labels=g_label, eval=eval_instance)

        cor = 0
        totol_leng = sum([len(predict_label) for predict_label in predict_labels])
        for p_lable, g_lable in zip(predict_labels, gold_labels):
            for p_lable_, g_lable_ in zip(p_lable, g_lable):
                if p_lable_ == g_lable_:
                    cor += 1
        acc_ = cor / totol_leng * 100

        p, r, f = eval_instance.getFscore()

        if test is False:  # dev
            best_score.current_dev_score = f
            if f >= best_score.best_dev_f1_score:
                best_score.best_dev_f1_score = f
                best_score.best_dev_p_score = p
                best_score.best_dev_r_score = r
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:  # test
            best_score.p = p
            best_score.r = r
            best_score.f = f
        self.config.logger.info(
            "{} at current epoch, precision: {:.4f}%  recall: {:.4f}% , f-score: {:.4f}%,  [TAG-ACC: {:.3f}%]".format(
                test_flag, p, r, f, acc_))
        if test is False:
            self.config.logger.info(
                "Till now, The Best Dev Result: precision: {:.4f}%  recall: {:.4f}% , f-score: {:.4f}%, Locate on {} Epoch.".format(
                    best_score.best_dev_p_score, best_score.best_dev_r_score, best_score.best_dev_f1_score,
                    best_score.best_epoch))
        elif test is True:
            self.config.logger.info(
                "Till now, The Best Test Result: precision: {:.4f}%  recall: {:.4f}% , f-score: {:.4f}%, Locate on {} Epoch.".format(
                    best_score.p, best_score.r, best_score.f, best_score.best_epoch))
            best_score.best_test = False

    @staticmethod
    def getAcc(eval_acc, batch_features, logit, config):

        eval_acc.clear_PRF()
        predict_ids = torch_max(logit)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            label_ids = predict_ids[id_batch]
            predict_label = []
            gold_lable = inst.labels
            for id_word in range(inst.words_size):
                predict_label.append(config.create_alphabet.label_alphabet.from_id(label_ids[id_word]))
            assert len(predict_label) == len(gold_lable)
            cor = 0
            for p_lable, g_lable in zip(predict_label, gold_lable):
                if p_lable == g_lable:
                    cor += 1
            eval_acc.correct_num += cor
            eval_acc.gold_num += len(gold_lable)



def load_test_data(train_iter=None, dev_iter=None, test_iter=None, config=None):

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

    def __init__(self, model, data, path_source, path_result, alphabet, use_crf, config):

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

        self.config.logger.info('\n\n')
        self.config.logger.info('=-' * 20)
        self.config.logger.info("infer.....")
        self.model.eval()
        predict_label = []
        all_count = len(self.data)
        now_count = 0
        for data in self.data:
            now_count += 1
            self.config.logger.info("infer with batch number {}/{} .".format(now_count, all_count))
            word, mask, sentence_length, tags = self._get_model_args(data)
            logit = self.model(word, sentence_length, train=False)
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

        word = batch_features.word_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        tags = batch_features.label_features
        return word, mask, sentence_length, tags


def start_train(train_iter, dev_iter, test_iter, model, config):

    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()
    config.logger.info("Finish Train.")


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):

    config.logger.info("\nTesting Start......")
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = Inference(model=model, data=data, path_source=path_source, path_result=path_result, alphabet=alphabet,
                      use_crf=config.use_crf, config=config)
    infer.infer2file()
    config.logger.info("Finished Test.")


def main():

    if not os.path.exists(config.save_model_dir): os.mkdir(config.save_model_dir)

    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)

    get_params(config=config, alphabet=alphabet)

    save_dictionary(config=config)

    model = load_model(config)

    if config.mode == 'train':
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.mode == 'test':
        start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
        exit()


def get_logger(log_dir):
    log_file = log_dir
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger


def parse_argument():
    
    parser = argparse.ArgumentParser(description="POS tagging")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    args = parser.parse_args()
    config = configurable.Configurable(config_file=args.config_file)

    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    subset_name = config.train_file.split('/')[-3]
    subset_name_dir = os.path.join(config.save_checkpoint, subset_name)
    if not os.path.isdir(subset_name_dir): os.makedirs(subset_name_dir)

    config.save_dir = os.path.join(subset_name_dir, config.mulu)
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)

    logger = get_logger(os.path.join(config.save_dir, 'system.log'))
    config.logger = logger

    return config


def set_cuda():
    config.logger.info("\nUsing GPU To Train......")
    device_number = config.device[-1]
    torch.cuda.set_device(int(device_number))
    config.logger.info("Current Cuda Device {}".format(torch.cuda.current_device()))
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    config.logger.info("torch.cuda.initial_seed", torch.cuda.initial_seed())


if __name__ == "__main__":
    config = parse_argument()
    config.print_args()
    config.logger.info("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    config.logger.info('\n')
    if config.device != cpu_device:
        set_cuda()
    config.logger.info('\n')
    main()
