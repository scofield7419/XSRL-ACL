import time
import random, tqdm
import torch
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.eval import Eval
from DataUtils.Common import *

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
        self.target = kwargs["target"]
        self.average_batch = self.config.average_batch
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        self.loss_function = self._loss(learning_algorithm=self.config.learning_algorithm,
                                        label_paddingId=self.config.arg_paddingId, use_crf=self.use_crf)
        self.config.logger.info(self.optimizer)
        self.config.logger.info(self.loss_function)
        self.best_score = Best_Result()
        self.train_eval, self.dev_eval, self.test_eval = Eval(), Eval(), Eval()


    def _loss(self, learning_algorithm, label_paddingId, use_crf=False):
        """
        :param learning_algorithm:
        :param label_paddingId:
        :param use_crf:
        :return:
        """
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
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                    epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):
        """lr decay 

        Args:
            epoch: int, epoch 
            init_lr:  initial lr
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :param config:
        :param backward_count:
        :return:
        """
        if backward_count % config.backward_batch_size == 0:  # or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch, config):
        """
        :param epoch:
        :return:
        """
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
                return True  # exit()
            else:
                return False
        else:
            return False

    @staticmethod
    def _get_model_args(batch_features):
        """
        :param batch_features:  Batch Instance
        :return:
        """
        elmo_char_seqs = batch_features.elmo_char_seqs
        elmo_word_seqs = batch_features.elmo_word_seqs
        word = batch_features.word_features
        lang = batch_features.lang
        pos = batch_features.pos_features
        prd = batch_features.prd_features
        x_prd_posi = batch_features.prd_posi_features
        mask = batch_features.mask
        sentence_length = batch_features.sentence_length
        tags = batch_features.label_features
        return elmo_char_seqs, elmo_word_seqs, word,lang, pos, prd, x_prd_posi, mask, sentence_length, tags

    def _calculate_loss(self, feats, mask, tags):
        """
        Args:
            feats: size = (batch_size, seq_len, tag_size)
            mask: size = (batch_size, seq_len)
            tags: size = (batch_size, seq_len)
        """
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
        """
        :return:
        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        self.config.logger.info('\n\n')
        self.config.logger.info('=-' * 50)

        for epoch in range(1, epochs + 1):
            self.train_iter.reset_flag4trainset()
            self.config.logger.info("\n\n### Epoch: {}/{} ###".format(epoch, epochs))
            self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            self.config.logger.info("current lr: {}".format(self.optimizer.param_groups[0].get("lr")))
            start_time = time.time()
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            self.config.logger.info('=-' * 10)
            batch_count = 0
            for batch_features in tqdm.tqdm(self.train_iter):
                batch_count += 1
                backward_count += 1
                elmo_char_seqs, elmo_word_seqs, word,lang, pos, prd, x_prd_posi, mask, sentence_length, tags = self._get_model_args(
                    batch_features)
                logit = self.model(elmo_char_seqs, elmo_word_seqs, word,lang,pos,  prd, x_prd_posi, mask, sentence_length,
                                   train=True)
                loss = self._calculate_loss(logit, mask, tags)
                loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                steps += 1

            if self.use_crf is True:
                p, r, f, acc_ = self.getAccCRF(self.train_eval, batch_features, logit, mask, self.config)
            else:
                p, r, f, acc_ = self.getAcc(self.train_eval, batch_features, logit, self.config)
            self.config.logger.info(
                "batch_count:{} , loss: {:.4f}, p: {:.4f}%  r: {:.4f}% , f: {:.4f}%, ACC: {:.4f}%".format(
                    batch_count, loss.item(), p, r, f, acc_))

            end_time = time.time()
            self.config.logger.info("Train Time {:.3f}".format(end_time - start_time))
            self.config.logger.info('=-' * 10)
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self.config.logger.info('=-' * 10)
            if self._early_stop(epoch=epoch, config=self.config):
                return
            self.config.logger.info('=-' * 15)

        self.save_training_summary()

    def save_training_summary(self):
        self.config.logger.info("Copy the last model ckps to {} as backup.".format(self.config.save_dir))


        self.config.logger.info("save the training summary at end of the log file.")
        self.config.logger.info("\n")
        self.config.logger.info("*" * 25)

        self.config.logger.info("*" * 10)
        self.config.logger.info("features:")
        if self.config.is_predicate:
            self.config.logger.info("\tpredicate, dim: %d" % self.config.prd_embed_dim)


        self.config.logger.info("*" * 10)
        self.config.logger.info("model:")
        self.config.logger.info(self.model)

        self.config.logger.info("*" * 10)
        self.config.logger.info("training:")
        self.config.logger.info('\tbatch size: %d' % self.config.batch_size)

        self.config.logger.info("*" * 10)
        self.config.logger.info("best performance:")
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
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """

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
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config, config.save_model_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config, config.save_model_dir, config.model_name, self.best_score)
        else:
            self.config.logger.info()

    def eval_batch(self, data_iter, model, eval_instance, best_score, epoch, config, test=False):
        """
        :param data_iter:  eval batch data iterator
        :param model: eval model
        :param eval_instance:
        :param best_score:
        :param epoch:
        :param config: config
        :param test:  whether to test
        :return: None
        """
        test_flag = "Test"
        if test is False:
            test_flag = "Dev"

        model.eval()
        gold_labels = []
        predict_labels = []
        all_sentence_length = []
        for batch_features in tqdm.tqdm(data_iter):
            elmo_char_seqs, elmo_word_seqs, word,lang, pos, prd, x_prd_posi, mask, sentence_length, tags = self._get_model_args(
                batch_features)
            logit = model(elmo_char_seqs, elmo_word_seqs, word,lang,pos,  prd, x_prd_posi, mask, sentence_length, train=False)
            all_sentence_length.extend(sentence_length)

            if self.use_crf is False:
                predict_ids = torch_max(logit)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    label_ids = predict_ids[id_batch]
                    predict_label = []
                    for id_word in range(inst.words_size):
                        predict_label.append(config.argvocab.i2c[int(i)])
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
                        label.append(config.argvocab.i2c[int(i)])
                    predict_labels.append(label)

        p, r, f, acc_ = eval_instance.getFscore(predict_labels, gold_labels, all_sentence_length)

        if test is False:

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
            "{} at current epoch, p: {:.4f}%  r: {:.4f}% , f: {:.4f}%,  ACC: {:.3f}%".format(
                test_flag, p, r, f, acc_))

        if test is False:
            self.config.logger.info(
                "Till now, The Best Dev Result: p: {:.4f}%  r: {:.4f}% , f: {:.4f}%, Locate on {} Epoch.".format(
                    best_score.best_dev_p_score, best_score.best_dev_r_score, best_score.best_dev_f1_score,
                    best_score.best_epoch))
        elif test is True:
            self.config.logger.info(
                "Till now, The Best Test Result: p: {:.4f}%  r: {:.4f}% , f: {:.4f}%, Locate on {} Epoch.".format(
                    best_score.p, best_score.r, best_score.f, best_score.best_epoch))
            best_score.best_test = False


    def eval_external_batch(self, data_iter, config, meta_info=''):
        """
        :param data_iter:  eval batch data iterator
        :param model: eval model
        :param eval_instance:
        :param best_score:
        :param epoch:
        :param config: config
        :param test:  whether to test
        :return: None
        """
        eval = Eval()

        self.model.eval()

        gold_labels = []
        predict_labels = []
        all_sentence_length = []
        for batch_features in tqdm.tqdm(data_iter):
            elmo_char_seqs, elmo_word_seqs, word,lang, pos, prd, x_prd_posi, mask, sentence_length, tags = self._get_model_args(
                batch_features)
            logit = self.model(elmo_char_seqs, elmo_word_seqs, word,lang,pos,  prd, x_prd_posi, mask, sentence_length,
                               train=False)
            all_sentence_length.extend(sentence_length)

            if self.use_crf is False:
                predict_ids = torch_max(logit)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    label_ids = predict_ids[id_batch]
                    predict_label = []
                    for id_word in range(inst.words_size):
                        predict_label.append(config.argvocab.i2c[int(i)])
                    gold_labels.append(inst.labels)
                    predict_labels.append(predict_label)
            else:
                path_score, best_paths = self.model.crf_layer(logit, mask)
                for id_batch in range(batch_features.batch_length):
                    inst = batch_features.inst[id_batch]
                    gold_labels.append(inst.labels)
                    label_ids = best_paths[id_batch].cpu().data.numpy()[:inst.words_size]
                    label = []
                    for i in label_ids:
                        label.append(config.argvocab.i2c[int(i)])
                    predict_labels.append(label)

        p, r, f, acc_ = eval.getFscore(predict_labels, gold_labels, all_sentence_length)

        self.config.logger.info(
            "eval on {}%, p: {:.4f}%  r: {:.4f}% , f: {:.4f}%, ACC: {:.4f}%".format(
                meta_info, p, r, f, acc_))

    @staticmethod
    def getAcc(eval_train, batch_features, logit, config):
        """
        :param eval_acc:  eval instance
        :param batch_features:  batch data feature
        :param logit:  model output
        :param config:  config
        :return:
        """
        eval_train.clear_PRF()
        predict_ids = torch_max(logit)

        predict_labels = []
        gold_labels = []
        batch_length = []

        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            label_ids = predict_ids[id_batch]
            predict_label = []
            gold_label = inst.labels
            for id_word in range(inst.words_size):
                predict_label.append(config.argvocab.i2c[label_ids[id_word]])

            predict_labels.append(predict_label)
            gold_labels.append(gold_label)
            batch_length.append(inst.words_size)

            assert len(predict_label) == len(gold_label)

        p, r, f, acc_ = eval_train.getFscore(predict_labels, gold_labels, batch_length)
        return p, r, f, acc_

    def getAccCRF(self, eval_train, batch_features, logit, mask, config):

        eval_train.clear_PRF()

        predict_labels = []
        gold_labels = []
        batch_length = []

        path_score, best_paths = self.model.crf_layer(logit, mask)
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            gold_labels.append(inst.labels)
            label_ids = best_paths[id_batch].cpu().data.numpy()[:inst.words_size]
            label = []
            for i in label_ids:
                label.append(config.argvocab.i2c[int(i)])
            predict_labels.append(label)
            batch_length.append(inst.words_size)

            assert len(label) == len(inst.labels)

        p, r, f, acc_ = eval_train.getFscore(predict_labels, gold_labels, batch_length)
        return p, r, f, acc_
