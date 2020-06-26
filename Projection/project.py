import argparse
import datetime
import logging
import os, sys

import Config.config as configurable
from DataUtils.Alphabet import *
from DataUtils.BatchIterator import *
from DataUtils.DataLoader import DataLoader
from DataUtils.Embed import Embed
from Engines.SequenceLabeler import SequenceLabeler
from DataUtils.mainHelp import *
from DataUtils.Common import *

import torch
import random
import shutil
import torch.nn as nn
import torch.nn.utils as utils


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


def load_meta_data(config):
    config.logger.info("load meta data for checkpoint file.")
    start_time = time.time()
    # load alphabet from pkl
    alphabet_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_alphabet))
    config.logger.info(alphabet_dict.keys())
    alphabet = alphabet_dict["alphabet"]
    end_time = time.time()
    return alphabet


def load_model(config):
    config.logger.info("***************************************")
    model = SequenceLabeler(config)
    if config.device != cpu_device:
        model = model.cuda()

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

    config.logger.info(model)
    return model


def load_tagger(config):
    alphabet = load_meta_data(config)
    get_params(config=config, alphabet=alphabet)
    model = load_model(config)
    config.logger.info("model loaded successfully......")
    return model, alphabet


def load_parallel_data(source_file_path, target_file_path, aligning_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ''' 
    Src: upb of conllu style
    token	lemma	pos	dep_head	dep_lb	is_predicate  predicate	argument
    
    Tgt:
    token
    '''
    source_dataset = list()
    target_dataset = list()
    text_new_line = list()
    prd_new_line = list()
    arg_new_line = list()

    for index, li in enumerate(lines):
        if index == 0: continue
        li = li.strip()
        if li == '':
            source_dataset.append((text_new_line, prd_new_line, arg_new_line))
            text_new_line = list()
            prd_new_line = list()
            arg_new_line = list()
        else:
            cols = li.split('\t')
            token = cols[0].strip()
            predicate = cols[5].strip()
            argument = cols[6].strip()

            text_new_line.append(token)
            prd_new_line.append(predicate)
            arg_new_line.append(argument)

    # print('length of %s: %i' % (mode, len(dataset)))

    # --------
    text_new_line = list()

    with open(target_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for index, li in enumerate(lines):
            if index == 0: continue
            li = li.strip()
            if li == '':
                target_dataset.append(text_new_line)
                text_new_line = list()
            else:
                cols = li.split('\t')
                token = cols[0].strip()
                text_new_line.append(token)

    # --------
    ### core aligning probabilities
    def read_align(aligning_sentence, src_len, tgt_len):
        align_probs = np.full((tgt_len, src_len), 0.0, dtype=np.float64)
        index = int(0)
        # while index < tgt_len:
        for line in aligning_sentence:
            content_pieces = line.split('\t')
            content_index = int(content_pieces[0])
            if content_index != index:
                raise Exception('align file error: content_index %d, actual_index %d' % (content_index, index))
            for i in range(1, len(content_pieces) - 1):
                atom_values = content_pieces[i].split(' ')
                if len(atom_values) != 2:
                    raise Exception('invalid source index and prob: ' + content_pieces[i])
                src_index = int(atom_values[0])
                src_prob = float(atom_values[1])
                if src_index == -1:
                    src_index = src_len
                align_probs[index, src_index] = src_prob
            index = index + 1

        for index in range(1, tgt_len):
            align_probs[index, 0] = -1.0

        return align_probs


    aligning_probs_sentences = list()
    aligning_sentence = list()
    with open(aligning_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for index, li in enumerate(lines):
        li = li.strip()
        if li == '':
            # aligning_sentences.append(aligning_sentence)
            src_len = len(source_dataset[index])
            tgt_len = len(target_dataset[index])
            if tgt_len != len(aligning_sentence): raise Exception('bad alignment error! aligned length does not equal to target setence.')
            align_probs = read_align(aligning_sentence, src_len, tgt_len)
            aligning_probs_sentences.append(align_probs)

            aligning_sentence = list()
        else:
            aligning_sentence.append(li)

    return aligning_probs_sentences, source_dataset, target_dataset

from DataUtils.Instance import *
import torch.nn.functional as F


def infer_prob(tagger, alphabet, text_new_line):
    # create instanace
    words_size = len(text_new_line)
    word_features = np.zeros((1, words_size))
    for index in range(words_size):
        word = text_new_line[index]
        wordId = alphabet.word_alphabet.from_string(word)
        if wordId == -1:  wordId = alphabet.word_unkId  # if wordID is None:
        word_features[0][index] = wordId
    mask = word_features > 0
    sentence_length = words_size
    batch_word_features = torch.from_numpy(word_features).long()

    logit = tagger(batch_word_features, sentence_length, train=False)
    output = F.softmax(logit, dim=1)
    values, arg_max = torch.max(output, dim=2)
    pos_probs = values[0].cpu().data.numpy()

    # tagger.infer_prob()
    return pos_probs


def projecting(source_dataset, target_dataset, align_probs, tagger, alphabet, config):
    sucess = 0
    all_cnt = len(source_dataset)
    pseudo_target_dataset = []

    index = 0
    for src_sentence, tgt_sentence, align_prob in zip(source_dataset, target_dataset, align_probs):
        
        text_new_line, prd_new_line, arg_new_line = src_sentence

        # POS prob
        pos_probs = infer_prob(tagger, alphabet, tgt_sentence)


        # prd, arg labels at src side
        src_prd_indexs = [ind for ind, item, tk in zip(range(len(prd_new_line)), prd_new_line, text_new_line) if
                      item != '_']
        src_prd_labels = [item for item, tk in zip(prd_new_line, text_new_line)]# if item != '_']

        src_arg_indexs = [ind for ind, item, tk in zip(range(len(prd_new_line)), arg_new_line, text_new_line) if
                      item != '_']
        src_arg_labels = [item for item, tk in zip(arg_new_line, text_new_line)]# if item != '_']


        src_alignment_dict = {}
        tgt_alignment_dict = {}

        # get confidence scores
        max_src_alignment = np.argmax(align_prob, axis=0)
        for ind in range(len(prd_new_line)):
            if ind in src_prd_indexs or ind in src_arg_indexs:
                max_tgt_index = max_src_alignment[ind]
                src_alignment_dict[ind] = (max_tgt_index, align_prob[max_tgt_index][ind] * pos_probs[max_tgt_index])
                if max_tgt_index not in tgt_alignment_dict:
                    tgt_alignment_dict[max_tgt_index] = [ind]
                else:
                    tgt_alignment_dict[max_tgt_index].append(ind)


        # handle collision, thereafter only one element in each list in tgt_alignment_dict. 
        for tgt_ind in tgt_alignment_dict.keys():
            src_prd_args = tgt_alignment_dict[tgt_ind]
            if len(src_prd_args) >1:
                intersect = set(src_prd_indexs)&set(src_prd_args)
                if len(intersect) >= 1: #there prd for tgt_ind from src, and let it be
                    assert len(intersect) == 1
                    tgt_alignment_dict[tgt_ind] = intersect
                else: # no prd, solve the arg-arg collision
                    max_conf = -1
                    best_src_ = -1
                    for src_ in src_prd_args:
                        tgt_d_, confid_score = src_alignment_dict[src_]
                        if confid_score > max_conf: max_conf = confid_score; best_src_=src_
                    tgt_alignment_dict[tgt_ind] = [(max_conf, best_src_)]
            else:
                tgt_d_, confid_score = src_alignment_dict[src_prd_args[0]]
                tgt_alignment_dict[tgt_ind] = [(confid_score, src_prd_args[0])]

        # handle other outliers:
        # 1) must at least and only with one prd, no limitation for args.
        # 2) p(prd), p(arg) > alpha,
        prd_cnt = 0
        for tgt_ind in tgt_alignment_dict.keys():
            src_prd_args = tgt_alignment_dict[tgt_ind]
            # if len(src_prd_args) == 0: continue
            assert len(src_prd_args) == 1
            confid_score, src_ind = src_prd_args[0]
            if src_ind in src_prd_indexs:
                if confid_score < config.alpha:
                    prd_cnt = 0
                    break
                else:
                    prd_cnt += 1
            if src_ind in src_arg_indexs:
                if confid_score < config.alpha:
                    tgt_alignment_dict.pop(tgt_ind)


        # assert prd_cnt == 1, 'prd number does not equal to 1.'

        # transfer the prd&arg labels
        tgt_prd_labels = ['_' for _ in range(len(tgt_sentence))]
        tgt_arg_labels = ['_' for _ in range(len(tgt_sentence))]
        for tgt_ind in tgt_alignment_dict.keys():
            if prd_cnt == 1: # the only valid situation 
                src_prd_args = tgt_alignment_dict[tgt_ind]
                if len(src_prd_args) == 0: continue
                score, src_ind = src_prd_args[0]
                if src_ind in src_prd_indexs:
                    tgt_prd_labels[tgt_ind] = src_prd_labels[src_ind]
                if src_ind in src_arg_indexs:
                    tgt_arg_labels[tgt_ind] = src_arg_labels[src_ind]


        pseudo_target_dataset.append((tgt_sentence, tgt_prd_labels, tgt_arg_labels))
        sucess += 1

        print('[%d/%d], success: %d' % (index, all_cnt, sucess))
        index += 1

    return pseudo_target_dataset

def save_data(data, file_name):
    '''
    token	predicate	argument
    '''
    text_new_line = list()
    for text, prd, arg in data:
        for tk_text, tk_prd, tk_arg in zip(text, prd, arg):
            text_new_line.append(tk_text + '\t' + tk_prd + '\t' + tk_arg)
        text_new_line.append('')

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_new_line))



def main():
    ### loading configuration
    config = parse_argument()

    ### loading POS tagger
    tagger,alphabet = load_tagger(config)

    modes = ['train', 'test', 'dev']
    for lan_pair in os.listdir(config.parallel_path_root_path):
        cur_lan = os.path.join(config.parallel_path_root_path, lan_pair)
        for item in os.listdir(cur_lan):
            primary_key_names = []
            for mode in modes:
                if (mode not in item) or ('src' not in item): continue
                primary_key_names.append(item)

            for item in primary_key_names:
                assert 'src' in item

                source_file_name = item
                source_file_path = os.path.join(cur_lan,source_file_name)
                target_file_name = item.replace('src', 'tgt')
                target_file_path = os.path.join(cur_lan,target_file_name)
                aligning_file_name = item.replace('src', 'src2tgt-align').replace('conllu', 'prob')
                aligning_file_path = os.path.join(cur_lan,aligning_file_name)
                pseudo_target_file_name = item.replace('src', 'pseudo_tgt')
                pseudo_target_file_path = os.path.join(pseudo_target_file_name,target_file_name)


                ### loading parallel data
                align_probs, source_dataset, target_dataset = load_parallel_data(source_file_path, target_file_path, aligning_file_path)

                ### annotation projection
                pseudo_target_dataset = projecting(source_dataset, target_dataset, align_probs, tagger, alphabet, config)

                save_data(pseudo_target_dataset, pseudo_target_file_path)


if __name__ == "__main__":
    main()
