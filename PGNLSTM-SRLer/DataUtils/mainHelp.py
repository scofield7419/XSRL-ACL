import shutil
import time, tqdm, os, sys
from imp import reload
from DataUtils.DataLoader import DataLoader

from DataUtils.Alphabet import *
from DataUtils.BatchIterator import *
from DataUtils.Embed import Embed, Embeddings
from Engines.SequenceLabeler import SequenceLabeler
from inference import load_test_model

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)

def get_learning_algorithm(config):
    """
    :param config:  config
    :return:  optimizer algorithm
    """
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    config.logger.info("learning algorithm is {}.".format(algorithm))
    return algorithm

def save_dict2file(config, dict, path):
    """
    :param dict:  dict
    :param path:  path to save dict
    :return:
    """
    config.logger.info("Saving dictionary")
    if os.path.exists(path):
        config.logger.info("path {} is exist, deleted.".format(path))
    file = open(path, encoding="UTF-8", mode="w")
    for word, index in dict.items():
        file.write(str(word) + "\t" + str(index) + "\n")
    file.close()
    config.logger.info("Save dictionary finished.")


def save_dictionary(config):
    """
    :param config: config
    :return:
    """
    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)

        config.word_dict_path = "/".join([config.dict_directory, config.word_dict])
        config.label_dict_path = "/".join([config.dict_directory, config.label_dict])
        config.logger.info("word_dict_path : {}".format(config.word_dict_path))
        config.logger.info("label_dict_path : {}".format(config.label_dict_path))

        save_dict2file(config, config.create_alphabet.word_alphabet.words2id, config.word_dict_path)
        save_dict2file(config, config.create_alphabet.label_alphabet.words2id, config.label_dict_path)

        config.logger.info("copy dictionary to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory + "_bak"]))

def create_all_embeds_and_vocabs_v2(config, logger, embedding_file_path_pa, multi_lingual_embed_dict, ordered_domains):
    pos_set = list([paddingkey, nullkey])
    pred_set = list([paddingkey, nullkey, unkkey])
    arg_set = list([paddingkey, nullkey])

    word_set = list([paddingkey, unkkey])
    id2word_dict = dict({0: paddingkey, 1: unkkey})
    word2id_dict = dict({paddingkey: 0, unkkey: 1})

    for domain in ordered_domains:
        logger.info('loading for %s .' % domain)
        start_time = time.time()
        cur_word_set = list()
        cur_valid_word_set = list()

        input_filenames = get_file_path(config, domain)
        for input_filename in input_filenames:

            with open(input_filename, 'r', encoding='utf-8') as file:
                def _add_(coll, obj):
                    if obj not in coll:
                        coll.append(obj)

                ind = 0
                for row in file:
                    ind += 1
                    if ind == 1: continue  # columns, skip first line

                    line = row.strip()

                    if line == "": continue

                    tokens = line.split()

                    word_ = tokens[0]
                    _add_(cur_word_set, word_)

                    pos = tokens[1]
                    _add_(pos_set, pos)
                    pred = tokens[2]
                    _add_(pred_set, pred)

                    arg = tokens[3]
                    _add_(arg_set, arg)

        logger.info('cur_word_set: %d' % len(cur_word_set))
        for item in cur_word_set:
            if item not in word_set:
                id2word_dict[len(word_set)] = item
                word2id_dict[item] = len(word_set)
                word_set.append(item)
                cur_valid_word_set.append(item)
        logger.info('cur_valid_word_set: %d' % len(cur_valid_word_set))

    wordembeddings = Embeddings(config.embed_dim, word_set)
    posembeddings = Embeddings(config.pos_embed_dim, pos_set)
    predembeddings = Embeddings(config.prd_embed_dim, pred_set)
    argvocab = Vocab(arg_set)

    return wordembeddings,posembeddings, predembeddings, argvocab


def create_all_embeds_and_vocabs(config, logger, embedding_file_path_pa, multi_lingual_embed_dict, ordered_domains):
    pred_set = list([paddingkey, nullkey, unkkey])
    arg_set = list([paddingkey, nullkey])
    word_set = list([paddingkey, unkkey])
    id2word_dict = dict({0: paddingkey, 1: unkkey})
    word2id_dict = dict({paddingkey: 0, unkkey: 1})

    id2vec_embed_big = np.random.normal(size=[350000, config.embed_dim])

    total_ling_word_loading_cnt = 0
    for domain in ordered_domains:
        logger.info('loading for %s .' % domain)
        start_time = time.time()
        cur_word_set = list()
        cur_valid_word_set = list()

        input_filenames = get_file_path(config, domain)
        for input_filename in input_filenames:

            with open(input_filename, 'r', encoding='utf-8') as file:
                def _add_(coll, obj):
                    if obj not in coll:
                        coll.append(obj)

                ind = 0
                for row in file:
                    ind += 1
                    if ind == 1: continue  ## columns, skip first line

                    line = row.strip()

                    if line == "": continue

                    tokens = line.split()
                    word_ = tokens[0]
                    _add_(cur_word_set, word_)

                    pred = tokens[1]
                    _add_(pred_set, pred)

                    arg = tokens[-1]
                    _add_(arg_set, arg)

        logger.info('cur_word_set: %d' % len(cur_word_set))
        for item in cur_word_set:
            if item not in word_set:
                id2word_dict[len(word_set)] = item
                word2id_dict[item] = len(word_set)
                word_set.append(item)
                cur_valid_word_set.append(item)
        logger.info('cur_valid_word_set: %d' % len(cur_valid_word_set))

        cur_ling_word_loading_cnt = 0
        shortcut = multi_lingual_embed_dict[domain]
        embedding_file_path_path = os.path.join(embedding_file_path_pa, shortcut)
        with open(embedding_file_path_path, 'r', encoding='utf-8') as file:
            emb_lines = file.readlines()

            for line in tqdm.tqdm(emb_lines):

                line_ = line.split()
                try:
                    word = line_[0]
                    vec = line_[1:]

                    if len(vec) != config.embed_dim: continue
                    if word in cur_valid_word_set:
                        vec = np.asarray(vec)
                        id2vec_embed_big[word2id_dict[word]] = vec
                        cur_ling_word_loading_cnt += 1

                except:
                    logger.info('errors when loading embedding.')
                    logger.info(domain)
                    logger.info(embedding_file_path_path)
                    exit(1)

            total_ling_word_loading_cnt += cur_ling_word_loading_cnt
            logger.info('loading %d word from %s, total %d valid words in vocab.'
                        % (cur_ling_word_loading_cnt, shortcut, len(cur_valid_word_set)))
            end_time = time.time()
            logger.info('time spending %.3f min.' % ((end_time - start_time) / 60))
            logger.info('\n')

    logger.info('%d words from multi-lingual embeddings, total %d words in vocab.'
                % (total_ling_word_loading_cnt, len(id2word_dict)))

    id2vec_embed_big = id2vec_embed_big[:len(id2word_dict)]

    wordembeddings = Embeddings(config.embed_dim, id2word_dict, word2id_dict, id2vec_embed_big)
    predembeddings = Embeddings(config.prd_embed_dim, pred_set)

    argvocab = Vocab(arg_set)

    return wordembeddings, predembeddings, argvocab


def initial_those_embeds(config, wordembeddings,posembeddings, predembeddings, argvocab):
    word_set = wordembeddings.word_set
    pos_set = posembeddings.word_set
    pred_set = predembeddings.word_set
    arg_set = argvocab.word_vocab

    save_(word_set, os.path.join(config.dict_directory, config.word_dict))
    save_(pos_set, os.path.join(config.dict_directory, config.pos_dict))
    save_(pred_set, os.path.join(config.dict_directory, config.prd_dict))
    save_(arg_set, os.path.join(config.dict_directory, config.arg_dict))

    config.posembeddings = posembeddings
    config.predembeddings = predembeddings
    config.argvocab = argvocab

    config.embed_num = len(word_set)
    config.pos_embed_num = len(pos_set)
    config.prd_embed_num = len(pred_set)
    config.class_num = len(arg_set)

    config.word_paddingId = wordembeddings.pad_id
    config.pos_paddingId = posembeddings.pad_id
    config.prd_paddingId = predembeddings.pad_id
    config.arg_paddingId = argvocab.pad_id

    config.logger.info("embed_num : {}, prd_embed_num: {}, arg_num : {}".format(
        config.embed_num,
        config.prd_embed_num,
        config.class_num))

    config.logger.info('initalized with all embeddings.')


def save_(list_, file_):
    with open(file_, 'w') as f:
        f.write('\n'.join(list_))


def get_file_path(config, s_domain):
    training_data = config.base_data_path + '/%s/train' % s_domain
    eval_data = config.base_data_path + '/%s/dev' % s_domain
    test_data = config.base_data_path + '/%s/test' % s_domain
    for item in os.listdir(training_data.replace('/train', '')):
        if '-dev-pos.conllu' in item:
            eval_data = eval_data.replace('dev', item)
        if '-train-pos.conllu' in item:
            training_data = training_data.replace('train', item)
        if '-test-pos.conllu' in item:
            test_data = test_data.replace('test', item)
    return training_data, eval_data, test_data


def prepare_data(config, domains):
    """
    :param config: config
    :return:
    """
    config.logger.info("Processing All Data......")

    base_file_name = 'wiki.%s.align.vec'
    multi_lingual_embed_dict = {
        'UP_English': base_file_name % 'en',
        'UP_Finnish': base_file_name % 'fi',
        'UP_French': base_file_name % 'fr',
        'UP_German': base_file_name % 'de',
        'UP_Italian': base_file_name % 'it',
        'UP_Portuguese-Bosque': base_file_name % 'pt',
        'UP_Spanish': base_file_name % 'es',
        'UP_Spanish-AnCora': base_file_name % 'es',
        'UP_Chinese': base_file_name % 'zh',
    }

    if os.path.exists(os.path.join(config.pkl_directory, config.loaded_word_embed)) and \
            os.path.exists(os.path.join(config.pkl_directory, config.other_data_embed)):
        wordembeddings = torch.load(f=os.path.join(config.pkl_directory, config.loaded_word_embed))
        other_data_embed = torch.load(f=os.path.join(config.pkl_directory, config.other_data_embed))
        (posembeddings, predembeddings, argvocab) = other_data_embed

    else:
        if not os.path.exists(config.pkl_directory):
            os.mkdir(config.pkl_directory)

        wordembeddings, posembeddings, predembeddings, argvocab = \
            create_all_embeds_and_vocabs_v2(config, config.logger, config.pretrained_embed_file, multi_lingual_embed_dict,
                                         domains)

        other_data_embed = (posembeddings, predembeddings, argvocab)
        torch.save(obj=wordembeddings, f=os.path.join(config.pkl_directory, config.loaded_word_embed))
        torch.save(obj=other_data_embed, f=os.path.join(config.pkl_directory, config.other_data_embed))

    initial_those_embeds(config, wordembeddings,posembeddings, predembeddings, argvocab)


def load_model(config, lang2id):
    """
    :param config:  config
    :return:  nn model
    """
    config.logger.info("***************************************")
    model = SequenceLabeler(config, lang2id)

    if config.device != cpu_device:
        model = model.cuda()

    if config.mode == 'test':
        model = load_test_model(model, config)
    else:
        if os.path.exists(config.save_model_dir):
            shutil.rmtree(config.save_model_dir)

    config.logger.info(model)
    return model


def pre_embed(config, alphabet):
    """
    :param config: config
    :param alphabet:  alphabet dict
    :return:  pre-train embed
    """
    config.logger.info("***************************************")
    pretrain_embed = None
    embed_types = ""
    if config.pretrained_embed and config.zeros:
        embed_types = "zero"
    elif config.pretrained_embed and config.avg:
        embed_types = "avg"
    elif config.pretrained_embed and config.uniform:
        embed_types = "uniform"
    elif config.pretrained_embed and config.nnembed:
        embed_types = "nn"

    if config.pretrained_embed is True:
        p = Embed(path=config.pretrained_embed_file, words_dict=alphabet.word_alphabet.id2words, embed_type=embed_types,
                  pad=paddingkey)
        pretrain_embed = p.get_embed()

        embed_dict = {"pretrain_embed": pretrain_embed}
        torch.save(obj=embed_dict, f=os.path.join(config.pkl_directory, config.pkl_embed))

    return pretrain_embed
