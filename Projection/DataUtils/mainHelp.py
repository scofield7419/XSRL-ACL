import shutil
import time
# solve default encoding problem
from imp import reload

from DataUtils.Alphabet import *
from DataUtils.BatchIterator import *
from DataUtils.DataLoader import DataLoader
from DataUtils.Embed import Embed
from Engines.SequenceLabeler import SequenceLabeler

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

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


def get_params(config, alphabet):
    """
    :param config: config
    :param alphabet:  alphabet dict
    :return:
    """
    config.learning_algorithm = get_learning_algorithm(config)

    # get params
    config.embed_num = alphabet.word_alphabet.vocab_size
    config.class_num = alphabet.label_alphabet.vocab_size
    config.word_paddingId = alphabet.word_paddingId
    config.label_paddingId = alphabet.label_paddingId
    config.create_alphabet = alphabet

    config.logger.info("embed_num : {}, class_num : {}".format(config.embed_num, config.class_num))
    config.logger.info("word PaddingID {}".format(config.word_paddingId))
    config.logger.info("label PaddingID {}".format(config.label_paddingId))


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
        # config.logger.info(word, index)
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
        # copy to mulu
        config.logger.info("copy dictionary to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory + "_bak"]))


# load data / create alphabet / create iterator
def preprocessing(config):
    """
    :param config: config
    :return:
    """
    config.logger.info("Processing Data......")
    # read file
    data_loader = DataLoader(path=[config.train_file, config.dev_file, config.test_file], shuffle=True, config=config)
    train_data, dev_data, test_data = data_loader.dataLoader()
    config.logger.info(
        "train sentence {}, dev sentence {}, test sentence {}.".format(len(train_data), len(dev_data), len(test_data)))
    config.train_cnt = len(train_data)
    config.dev_cnt = len(dev_data)
    config.test_cnt = len(test_data)
    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}
    if config.save_pkl:
        torch.save(obj=data_dict, f=os.path.join(config.pkl_directory, config.pkl_data))

    # create the alphabet
    alphabet = CreateAlphabet(min_freq=config.min_freq, train_data=train_data, dev_data=dev_data,
                                  test_data=test_data, config=config)
    alphabet.build_vocab()

    alphabet_dict = {"alphabet": alphabet}
    if config.save_pkl:
        torch.save(obj=alphabet_dict, f=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # create iterator
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_data, dev_data, test_data], operator=alphabet, device=config.device,
                            config=config)
    train_iter, dev_iter, test_iter = create_iter.createIterator()
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    if config.save_pkl:
        torch.save(obj=iter_dict, f=os.path.join(config.pkl_directory, config.pkl_iter))
    return train_iter, dev_iter, test_iter, alphabet


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



def load_test_model(model, config):

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


def load_model(config):
    """
    :param config:  config
    :return:  nn model
    """
    config.logger.info("***************************************")
    model = SequenceLabeler(config)
    if config.device != cpu_device:
        model = model.cuda()
    if config.mode == 'test':
        model = load_test_model(model, config)
    else:
        if os.path.exists(config.save_model_dir):
            shutil.rmtree(config.save_model_dir)
    config.logger.info(model)
    return model


def load_data(config):
    """
    :param config:  config
    :return: batch data iterator and alphabet
    """
    config.logger.info("load data for original data file or pkl file.")
    train_iter, dev_iter, test_iter = None, None, None
    alphabet = None
    start_time = time.time()
    if (config.mode == 'train') and (config.restore_pkl is False):
        config.logger.info("process data")
        if config.save_pkl:
            if os.path.exists(config.pkl_directory): shutil.rmtree(config.pkl_directory)
            if not os.path.isdir(config.pkl_directory): os.makedirs(config.pkl_directory)
        train_iter, dev_iter, test_iter, alphabet = preprocessing(config)
        config.pretrained_weight = pre_embed(config=config, alphabet=alphabet)
    elif ((config.mode == 'train') and (config.restore_pkl is True)) or (config.mode == 'test'):
        config.logger.info("load data from pkl file")
        # load alphabet from pkl
        alphabet_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_alphabet))
        config.logger.info(alphabet_dict.keys())
        alphabet = alphabet_dict["alphabet"]
        # load iter from pkl
        iter_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_iter))
        config.logger.info(iter_dict.keys())
        train_iter, dev_iter, test_iter = iter_dict.values()
        # load embed from pkl
        config.pretrained_weight = None
        if os.path.exists(os.path.join(config.pkl_directory, config.pkl_embed)):
            embed_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_embed))
            config.logger.info(embed_dict.keys())
            embed = embed_dict["pretrain_embed"]
            config.pretrained_weight = embed
    end_time = time.time()
    config.logger.info("All Data/Alphabet/Iterator Use Time {:.4f}".format(end_time - start_time))
    config.logger.info("***************************************")
    return train_iter, dev_iter, test_iter, alphabet
