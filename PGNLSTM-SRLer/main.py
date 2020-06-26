import argparse
import datetime
import Config.config as configurable

from DataUtils.mainHelp import *
from DataUtils.Alphabet import *
from inference import load_test_data
from inference import Inference
from train import Train
import random
import logging

# solve default encoding problem
from imp import reload
import sys, os

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

# random seed
torch.manual_seed(seed_num)
random.seed(seed_num)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def start_train(train_iter, dev_iter, test_iter, model, config):
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()
    config.logger.info("Finish Train.")
    return t


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    config.logger.info("\nTesting Start......")
    data, path_source, path_result = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = Inference(model=model, data=data, path_source=path_source, path_result=path_result, alphabet=alphabet,
                      use_crf=config.use_crf, config=config)
    infer.infer2file()
    config.logger.info("Finished Test.")


def load_vocab_elmo(elmo_vocab_path):
    with open(elmo_vocab_path, 'r') as f:
        elmo_vocab = [item.strip() for item in f.readlines()]
    return elmo_vocab


def main():

    if not os.path.exists(config.save_model_dir): os.mkdir(config.save_model_dir)

    s_domains = [
        'UP_English',
        'UP_German',
        'UP_French',
        'UP_Finnish',
        'UP_Italian',
        'UP_Portuguese',
        'UP_Spanish',
    ]
    lang2id = {k: i for i, k in enumerate(s_domains)}

    t_domain = s_domains[1]

    config.elmo_vocab = load_vocab_elmo(config.elmo_vocab_path)

    prepare_data(config, s_domains)

    config.learning_algorithm = get_learning_algorithm(config)

    model = load_model(config, lang2id)


    if config.mode == 'train':
        # loading digitalized data for English
        config.logger.info('loading for [%s] .' % ','.join(s_domains))

        training_paths, eval_paths, test_paths = {}, {}, {}
        for s_domain in s_domains:
            training_path, eval_path, test_path = get_file_path(config, s_domain)
            training_paths[s_domain] = training_path
            eval_paths[s_domain] = eval_path
            test_paths[s_domain] = test_path

        data_loader = DataLoader(domains=s_domains, path=[training_paths, eval_paths, test_paths], shuffle=True,
                                 config=config)
        train_data, dev_data, test_data = data_loader.load()

        config.logger.info("train sentence {}, dev sentence {}, test sentence {}.".
                           format(len(train_data), len(dev_data), len(test_data)))

        create_iter = Iterators(domains=s_domains,lang2id=lang2id,
                                batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                                data=[train_data, dev_data, test_data], device=config.device, config=config)

        train_iter = create_iter.createSetIterator(0)
        dev_iter = create_iter.createSetIterator(1,T_domain=t_domain,isTrain=False)
        test_iter = create_iter.createSetIterator(2,T_domain=t_domain,isTrain=False)


        trainer = Train(target=t_domain,train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
        trainer.train()
        config.logger.info("Finish Train.")

        multi_lingual_eval(trainer, [t_domain], model, lang2id)



def multi_lingual_eval(trainer, ordered_domains, model,lang2id):
    config.logger.info('\n\n')
    config.logger.info('=-' * 50)

    for domain in ordered_domains:
        config.logger.info('evaluate for %s .' % domain)

        training_path, eval_path, test_path = get_file_path(config, domain)
        data_loader = DataLoader(domains=[domain], path=[training_path, eval_path, test_path], shuffle=True,
                                 config=config)
        train_data, dev_data, test_data = data_loader.load_single(domain)
        config.logger.info('done for loading data.')

        config.logger.info("train sentence {}, dev sentence {}, test sentence {}.".
                           format(len(train_data), len(dev_data), len(test_data)))

        create_iter = Iterators(domains=[domain],lang2id=lang2id,
                                batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                                data=[train_data, dev_data, test_data], device=config.device, config=config)

        train_iter = create_iter.createSetIterator(0,T_domain=domain,isTrain=False)

        config.logger.info("evaluation on dev on %s..." % domain)
        config.logger.info("evaluation on train on %s..." % domain)
        trainer.eval_external_batch(train_iter, config, domain)

        config.logger.info("evaluation on dev on %s..." % domain)
        dev_iter = create_iter.createSetIterator(1,T_domain=domain,isTrain=False)
        trainer.eval_external_batch(dev_iter, config, domain)

        config.logger.info("evaluation on test on %s..." % domain)
        test_iter = create_iter.createSetIterator(2,T_domain=domain,isTrain=False)
        trainer.eval_external_batch(test_iter, config, domain)

        config.logger.info('=-' * 50)


def get_logger(log_dir):
    log_file = log_dir  # + "/" + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.log'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(message)s')

    # log into file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # log into terminal
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger


def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="multi-SRL")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    args = parser.parse_args()
    config = configurable.Configurable(config_file=args.config_file)

    # save file
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    subset_name_dir = os.path.join(config.save_checkpoint)
    if not os.path.isdir(subset_name_dir): os.makedirs(subset_name_dir)

    config.save_dir = os.path.join(subset_name_dir, config.mulu + '.log')

    logger = get_logger(config.save_dir)
    config.logger = logger

    return config


def set_cuda():
    config.logger.info("\nUsing GPU To Train......")
    device_number = config.device[-1]
    print(device_number)
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
