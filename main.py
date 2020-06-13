'''
Created on 2018.03.05

@author: caoyh
'''
import os
import torch
import torchvision
import numpy as np
from Model import *
from DataLoader import perm_mask
from trainer import train
from tester import test
from utils.config import cfg
from utils.logger import setup_logger
from utils.loading import *
import argparse

os.environ['OMP_NUM_THREADS'] = "10"
os.environ['MKL_NUM_THREADS'] = "10"


def get_model(cfg):
    '''
    param cfg:
    return model meta_architecture
    '''

    '''
    UciNet, UciCatNet, CascadeNet, CascadeCatNet, Forest, ForestSelect
    '''

    model_type = cfg.MODEL.META_ARCHITECTURE

    #ResNet50+NRS
    if model_type.startswith('resnet') or model_type.startswith('vgg') or model_type.startswith('mobile'):
        model = eval(model_type)(cfg, pretrained=True)
    elif model_type.startswith('Uci'):
        model = UciNRSNet(cfg)
        if 'FC' in model_type:
            model = UciFCNet(cfg)
    else:
        raise NotImplementedError("META_ARCHITECTURE is not implemented")

    return model

def main():
    parser = argparse.ArgumentParser(description="PyTorch NRS Experiment")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    print(args.opts)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'nPer_{}'.format(cfg.MODEL.N_PER_GROUP))
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    print(output_dir)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("NRS", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n"+cf.read()
        logger.info(config_str)

    logger.info("Runnig with config:\n{}".format(cfg))

    if cfg.TRAIN:
        logger.info("Start training for {} rounds".format(cfg.SOLVER.NUM_ROUNDS))
        accs = []
        best_accs = []
        for i in range(cfg.SOLVER.NUM_ROUNDS):
            model = get_model(cfg)
            _, acc, best_acc = train(model, cfg, logger)
            #_, acc, best_acc = mixup_train(model, cfg, logger)
            accs.append(acc)
            best_accs.append(best_acc)
            print(acc, best_acc)
            logger.info("Round {} finish, acc is {}, best_acc is {}".format(i+1, acc, best_acc))

        logger.info(accs)
        logger.info(best_accs)
        logger.info("Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(accs), np.std(accs)))
        logger.info("Mean(acc)={:.2f}%, Std(acc)={:.2f}%".format(np.mean(best_accs), np.std(best_accs)))
    else:
        model = get_model(cfg)
        test(model, cfg, logger)

if __name__=="__main__":
    main()
