import os
from yacs.config import CfgNode as CN

#-----------------------------------------------
#Config definition
#-----------------------------------------------
_C = CN()

#Model
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "CNNModel"
_C.MODEL.DH = 2
_C.MODEL.DW = 2
_C.MODEL.N_MUL = 5
_C.MODEL.DD = 36
_C.MODEL.N_PER_GROUP = 1

_C.MODEL.N_MUL_LIST = [1, 4]
_C.MODEL.N_PER_LIST = [1, 1]
_C.MODEL.DH_LIST = [3, 3]

_C.MODEL.NUM_TREES = 5

#Model FC
_C.MODEL.FC = CN()
#FC units number
_C.MODEL.FC.N_FC = 128
#dense or one_fc
_C.MODEL.FC.B_FC = True

#whether to add sigmoid after network
_C.MODEL.SIGMOID = False

#whether to use dynamic convolution
# https://arxiv.org/pdf/1908.05867.pdf
_C.MODEL.DGCONV = False

#used for resume from checkpoint
_C.MODEL.RESUME = ""
#only load weights
_C.MODEL.WEIGHT = ""

#DataSet
_C.DATASETS = CN()
_C.DATASETS.NAME = 'cub200'
_C.DATASETS.CLASS = 200

#Input
_C.INPUT = CN()
_C.INPUT.SIZE = 224

#Solver
_C.SOLVER = CN()
_C.SOLVER.NUM_ROUNDS = 5
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.TRAIN_PER_BATCH = 128
_C.SOLVER.TEST_PER_BATCH = 128
_C.SOLVER.LR_SCHEDULER_ON = False
_C.SOLVER.LR_SCHEDULER_GAMMA = 0.1
_C.SOLVER.LR_SCHEDULER_MILESTONE = [100, 150]

_C.SOLVER.OPTIMIZER = 'SGD'

_C.SOLVER.LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 2e-4

#used for two stage finetune, True then train new layers first, then train all layers
_C.SOLVER.FREEZE = False

_C.TRAIN = True

#Misc options
_C.OUTPUT_DIR = "."
_C.LOG_EPOCHS = 1
