import os
import sys

import nni
import torch
import warnings

BASE_DIR = os.path.dirname(os.path.abspath('.'))

sys.path.append(BASE_DIR)

from utils.logger import *

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
torch.set_num_threads(8)
info(torch.get_num_threads())
info(torch.__config__.parallel_info())



torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

os.environ['DATA_ABS_PATH'] = BASE_DIR + '/data/processed'

import argparse


def init_param():
    parser = argparse.ArgumentParser(description='PyTorch Experiment')
    parser.add_argument('--name', type=str, default='wine_red',
                        help='data name')
    parser.add_argument('--log-level', type=str, default='info', help=
    'log level, check the utils.logger')
    parser.add_argument('--episodes', type=int, default=10, help=
    'episodes for training')
    parser.add_argument('--steps', type=int, default=10, help=
    'steps for each episode')
    parser.add_argument('--enlarge_num', type=int, default=4, help=
    'feature space enlarge')
    parser.add_argument('--cuda', type=str, default='cpu')
    parser.add_argument('--memory', type=int, default=8, help='memory capacity')
    parser.add_argument('--a', type=float, default=1, help='a')
    parser.add_argument('--b', type=float, default=1, help='b')
    parser.add_argument('--c', type=float, default=1, help='c')
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--state-method', type=str, default='ae',
                        help='reinforcement state representation method')
    parser.add_argument('--replay-strategy', type=str, default='random')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--init-w', type=float, default=1e-6)
    parser.add_argument('--ablation-mode', type=str, default='')
    parser.add_argument('--distance', type=str, default='mi')
    parser.add_argument('--id', type=str, default='test')
    args, _ = parser.parse_known_args()
    return args
