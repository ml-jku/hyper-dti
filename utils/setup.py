import os
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def setup(seed, name):
    """ Setup random seeds, cuda and checkpoint directory. """

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    # prepare file structures
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + name):
        os.makedirs('checkpoints/' + name)
    if not os.path.exists('checkpoints/' + name + '/' + 'models'):
        os.makedirs('checkpoints/' + name + '/' + 'models')
    if not os.path.exists('checkpoints/' + name + '/' + 'models' + '/' + 'continuous'):
        os.makedirs('checkpoints/' + name + '/' + 'models' + '/' + 'continuous')

