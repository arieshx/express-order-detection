from .VGGnet_test import VGGnet_test
from .VGGnet_train import VGGnet_train
from .Densenet_test import densenet_test
from .Densenet_train import  densenet_train


def get_network(name, training_flag = None):
    """Get a network by name."""
    if name.split('_')[0] == 'VGGnet':
        if name.split('_')[1] == 'test':
           return VGGnet_test()
        elif name.split('_')[1] == 'train':
           return VGGnet_train()
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    elif name.split('_')[0] == 'Densenet':
        if name.split('_')[1] == 'test':
            return densenet_test(training_flag)
        elif name.split('_')[1] == 'train':
           return densenet_train(training_flag)
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))
