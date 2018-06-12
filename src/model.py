# code for selecting model to use

import sys
import resNet
# TODO
#import vgg
#import alexNet

def get_model(hps, xs, ys, learning_rate, trainable):
    """ get model for train or evaluation

    Args
        hps: hyperparameters
        xs: placeholder
        ys: placeholder
        learning_rate: placeholder
        trainable: placeholder
    Returns:
        model (Object)
    """

    if hps.model == 'resNet':
        resNet.ResNet(hps, xs, ys, learning_rate, trainable)
    elif hps.model == 'vgg':
        #TODO
        pass
    elif hps.model == 'alexnet':
        #TODO
        pass
    else:
        print('No matched model provided')
        sys.exit(-1)


