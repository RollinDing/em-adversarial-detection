import argparse

def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='Detect adversarial samples using EM side channel signal')
    
    parser.add_argument('--trace-path', 
                        type=str, 
                        help='Directory of EM traces.')

    parser.add_argument('--attack-method', 
                        choices=['org', 'train', 'pgd', 'cw', 'targetcw', 'fgm'], 
                        type=str,
                        help='choose the adverserial dataset with certain adversarial attack method, org means the original dataset, train means using the training dataset for analyze.')
    
    parser.add_argument('--segment-id', 
                        default=0, 
                        type=int,
                        help='Select the batches to analyze from [0 -27]')
    
    parser.add_argument('--dataset', 
                        default='fm', 
                        choices=['fm', 'cifar10'], 
                        type=str,
                        help="The dataset of both original cnn model and adversarial attack")

    parser.add_argument('--victim-model', 
                        default='lenet', 
                        choices=['lenet', 'vgg'], 
                        type=str,
                        help="the victim model type")

    parser.add_argument('--rate', 
                        default=1, 
                        type=float,
                        help='the downsample rate of preprocessing when analyzing the raw trace')

    parser.add_argument('--date', 
                        type=int,
                        help='the date when the dataset is collected.')

    parser.add_argument('--channels', 
                        default=22, 
                        type=int, 
                        help='the number of channel that the detector will persever for further classification, 22 by default')

    parser.add_argument('--winsize', 
                        default=256, 
                        type=int,
                        help='the size of short term fourier transform window')

    parser.add_argument('--target', 
                        default=None, 
                        type=int,
                        help='the target class to train the derived model')

    parser.add_argument('--gradcam', 
                        default=False, 
                        type=bool,
                        help='using GradCAM to record the gradient of EM classifiers')
    
    parser.add_argument('--segment-num',
                        default=6,
                        type=int,
                        help='the number of segments to analyze')

    args = parser.parse_args()

    return args
