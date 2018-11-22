import os

def parse_command():
    data_names = ['cifar10', 'mnist']

    import argparse
    parser = argparse.ArgumentParser(description='Deep Learning with Differential Privacy')
    parser.add_argument('--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, dest='learning_rate',
                        help='initial learning rate (default 0.0001)')
    parser.add_argument('-e', '--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-b', '--batch_size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('-s', '--steps_per_epoch', default=None, type=int,
                        help='Steps to train per epoch (default: None)')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, dest='save_dir',
                        help='directory to save the checkpoints (default: ''./checkpoints/'')')
    parser.add_argument('-m', '--model_name', default='cifar10cnn', type=str, dest='model_name',
                        help='model name (default: ''cifar10cnn'')')
    parser.add_argument('-d', '--data', metavar='DATA', default='cifar10',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: cifar10)')
    args = parser.parse_args()
    return args