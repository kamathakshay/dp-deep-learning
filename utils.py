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
                        help='directory to save the checkpoints (default: ''./saved_models/'')')
    parser.add_argument('-m', '--model_name', default='cifar10cnn', type=str, dest='model_name',
                        help='model name (default: ''cifar10cnn'')')
    parser.add_argument('-d', '--data', metavar='DATA', default='mnist',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: mnist)')
    args = parser.parse_args()
    return args

def parse_visualization_command():
    import argparse
    parser = argparse.ArgumentParser(description='Deep Learning with Differential Privacy')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, dest='save_dir',
                        help='directory where the checkpoints are saved (default: ''./saved_models/'')')
    parser.add_argument('--model_name', dest='model_name', type=str, default='',
                        help='model name. If not specified, read the first model in directory.')
    args = parser.parse_args()
    return args