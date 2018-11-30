import os
import json
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='Deep Learning with Differential Privacy')
parser.add_argument('--save_dir', default='./results_cifar/', type=str, dest='save_dir',
                    help='directory where the checkpoints are saved (default: ''./results_cifar/'')')
parser.add_argument('--model_name', dest='model_name', type=str, default='',
                    help='model name. If not specified, read the first model in directory.')
args = parser.parse_args()

model_dir = ''
if args.model_name:
    model_dir = os.path.join(args.save_dir, args.model_name)
else:
    dirs = os.listdir(args.save_dir)
    dirs.reverse()
    for f in dirs:
        if f.startswith('m_'):
            model_dir = f
            break
    model_dir = os.path.join(args.save_dir, model_dir)
if not model_dir:
    raise ValueError('Model not found in '+args.save_dir)

history_dir = os.path.join(model_dir, 'results-0.json')
plot_dir = os.path.join(model_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

print('Load model: '+model_dir)

model_name = model_dir.split('/')[-1][17:]

def plot_training(history_dir, plot_dir):
    with open(history_dir) as f:
        history = json.load(f)
    learning_rate = history['learning_rate']
    results = history['result_series']
    steps = []
    train_acc = []
    test_acc = []
    for r in results:
        steps.append(r['step'])
        train_acc.append(r['train_accuracy'])
        test_acc.append(r['test_accuracy'])

    plt.plot(steps, train_acc, 'b-', label='Training')
    plt.plot(steps, test_acc, 'r-', label='Validation')
    plt.title('Accuracy for '+model_name)
    plt.xlabel('step')
    plt.xlabel('accuracy')
    plt.legend()

    plt.savefig(os.path.join(plot_dir, 'acc.png'))
    print('Plot saved at '+plot_dir)
    plt.close()

def main():
    plot_training(history_dir=history_dir, plot_dir=plot_dir)


if __name__ == '__main__':
    main()
