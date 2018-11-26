import os
import pickle
import json
from keras.models import load_model
import matplotlib.pyplot as plt


import utils


import argparse
parser = argparse.ArgumentParser(description='Deep Learning with Differential Privacy')
parser.add_argument('--save_dir', default='./results/', type=str, dest='save_dir',
                    help='directory where the checkpoints are saved (default: ''./results/'')')
parser.add_argument('--model_name', dest='model_name', type=str, default='',
                    help='model name. If not specified, read the first model in directory.')
args = parser.parse_args()

model_name = 'model'
model_dir = os.path.join(args.save_dir, 'checkpoint')
history_dir = os.path.join(args.save_dir, 'results-0.json')
plot_dir = os.path.join(args.save_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

print('Load model: '+model_dir)

def plot_training():
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
    plt.title('Accuracy for ' + model_name + "(LR: " + str(learning_rate) + " )")
    plt.xlabel('step')
    plt.xlabel('accuracy')
    plt.legend()

    plt.savefig(os.path.join(plot_dir, 'acc.png'))
    plt.close()

def main():
    plot_training()


if __name__ == '__main__':
    main()
