import os
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt

import utils

args = utils.parse_visualization_command()
#model_dir = load_model('./saved_models/checkpoints/m_2018-11-23_22:10_cifar10cnn_lr0.0001_tl.hdf5')
if args.model_name:
    model_dir = os.path.join(args.save_dir, args.model_name)
else:
    for f in os.listdir(args.save_dir):
        if f.startswith('m_'):
            model_dir = f
            break
    model_dir = os.path.join(args.save_dir, model_dir)

plot_dir = os.path.join(model_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

print('Load model: '+model_dir)

model_name = model_dir.split('_')[2]
learning_rate = model_dir.split('_')[-1][2:]


def plot_training():
    with open(model_dir+'/training_history', 'rb') as pickle_file:
        history = pickle.load(pickle_file)
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b-', label='Training')
    plt.plot(epochs, val_acc, 'r-', label='Validation')
    plt.title('Accuracy for ' + model_name + "(LR: " + str(learning_rate) + " )")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, 'acc.png'))
    plt.close()

    plt.figure()

    plt.plot(epochs, loss, 'b-', label='Training')
    plt.plot(epochs, val_loss, 'r-', label='Validation')
    plt.title('Loss for ' + model_name + "(LR: " + str(learning_rate) + " )")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.close()

def plot_activation():
    from vis.visualization import visualize_activation
    from vis.utils import utils
    from keras import activations

    model = load_model(os.path.join(model_dir, 'checkpoint.hdf5'))
    plt.rcParams['figure.figsize'] = (18, 6)

    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, 'final_preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    # This is the output node we want to maximize.
    filter_idx = 0
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
    plt.imshow(img[..., 0])
    plt.savefig(os.path.join(plot_dir, 'activation.png'))
    #plt.show()


def main():
    #plot_training()
    plot_activation()


if __name__ == '__main__':
    main()
