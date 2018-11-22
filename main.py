from __future__ import print_function
import os
import keras
import datetime
from keras.datasets import mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

from models import *

import utils

args = utils.parse_command()


def load_data(data_type, num_classes=10):
    if data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # print(x_train)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


def get_optimizer(lr, decay=1e-6):
    opt = keras.optimizers.rmsprop(lr=lr, decay=decay)
    return opt


def train(model, x_train, y_train, x_test, y_test):
    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    datenow = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
    logs_dir = args.save_dir + '/logs/'
    checkpoint_dir = args.save_dir + '/checkpoints/'
    tensorboard_dir = args.save_dir + '/tb_logs/'
    for p in [args.save_dir, logs_dir, checkpoint_dir, tensorboard_dir]:
        if not os.path.isdir(p):
            os.makedirs(p)

    output_base = "m_" + datenow + "_" + args.model_name \
                  + "_lr" + str(args.learning_rate)
    logger_output_tl = output_base + "_tl.log"

    csv_logger_tl = CSVLogger(logs_dir + logger_output_tl, append=True)
    weights_tl = output_base + "_tl.hdf5"
    checkpointer_tl = ModelCheckpoint(filepath=checkpoint_dir + weights_tl, verbose=1,
                                      monitor='val_loss', save_best_only=True, mode='min')
    tensorboard_dir = tensorboard_dir + output_base + "_tl"
    tensorboard_tl = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_images=True)

    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=args.batch_size),
                                  epochs=args.epochs,
                                  validation_data=(x_test, y_test),
                                  workers=4,
                                  steps_per_epoch=x_train.shape[0] / args.batch_size,
                                  callbacks=[csv_logger_tl, checkpointer_tl, tensorboard_tl])
    print(history.history.keys())
    return model


def main():
    x_train, y_train, x_test, y_test = load_data(data_type=args.data)
    if args.data.lower() == 'cifar10':
        model = build_model_cifar(input_shape=x_train.shape[1:])
    elif args.data.lower() == 'mnist':
        model = build_model_mnist(input_shape=(28, 28, 1))
    else:
        print('Model %s does not exist. ' % args.model)

    if args.evaluate:
        # Score trained model.
        #scores = model.evaluate(x_test, y_test, verbose=1)
        #print('Test loss:', scores[0])
        #print('Test accuracy:', scores[1])
        pass

    else:
        opt = get_optimizer(args.learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model = train(model, x_train, y_train, x_test, y_test)

        print('Finish training.')


if __name__ == '__main__':
    main()
