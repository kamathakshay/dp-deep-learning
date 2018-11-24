from __future__ import print_function
import os
import keras
import pickle
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
    save_dir = os.path.join(args.save_dir, "m_" + datenow + "_" + args.data
                            + "_lr" + str(args.learning_rate))
    for p in [args.save_dir, save_dir]:
        if not os.path.isdir(p):
            os.mkdir(p)

    csv_logger_tl = CSVLogger(os.path.join(save_dir, 'csv_logger.log'), append=True)
    checkpointer_tl = ModelCheckpoint(filepath=os.path.join(save_dir, 'checkpoint.hdf5'), verbose=1,
                                      monitor='val_loss', save_best_only=True, mode='min')
    tensorboard_tl = TensorBoard(log_dir=os.path.join(save_dir, 'tb_logs/'), histogram_freq=0, write_images=True)

    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=args.batch_size),
                                  epochs=args.epochs,
                                  validation_data=(x_test, y_test),
                                  workers=4,
                                  steps_per_epoch=x_train.shape[0] / args.batch_size,
                                  callbacks=[csv_logger_tl, checkpointer_tl, tensorboard_tl])
    with open(os.path.join(save_dir, 'training_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
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
