from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time, os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras import backend as K

from sklearn import decomposition

from keras.applications import MobileNetV2
import matplotlib.pyplot as plt


# ======================================================================================================================
def make_model_MLP1(num_feature, nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape=(num_feature,)))
    model.add(Dense(1, activation='tanh'))
    # model.add(Activation('softmax'))
    # print model structure
    model.summary()
    return model


def make_model_MLP2(num_feature, nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape=(num_feature,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='tanh'))
    # print model structure
    model.summary()
    return model


def make_model_CNN1(input_shape, nb_classes):
    model_2 = Sequential()
    model_2.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))
    model_2.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))
    model_2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))
    # model_2.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    # model_2.add(MaxPooling2D(pool_size=(2,2)))
    model_2.add(Dropout(0.25))
    model_2.add(Flatten())
    # model_2.add(Dense(128, activation='relu'))
    # model_2.add(Dropout(0.5))
    model_2.add(Dense(nb_classes, activation='softmax'))
    model_2.summary()
    return model_2


def make_model_CNN2(input_shape, nb_classes):
    ###################
    # Model
    #####################
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    # x = keras.layers.GlobalAveragePooling2D(base_model.input)
    flatten = keras.layers.Flatten()
    x = flatten(base_model.output)
    # class1 = Dense(100, activation='relu', name='dense_feature')(model_base.output)
    output = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=output)
    # freeze some layers
    # for layer in base_model.layers:
    #    layer.trainable = False
    model.summary()
    return model


class A2():
    def __init__(self, model, task, image_shape):
        super(A2, self).__init__()
        self.task = task
        self.num_feature = image_shape[0] * image_shape[1] * 3
        self.input_shape = (image_shape[0], image_shape[1], 3)
        print(self.input_shape)
        if self.task == 0:
            self.nb_classes = 2
        elif self.task == 1:
            self.nb_classes = 2
        elif self.task == 2:
            self.nb_classes = 5
        elif self.task == 3:
            self.nb_classes = 5
        self.model_name = model

        if self.model_name == 'MLP1':
            self.model = make_model_MLP1(self.num_feature, self.nb_classes)
        elif self.model_name == 'MLP2':
            self.model = make_model_MLP2(self.num_feature, self.nb_classes)
        elif self.model_name == 'CNN1':
            self.model = make_model_CNN1(self.input_shape, self.nb_classes)
        elif self.model_name == 'CNN2':
            self.model = make_model_CNN2(self.input_shape, self.nb_classes)

    def train(self, train_generator, val_generator, lr, epochs, STEP_SIZE_TRAIN, STEP_SIZE_VALID):
        if self.model_name == 'MLP1':
            print("======================logic regression training starts:======================\n")
            t1 = time.time()
            self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=Adam(lr=lr),
                               metrics=['accuracy'])
            self.history = self.model.fit_generator(train_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=val_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=epochs)
            t2 = time.time()
            self.plotting('MLP1')
            score_train = self.model.evaluate_generator(generator=train_generator, steps=STEP_SIZE_TRAIN)
            score_val = self.model.evaluate_generator(generator=val_generator, steps=STEP_SIZE_VALID)
            print('MLP1 finished with time costing:', t2 - t1)
            print('MLP1 Trained Accuracy: ', score_train[1])
            print('MLP1 Validation Accuracy: ', score_val[1])

            acc_train = score_train[1]
        elif self.model_name == 'MLP2':
            print("======================MLP2 training starts:======================\n")
            t1 = time.time()
            self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=Adam(lr=lr),
                               metrics=['accuracy'])
            self.history = self.model.fit_generator(train_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=val_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=epochs)
            t2 = time.time()
            self.plotting('MLP2')
            score_train = self.model.evaluate_generator(generator=train_generator, steps=STEP_SIZE_TRAIN)
            score_val = self.model.evaluate_generator(generator=val_generator, steps=STEP_SIZE_VALID)
            print('MLP2 finished with time costing:', t2 - t1)
            print('MLP2 Trained Accuracy: ', score_train[1])
            print('MLP2 Validation Accuracy: ', score_val[1])

            acc_train = score_train[1]
        elif self.model_name == 'CNN1':
            print("======================CNN1 training starts:======================\n")
            t1 = time.time()
            self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=Adam(lr=lr),
                               metrics=['accuracy'])
            print('optimizer compile done')
            self.history = self.model.fit_generator(train_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=val_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=epochs)
            t2 = time.time()
            self.plotting('CNN1')
            score_train = self.model.evaluate_generator(train_generator, steps=STEP_SIZE_TRAIN)
            score_val = self.model.evaluate_generator(val_generator, steps=STEP_SIZE_VALID)
            print('Train accuracy:', score_train[1])
            print('Val accuracy:', score_val[1])
            acc_train = score_train[1]
        elif self.model_name == 'CNN2':
            print("======================CNN2 training starts:======================\n")
            t1 = time.time()
            self.model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=Adam(lr=lr),
                               metrics=['accuracy'])
            self.history = self.model.fit_generator(train_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=val_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=epochs)
            t2 = time.time()
            self.plotting('CNN2')
            score_train = self.model.evaluate_generator(train_generator, steps=STEP_SIZE_TRAIN)
            score_val = self.model.evaluate_generator(val_generator, steps=STEP_SIZE_VALID)
            print('Train accuracy:', score_train[1])
            print('Val accuracy:', score_val[1])
            acc_train = score_train[1]

        return acc_train

    def test(self, test_generator, STEP_SIZE_TEST):
        if self.model_name == 'MLP1':
            score_test = self.model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
            print('MLP1 Tested Accuracy: ', score_test[1])
            test_acc = score_test[1]

        elif self.model_name == 'MLP2':
            score_test = self.model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
            print('MLP2 Tested Accuracy: ', score_test[1])
            test_acc = score_test[1]
        elif self.model_name == 'CNN1':
            score_test = self.model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
            print('Test accuracy:', score_test[1])
            test_acc = score_test[1]
        elif self.model_name == 'CNN2':
            score_test = self.model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
            print('Test accuracy:', score_test[1])
            test_acc = score_test[1]
        return test_acc

    def plotting(self, name):

        plt.figure()

        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'validation'], loc='upper left')
        plt.savefig(name + '_acc_' + str(self.task) + '_' + self.model_name)
        # plt.show()

        plt.figure()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'validation'], loc='upper left')
        plt.savefig(name + '_loss_' + str(self.task) + '_' + self.model_name)
        # plt.show()
