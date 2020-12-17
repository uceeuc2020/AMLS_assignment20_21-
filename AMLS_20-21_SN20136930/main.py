from A1.A1 import A1
from A2.A2 import A2
from B1.B1 import B1
from B2.B2 import B2

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

import argparse
import os
"""
args
"""
parser = argparse.ArgumentParser(description='HW')
parser.add_argument('--model',metavar='MODEL',default='MLP2',help='model: MLP1/MLP2/CNN1/CNN2')
parser.add_argument('--lr',default=0.001,type=float,help='learning rate')
parser.add_argument('--epochs',default=50,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--batch_size',default=256,type=int,metavar='N',help='mini-batch size for training (default: 16)')
parser.add_argument('--batch_size_test',default=128,type=int,help='mini-batch size for testing (default: 128)')
parser.add_argument('--task',default=0,type=int,help='task index, 0: A1, 1: A2, 2: B2, 3:B1')
parser.add_argument('image_shape_1',default = (32,32),type=tuple)#(218,178)
parser.add_argument('image_shape_2',default = (32,32),type=tuple)#(218,178)
args = parser.parse_args()

# ======================================================================================================================
# Data preprocessing
# define image loading
def data_preprocessing(task):
    if task == 0:#A1
        traindf=pd.read_csv('./Datasets/celeba/labels.csv',dtype=str)
        datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)
        train_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/celeba/img/",
          x_col="img_name",
          y_col="gender",
          subset="training",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="binary",
          target_size=args.image_shape_1)
        val_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/celeba/img/",
          x_col="img_name",
          y_col="gender",
          subset="validation",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="binary",
          target_size=args.image_shape_1)
        testdf=pd.read_csv('./Datasets/celeba_test/labels.csv',dtype=str)
        test_datagen=ImageDataGenerator(rescale=1./255.)
        test_generator=test_datagen.flow_from_dataframe(
          dataframe=testdf,
          directory="./Datasets/celeba_test/img/",
          x_col="img_name",
          y_col="gender",
          batch_size=args.batch_size_test,
          seed=42,
          shuffle=False,
          class_mode="binary",
          target_size=args.image_shape_1)
    elif task == 1:#A2
        traindf=pd.read_csv('./Datasets/celeba/labels.csv',dtype=str)
        datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)
        train_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/celeba/img/",
          x_col="img_name",
          y_col="smiling",
          subset="training",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="binary",
          target_size=args.image_shape_1)
        val_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/celeba/img/",
          x_col="img_name",
          y_col="smiling",
          subset="validation",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="binary",
          target_size=args.image_shape_1)
        testdf=pd.read_csv('./Datasets/celeba_test/labels.csv',dtype=str)
        test_datagen=ImageDataGenerator(rescale=1./255.)
        test_generator=test_datagen.flow_from_dataframe(
          dataframe=testdf,
          directory="./Datasets/celeba_test/img/",
          x_col="img_name",
          y_col="smiling",
          batch_size=args.batch_size_test,
          seed=42,
          shuffle=False,
          class_mode="binary",
          target_size=args.image_shape_1)
    elif task == 2:#B2
        traindf=pd.read_csv('./Datasets/cartoon_set/labels.csv',dtype=str)
        datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)
        train_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/cartoon_set/img/",
          x_col="file_name",
          y_col="eye_color",
          subset="training",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="categorical",
          target_size=args.image_shape_2)
        val_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/cartoon_set/img/",
          x_col="file_name",
          y_col="eye_color",
          subset="validation",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="categorical",
          target_size=args.image_shape_2)
        testdf=pd.read_csv('./Datasets/cartoon_set_test/labels.csv',dtype=str)
        test_datagen=ImageDataGenerator(rescale=1./255.)
        test_generator=test_datagen.flow_from_dataframe(
          dataframe=testdf,
          directory="./Datasets/cartoon_set_test/img/",
          x_col="file_name",
          y_col="eye_color",
          batch_size=args.batch_size_test,
          seed=42,
          shuffle=False,
          class_mode="categorical",
          target_size=args.image_shape_2)
    elif task == 3:#B1
        traindf=pd.read_csv('./Datasets/cartoon_set/labels.csv',dtype=str)
        datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)
        train_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/cartoon_set/img/",
          x_col="file_name",
          y_col="face_shape",
          subset="training",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="categorical",
          target_size=args.image_shape_2)
        val_generator=datagen.flow_from_dataframe(
          dataframe=traindf,
          directory="./Datasets/cartoon_set/img/",
          x_col="file_name",
          y_col="face_shape",
          subset="validation",
          batch_size=args.batch_size,
          seed=42,
          shuffle=True,
          class_mode="categorical",
          target_size=args.image_shape_2)
        testdf=pd.read_csv('./Datasets/cartoon_set_test/labels.csv',dtype=str)
        test_datagen=ImageDataGenerator(rescale=1./255.)
        test_generator=test_datagen.flow_from_dataframe(
          dataframe=testdf,
          directory="./Datasets/cartoon_set_test/img/",
          x_col="file_name",
          y_col="face_shape",
          batch_size=args.batch_size_test,
          seed=42,
          shuffle=False,
          class_mode="categorical",
          target_size=args.image_shape_2)
    return train_generator, val_generator, test_generator


#"""
# ======================================================================================================================

# Task A1
if task==0:
    print('start task 0')
    #task = 0
    model_A1 = A1(args.model,args.task,args.image_shape_1)                 # Build model object.
    train_generator, val_generator, test_generator = data_preprocessing(task=args.task)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    acc_A1_train = model_A1.train(train_generator, val_generator, args.lr, args.epochs, STEP_SIZE_TRAIN, STEP_SIZE_VALID) # Train model based on the training set (you should fine-tune your model based on validation set.)
    acc_A1_test = model_A1.test(test_generator, STEP_SIZE_TEST)   # Test model based on the test set.
    #Clean up memory/GPU etc...             # Some code to free memory if necessary.

    ##########
    # Logic Regression: 50 epoch

    # MLP: (99.57/ 89.89/ ) 50 epoch
    ##########
#"""

#"""
# ======================================================================================================================
# Task A2
if task==1:
    print('start task 1')
    #task = 1
    model_A2 = A1(args.model,args.task,args.image_shape_1)                 # Build model object.
    train_generator, val_generator, test_generator = data_preprocessing(task=args.task)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    acc_A2_train = model_A2.train(train_generator, val_generator, args.lr, args.epochs, STEP_SIZE_TRAIN, STEP_SIZE_VALID) # Train model based on the training set (you should fine-tune your model based on validation set.)
    acc_A2_test = model_A2.test(test_generator, STEP_SIZE_TEST)   # Test model based on the test set.
    #Clean up memory/GPU etc...             # Some code to free memory if necessary.

    ##########
    # Logic Regression: (94.12/ 86.40 / ) 50 epoch

    # MLP: (99.12/ 85.29/ ) 50 epoch

    ##########
    #"""

#"""
# ======================================================================================================================
# Task B2
if task==2:
    print('start task 2')
    #task = 2
    model_A3 = A1(args.model,args.task,args.image_shape_2)                 # Build model object.
    train_generator, val_generator, test_generator = data_preprocessing(task=args.task)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    acc_A3_train = model_A3.train(train_generator, val_generator, args.lr, args.epochs, STEP_SIZE_TRAIN, STEP_SIZE_VALID) # Train model based on the training set (you should fine-tune your model based on validation set.)
    acc_A3_test = model_A3.test(test_generator, STEP_SIZE_TEST)   # Test model based on the test set.

    ##########
    # CNN1:
    # CNN2:
    ##########


# ======================================================================================================================
# Task B1
if task==3:
    print('start task 3')
    #task = 3
    model_A4 = A1(args.model,args.task,args.image_shape_2)                 # Build model object.
    train_generator, val_generator, test_generator = data_preprocessing(task=args.task)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    acc_A4_train = model_A4.train(train_generator, val_generator, args.lr, args.epochs, STEP_SIZE_TRAIN, STEP_SIZE_VALID) # Train model based on the training set (you should fine-tune your model based on validation set.)
    acc_A4_test = model_A4.test(test_generator, STEP_SIZE_TEST)   # Test model based on the test set.
    #Clean up memory/GPU etc...             # Some code to free memory if necessary.

    ##########
    # CNN1:
    # CNN2:
    ##########
#"""

# ======================================================================================================================
## Print out your results with following format:
#print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                        acc_A2_train, acc_A2_test,
#                                                        acc_B1_train, acc_B1_test,
#                                                        acc_B2_train, acc_B2_test))
