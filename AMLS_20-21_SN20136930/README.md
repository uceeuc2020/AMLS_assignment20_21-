## AMS Homework

### Introduction

This repository contains four supervised tasks on two datasets. The datasets are provided by course, and includes two binary classification tasks and two multiclass classification tasks. 



### Code Usage

```python
# py file
python main.py --epochs 100 --lr 1e-3 --model CNN1 --task 2
# ipynb file, change the settings manually
```

task: 0 for A1, 1 for A2, 2 for B2, 3 for B1.

model: MLP1, MLP2, CNN1, CNN2

lr: learning rate for Adam optimizer

epochs: 100 training epochs

### File explanation

For py files:

main.py: main training file for training one model on one task only. A preprocessing of image such as resizing the shape is defined in main.py. 

A1.py: The object class defined for the model training, model testing, and plotting for loss and accuracy figures. A1.py, A2.py, B1.py, B2.py files are all the copied file as A1.py.

For ipynb file:

main.ipynb: contains all the definition for classes and functions, could run directly.

### Packages

time, os: for saving files, and recording training time
keras: for Neural Network training
A1, A2, A3, A4: Self-defined models objects 
pandas: for loading csv files
numpy: for tackling matrix data

