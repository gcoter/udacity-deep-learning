from __future__ import print_function
from MLP import MLP
from datasetmanager import get_and_reformat_all_datasets

# ===== MAIN =====
# DEFINE MLP
image_size = 28
num_labels = 10
network_shape = [image_size * image_size,600,300,150,num_labels]
initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.95
beta = 0.1
dropout_keep_prob = 0.5
mlp = MLP(network_shape, initial_learning_rate, decay_steps, decay_rate, regularization_parameter, dropout_keep_prob)

# GET AND REFORMAT ALL DATASETS
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_and_reformat_all_datasets()
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# RUN TRAINING
batch_size = 150
num_steps = 3001
mlp.train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, batch_size, num_steps)