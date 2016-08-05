# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
from math import sqrt
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import csv

data_folder_path = '../data/'
pickle_file_name = 'notMNIST.pickle'
def load():
	with open(data_folder_path + pickle_file_name, 'rb') as f:
	  save = pickle.load(f)
	  train_dataset = save['train_dataset']
	  train_labels = save['train_labels']
	  valid_dataset = save['valid_dataset']
	  valid_labels = save['valid_labels']
	  test_dataset = save['test_dataset']
	  test_labels = save['test_labels']
	  del save  # hint to help gc free up memory
	  print('Training set', train_dataset.shape, train_labels.shape)
	  print('Validation set', valid_dataset.shape, valid_labels.shape)
	  print('Test set', test_dataset.shape, test_labels.shape)
	  
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

image_size = 28
num_labels = 10

# DEFINE NETWORK
network_shape = [image_size * image_size,600,300,150,num_labels]
initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.95
beta = 0.1

# RUN TRAINING
batch_size = 150
num_steps = 3001
dropout_keep_prob = 0.5

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#def create_network(network_shape,initial_learning_rate,decay_steps,decay_rate,beta):
#graph = tf.Graph()
graph_nn_final = tf.Graph()
with graph_nn_final.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_dataset = tf.placeholder(tf.float32, shape=(None, network_shape[0]))
  tf_labels = tf.placeholder(tf.float32, shape=(None, network_shape[-1]))
	
  keep_prob = tf.placeholder(tf.float32)
  
  # Variables
  weights = []
  biases = []
  
  for i in range(len(network_shape)-1):
	weights.append(tf.Variable(tf.truncated_normal([network_shape[i], network_shape[i+1]],stddev=sqrt(2/network_shape[i]))))
	biases.append(tf.Variable(tf.zeros([network_shape[i+1]])))
	
  global_step = tf.Variable(0)
  
  # Training computation (with dropout)
  logits = tf.matmul(tf_dataset, weights[0]) + biases[0]
  for i in range(1,len(weights)):
	logits = tf.matmul(tf.nn.dropout(tf.nn.relu(logits), keep_prob), weights[i]) + biases[i]

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels))

  # L2 Regularization
  regularizers = tf.nn.l2_loss(weights[0]) + tf.nn.l2_loss(biases[0])
  for i in range(1,len(weights)):
	regularizers += tf.nn.l2_loss(weights[i]) + tf.nn.l2_loss(biases[i])

  loss += beta * regularizers

  learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
  
  # Passing global_step to minimize() will increment it at each step.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  prediction = tf.nn.softmax(logits)

#return graph

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load()

#graph_nn_final = create_network(network_shape,initial_learning_rate,decay_steps,decay_rate,beta)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
 
# RUN TRAINING
old_valid_accuracy = None
with tf.Session(graph=graph_nn_final) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : dropout_keep_prob}
        _, l, predictions = session.run(
          [optimizer, loss, prediction], feed_dict=feed_dict)
            
        if (step % 100 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(session.run(prediction, feed_dict={tf_dataset : batch_data, tf_labels : batch_labels, keep_prob : 1.0}), batch_labels))
          
          valid_prediction = session.run(prediction, feed_dict={tf_dataset : valid_dataset, tf_labels : valid_labels, keep_prob : 1.0})
          valid_accuracy = accuracy(valid_prediction, valid_labels)
          print("Validation accuracy: %.1f%%" % valid_accuracy)
        
          #if not old_valid_accuracy is None and old_valid_accuracy > valid_accuracy:
          #  print("Stop at step %d" % step)
          #  break
          #else:
          #  old_valid_accuracy = valid_accuracy
        
	  test_prediction = session.run(prediction, feed_dict={tf_dataset : test_dataset, tf_labels : test_labels, keep_prob : 1.0})
	  test_accuracy = accuracy(test_prediction, test_labels)
      print("Test accuracy: %.1f%%" % test_accuracy)

print("Saving results to results/results.csv")
with open('../results/results.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([str(network_shape),str(initial_learning_rate),str(decay_steps),str(decay_rate),str(beta),str(test_accuracy)])
print("Results saved")