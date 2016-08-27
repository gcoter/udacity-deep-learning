"""
A class I created to construct my neural networks
"""

from __future__ import print_function
from math import sqrt
import numpy as np
import tensorflow as tf
import csv

# Defines MLP (MultiLayer Perceptron) using tensorflow
class MLP(object):
	# Network shape : [num_neurons_input_layer, num_neurons_hidden_layer_1, num_neurons_hidden_layer_2, ... ,num_neurons_output_layer]
	def __init__(self, network_shape, initial_learning_rate, decay_steps, decay_rate, regularization_parameter, dropout_keep_prob = 0.5):
		self.network_shape = network_shape
		self.num_layers = len(self.network_shape)
		self.initial_learning_rate = initial_learning_rate
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.regularization_parameter = regularization_parameter
		self.dropout_keep_prob = dropout_keep_prob
		self.create_network()
		self.output_file_path = '../results/results.csv'
		
	def create_network(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			# Input
			self.tf_dataset = tf.placeholder(tf.float32, shape=(None, self.network_shape[0]))
			self.tf_labels = tf.placeholder(tf.float32, shape=(None, self.network_shape[-1]))

			# Dropout keep probability (set to 1.0 for validation and test)
			self.keep_prob = tf.placeholder(tf.float32)

			# Variables
			self.weights = []
			self.biases = []

			# Constructs the network according to the given shape array
			for i in range(self.num_layers-1):
				self.weights.append(tf.Variable(tf.truncated_normal([self.network_shape[i], self.network_shape[i+1]],stddev=1.0/(self.network_shape[i]))))
				self.biases.append(tf.Variable(tf.zeros([self.network_shape[i+1]])))

			# Global Step for learning rate decay
			global_step = tf.Variable(0)

			# Training computation (with dropout)
			logits = tf.matmul(self.tf_dataset, self.weights[0]) + self.biases[0]
			for i in range(1,num_layers-2):
				with tf.name_scope("layer_"+str(i)) as scope:
					logits = tf.matmul(tf.nn.dropout(tf.nn.relu(logits), keep_prob), weights[i]) + biases[i]
			
			logits = tf.matmul(logits, weights[-1]) + biases[-1]

			# Cross entropy loss
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_labels))

			# L2 Regularization
			regularizers = tf.nn.l2_loss(self.weights[0]) + tf.nn.l2_loss(self.biases[0])
			for i in range(1,self.num_layers-1):
				regularizers += tf.nn.l2_loss(self.weights[i]) + tf.nn.l2_loss(self.biases[i])

			self.loss += self.regularization_parameter * regularizers

			learning_rate = self.initial_learning_rate #tf.train.exponential_decay(self.initial_learning_rate, global_step, self.decay_steps, self.decay_rate)

			# Passing global_step to minimize() will increment it at each step.
			self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

			# Predictions for the training, validation, and test data.
			self.prediction = tf.nn.softmax(logits)
			
	def train(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, batch_size, num_steps):
		old_valid_accuracy = None
		with tf.Session(graph=self.graph) as session:
			tf.initialize_all_variables().run()
			print("Initialized ", str(self.network_shape))
			for step in range(num_steps+1):
				# Pick an offset within the training data, which has been randomized.
				offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
				# Generate a minibatch.
				batch_data = train_dataset[offset:(offset + batch_size), :]
				batch_labels = train_labels[offset:(offset + batch_size), :]

				feed_dict = {self.tf_dataset : batch_data, self.tf_labels : batch_labels, self.keep_prob : self.dropout_keep_prob}
				_, l, predictions = session.run([self.optimizer, self.loss, self.prediction], feed_dict=feed_dict)
					
				if (step % 100 == 0):
					print("Minibatch loss at step %d: %f" % (step, l))
					print("Minibatch accuracy: %.1f%%" % MLP.accuracy(session.run(self.prediction, feed_dict={self.tf_dataset : batch_data, self.tf_labels : batch_labels, self.keep_prob : 1.0}), batch_labels))

					valid_prediction = session.run(self.prediction, feed_dict={self.tf_dataset : valid_dataset, self.tf_labels : valid_labels, self.keep_prob : 1.0})
					valid_accuracy = MLP.accuracy(valid_prediction, valid_labels)
					print("Validation accuracy: %.1f%%" % valid_accuracy)

			test_prediction = session.run(self.prediction, feed_dict={self.tf_dataset : test_dataset, self.tf_labels : test_labels, self.keep_prob : 1.0})
			test_accuracy = MLP.accuracy(test_prediction, test_labels)
			print("Test accuracy: %.1f%%" % test_accuracy)
			self.save_results(test_accuracy)
	
	@staticmethod
	def accuracy(predictions, labels):
		return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
		
	def save_results(self,test_accuracy):
		print("Saving results to ", self.output_file_path)
		with open(self.output_file_path, 'a') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			spamwriter.writerow([str(self.network_shape),str(self.initial_learning_rate),str(self.decay_steps),str(self.decay_rate),str(self.regularization_parameter),str(test_accuracy)])
		print("Results saved")