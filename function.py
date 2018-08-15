import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")
   
    return X, Y



def initialize_parameters(layers_dims):
   
    tf.set_random_seed(1)  
    parameters = {}
    L = len(layers_dims)                  
        
    W1 = tf.get_variable("W1", [25, 30], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [25, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [25, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):
    
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)                                            
    A1 = tf.nn.relu(Z1)                                            
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                             
    A2 = tf.nn.relu(Z2)                                            
    Z3 = tf.add(tf.matmul(W3, A2), b3)   
    A3 = tf.nn.sigmoid(Z3)                                           
    
    return A3

def compute_error(predictions, Y):
    average_error = tf.reduce_mean(tf.abs(predictions - Y))
    rmse_error = tf.sqrt(tf.reduce_mean((predictions-Y)*(predictions-Y)))
    
    return rmse_error, average_error

def compute_cost(predictions, Y):
    
    cost = tf.sqrt(tf.reduce_mean((predictions-Y)*(predictions-Y)))
    #cost = tf.reduce_mean(-Y*tf.log(predictions) - (1-Y)*tf.log(predictions))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  
    return cost


