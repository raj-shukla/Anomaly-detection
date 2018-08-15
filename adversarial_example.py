import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import function
#import backpropagation
import parameters


image = tf.Variable(tf.zeros((30, 1)))
def nn_prediction(X_train, Y_train):

    (n_x, m) = X_train.shape                          
    n_y = Y_train.shape[0] 
    #rmse_error = tf.Variable(0, name= "rmse_error", dtype=tf.float32)
    #average_error = tf.Variable(0, name= "average_error", dtype=tf.float32)
    X, _ = function.create_placeholders(n_x, n_y)
    predictions = function.forward_propagation(X, parameters.parameters)
    #rmse_error, average_error = function.compute_error(predictions, Y)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
      
        predictions = predictions.eval({X: X_train})
        #average_error = average_error.eval({X: X_train, Y: Y_train})
        
    return predictions


#img = [[0.4], [0.3], [0.2], [0.2], [0.5], [0.4], [0.3], [0.2], [0.2], [0.5], [0.4], [0.3], [0.2], [0.2], [0.5], [0.4], [0.3], [0.2], [0.2], [0.5], [0.4], [0.3], [0.2], [0.2], [0.5], [0.4], [0.3], [0.2], [0.2], [0.5]]


#print(np.shape(img))

#img = np.asarray(img)
img_predict = [[0.4]]
img_predict = np.asarray(img_predict)

logits = nn_prediction(image, img_predict)

x = tf.placeholder(tf.float32, (30, 1))

x_hat = image # our trainable adversarial input
assign_op = tf.assign(x_hat, x)

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

#labels = tf.one_hot(y_hat, 1000)
labels = y_hat
#loss = logits
#loss = function.compute_cost(logits, labels)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = [labels]))
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])
    

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)
    
demo_epsilon = 2.0/255.0 # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 924 # "guacamole"

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i+1) % 10 == 0:
        print('step %d, loss=%g' % (i+1, loss_value))
    

adv = x_hat.eval() # retrieve the adversarial example
