import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import function
import predictionData
import parameters
        
        
def nn_prediction(X_train):

  
    
    predictions = function.forward_propagation(X, parameters.parameters)
    
    return predictions

inputData, outputData, outputList = predictionData.DataSet(40.68)

X = tf.Variable(tf.zeros(30, None))

X_hat = tf.add(X, 0.1)

Y = nn_prediction(X)
Y_hat = nn_prediction(X_hat)




learning_rate = tf.placeholder(tf.float32, ())


loss = function.compute_cost(Y, Y_hat)
optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)





epsilon = tf.placeholder(tf.float32, ())

below = X_hat - epsilon
above = X_hat + epsilon
projected = tf.clip_by_value(tf.clip_by_value(X_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(X_hat, projected)

demo_epsilon = 2.0/255.0 # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 1 # "guacamole"

inputData, outputData, outputList = predictionData.DataSet(40.68)

for i in range(demo_steps):
    _, loss_value = sess.run([optim_step, loss], feed_dict={learning_rate: demo_lr, Y_hat: demo_target})

    sess.run(project_step, feed_dict={X: inputData[:, 4000:4001], epsilon: demo_epsilon})
    


     


'''

inputData, outputData, outputList = predictionData.DataSet(40.68)
X_train = inputData[:, 0:4000]
Y_train = outputData[:, 0:4000]
X_test =  inputData[:, 4000:4600]
Y_test  =  outputData[:, 4000:4600]
Y_test_List = outputList

layers_dims = [30, 25,  25, 1]

print(np.shape(inputData))
print(np.shape(outputData))

print(np.shape(inputData))

#X_test[:, 0:5] = random.uniform(0, 1)

#X_test = X_test + random.uniform(-0.05, 0.05)
'''
 
parameters = model(X_train, Y_train, X_test, Y_test, layers_dims)

rmse_error, average_error = nn_prediction(X_test, Y_test, parameters)

print (parameters)

print(rmse_error)
print(average_error)





