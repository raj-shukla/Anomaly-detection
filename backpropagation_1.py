import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import function
import predictionData
import parameters


def model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.5,
          num_epochs = 10000, minibatch_size = 1, print_cost = True):
    
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                             
    seed = 3                                          
    (n_x, m) = X_train.shape                          
    n_y = Y_train.shape[0]                           
    costs = []                                        
    
    X, Y = function.create_placeholders(n_x, n_y)
    
    parameters = function.initialize_parameters(layers_dims)
    
    predictions = function.forward_propagation(X, parameters)
    cost = function.compute_cost(predictions, Y)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
   
    init = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                      
            seed = seed + 1

                
            minibatch_X = X_train
            minibatch_Y = Y_train
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
            epoch_cost = minibatch_cost

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        

        rmse_error, average_error = function.compute_error(predictions, Y)

        
        print ("Train Accuracy:", rmse_error.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", rmse_error.eval({X: X_test, Y: Y_test}))

        print ("Train Accuracy:", average_error.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", average_error.eval({X: X_test, Y: Y_test}))
        
        
        return parameters
        
        
        
def nn_prediction(X_train, Y_train, parameters):

    (n_x, m) = X_train.shape                          
    n_y = Y_train.shape[0] 
    rmse_error = tf.Variable(0, name= "rmse_error", dtype=tf.float32)
    average_error = tf.Variable(0, name= "average_error", dtype=tf.float32)
    X, Y = function.create_placeholders(n_x, n_y)
    predictions = function.forward_propagation(X, parameters)
    rmse_error, average_error = function.compute_error(predictions, Y)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
      
        rmse_error = rmse_error.eval({X: X_train, Y: Y_train})
        average_error = average_error.eval({X: X_train, Y: Y_train})
        
    return rmse_error, average_error



     
inputData, outputData, outputList = predictionData.DataSet(40.68)
X_train = inputData[:, 0:4000]
Y_train = outputData[:, 0:4000]
X_test =  inputData[:, 4000:4600]
Y_test  =  outputData[:, 4000:4600]
Y_test_List = outputList

layers_dims = [30, 25,  25, 1]

#print(np.shape(inputData))
#print(np.shape(outputData))

#print(np.shape(inputData))

#X_test[:, 0:5] = random.uniform(0, 1)

#X_test = X_test + random.uniform(-0.05, 0.05)
 
#parameters = model(X_train, Y_train, X_test, Y_test, layers_dims)

print (np.shape(X_test))

def diff_attack_analysis( parameters, n, a_start):

    N = np.shape(X_test)[0]
    #a_index = random.randint(0, N + 1 - n)
    a_index = random.randint(0, 1)
    a_end = a_start + a_start/100000

    X_test [a_index:a_index + n + 1:2, :] = X_test [a_index:a_index + n + 1:2, :] + random.uniform(a_start, a_end)
    X_test [a_index+1:a_index + n + 1:2, :] = X_test [a_index+1:a_index + n + 1:2, :] - random.uniform(a_start, a_end)
    #X_test [a_index:a_index + int((n + 1)/2), :] = X_test [a_index:a_index + int((n + 1)/2), :] + random.uniform(a_start, a_end)
    #X_test [a_index + int((n + 1)/2) : a_index + n + 1, :] = X_test [a_index + int((n + 1)/2) : a_index + n + 1, :] + random.uniform(a_start, a_end)
    if (n%2 == 1):
        X_test [a_index+1, :] = X_test [a_index+1, :] - random.uniform(a_start, a_end)
        

    rmse_error, average_error = nn_prediction(X_test, Y_test, parameters)
    
    return rmse_error, average_error

def attack_analysis( parameters, n, a_start):

    N = np.shape(X_test)[0]
    #a_index = random.randint(0, N + 1 - n)
    a_index = random.randint(0, 1)
    a_end = a_start/100

    X_test [a_index:a_index + n + 1, :] = X_test [a_index:a_index + n + 1, :] + random.uniform(a_start, a_end)
    

    rmse_error, average_error = nn_prediction(X_test, Y_test, parameters)
    
    return rmse_error, average_error

parameters = parameters.parameters
n = 30
a_start_1 = 0
a_start_2 = 0
rmse_error =  [[] for i in range(6)]
average_error = [[] for i in range(6)]


for i in range(0, 30):
    a_start_1 = a_start_1 + 0.01
    a_start_2 = a_start_2 - .01
    
    
    rmse_error_0, average_error_0 = attack_analysis(parameters, n, a_start_1)
    #rmse_error_1, average_error_1 = attack_analysis(parameters, n, a_start_2)
    #rmse_error_2, average_error_2 = diff_attack_analysis(parameters, n, a_start_1)
    
    rmse_error[0].append(rmse_error_0)
    #rmse_error[1].append(rmse_error_1)
    #rmse_error[2].append(rmse_error_2)
    
    average_error[0].append(average_error_0)
    #average_error[1].append(average_error_1)
    #average_error[2].append(average_error_2)
    
    
    
n = 0 
a_start = 0.15    
for i in range(0, 30):

    n = n + 1 
    
    rmse_error_0, average_error_0 = attack_analysis(parameters, n, a_start)
    rmse_error_1, average_error_1 = attack_analysis(parameters, n, a_start)
    rmse_error_2, average_error_2 = diff_attack_analysis(parameters, n, a_start)
    
    rmse_error[3].append(rmse_error_0)
    rmse_error[4].append(rmse_error_1)
    rmse_error[5].append(rmse_error_2)
    
    average_error[3].append(average_error_0)
    average_error[4].append(average_error_1)
    average_error[5].append(average_error_2)
    
    
    

print(rmse_error)
print(average_error)





