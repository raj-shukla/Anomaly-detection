import math
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import function
import predictionData
import parameters
import datetime


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
    a_index = random.randint(0, N + 1 - n)
    #a_index = random.randint(0, 1)
    a_end = a_start + a_start/100

    X_test [a_index:a_index + n + 1:2, :] = X_test [a_index:a_index + n + 1:2, :] + random.uniform(a_start, a_end)
    X_test [a_index+1:a_index + n + 1:2, :] = X_test [a_index+1:a_index + n + 1:2, :] - random.uniform(a_start, a_end)
    #X_test [a_index:a_index + int((n + 1)/2), :] = X_test [a_index:a_index + int((n + 1)/2), :] + random.uniform(a_start, a_end)
    #X_test [a_index + int((n + 1)/2) : a_index + n + 1, :] = X_test [a_index + int((n + 1)/2) : a_index + n + 1, :] + random.uniform(a_start, a_end)
    if (n%2 == 1):
        X_test [a_index+1, :] = X_test [a_index+1, :] - random.uniform(a_start, a_end)
        

    rmse_error, average_error = nn_prediction(X_test, Y_test, parameters)
    
    return rmse_error, average_error

def attack_analysis( parameters, n, a_start, a_type):

    N = np.shape(X_test)[0]
    #a_index = random.randint(0, N + 1 - n)
    a_index = random.randint(0, 1)
    a_end = a_start/100

    if (a_type == "additive"):
        print(np.shape(X_test))
        X_test [a_index:a_index + n + 1, :] = X_test [a_index:a_index + n + 1, :] + random.uniform(a_start, a_end)
    else:
        X_test [a_index:a_index + n + 1, :] = X_test [a_index:a_index + n + 1, :] - random.uniform(a_start, a_end)

    rmse_error, average_error = nn_prediction(X_test, Y_test, parameters)
    
    return rmse_error, average_error

parameters = parameters.parameters

rmse_error =  [[] for i in range(6)]
average_error = [[] for i in range(6)]




def find_attack(a_start, n, param, a_type):

    rmse = []
    average = []
    
    tmpArray_1 = []
    tmpArray_2 = []
    
    for i in range(0, 30):
        print(datetime.datetime.now())
    
        if (param == 1):
            print("####")
            a_start = a_start + 0.01
        else:
            print("########################################")
            n = n + 1
    
        for j in range(0, 10):
            
            if (a_type != "differential"):
                print("#########")
                rmse_error_0, average_error_0 = attack_analysis(parameters, n, a_start, a_type)
            else:
                print("#####################")
                rmse_error_0, average_error_0 = diff_attack_analysis(parameters, n, a_start)
            tmpArray_1.append(rmse_error_0)
            tmpArray_2.append(average_error_0)
        
        print(len(tmpArray_1))
        print(len(tmpArray_2))
        rmse.append(np.mean(tmpArray_1))
        average.append(np.mean(tmpArray_2))
        
        tmpArray_1 = []
        tmpArray_2 = []
       
    return rmse, average
 
  
rmse_error[0], average_error[0] = find_attack(0, 20, 1, "additive")

print("check_1")
rmse_error[1], average_error[1] = find_attack(0, 20, 1, "deductive") 
#rmse_error[2], average_error[2] = find_attack(0, 20, 1, "differential") 

#rmse_error[3], average_error[3] = find_attack(0.10, 0, 0, "additive")
#rmse_error[4], average_error[4] = find_attack(0.10, 0, 0, "deductive") 
#rmse_error[5], average_error[5] = find_attack(0.10, 0, 0, "differential")    
        
 

print(rmse_error)
print(average_error)





