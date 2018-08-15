import math
import numpy as np
import matplotlib.pyplot as plt
import random
import predict_function
import predictionData
import parameter
import datetime

     


def diff_attack_analysis(parameters, n, a_start, a_type):

    N = np.shape(parameter.X_test)[0]
    #a_index = random.randint(0, N + 1 - n)
    a_index = random.randint(0, 1)
    a_end = a_start + a_start/100
     
    tmp_array = np.copy(parameter.X_test)
    parameter.X_test [a_index:a_index + n + 1:2, :] = parameter.X_test [a_index:a_index + n + 1:2, :] + random.uniform(a_start, a_end)
    parameter.X_test [a_index+1:a_index + n + 1:2, :] = parameter.X_test [a_index+1:a_index + n + 1:2, :] - random.uniform(a_start, a_end)
      
    rmse_error, average_error = predict_function.predict(parameter.X_test, parameter.Y_test, parameters)
    
    parameter.X_test = np.copy(tmp_array)
    
    
    return rmse_error, average_error

def attack_analysis(parameters, n, a_start, a_type):

    N = np.shape(parameter.X_test)[0]
    a_index = random.randint(0, 1)
    a_end = a_start/100



    

    #print("###########")
    tmp_array = np.copy(parameter.X_test)
    #print (parameter.X_test[0:2, 1:2])
    if (a_type == "additive"):
        parameter.X_test [a_index:a_index + n + 1, :] = parameter.X_test [a_index:a_index + n + 1, :] + random.uniform(a_start, a_end)
    else:
        parameter.X_test [a_index:a_index + n + 1, :] = parameter.X_test [a_index:a_index + n + 1, :] - random.uniform(a_start, a_end)

    rmse_error, average_error = predict_function.predict( parameter.X_test, parameter.Y_test, parameters)
    
    parameter.X_test = np.copy(tmp_array)
    
    #print (parameter.X_test[0:2, 1:2])
    
    return rmse_error, average_error



rmse_error =  [[] for i in range(30)]
average_error = [[] for i in range(30)]




def find_attack(a_start, n, param, a_type):

    rmse = []
    average = []
    
    tmpArray_1 = []
    tmpArray_2 = []
    
    for i in range(0, 30):
    
        if (param == 1):
            a_start = a_start + 0.10
        else:
            n = n + 1
    
        for j in range(0, 1000):
            
            if (a_type != "differential"):
                rmse_error_0, average_error_0 = attack_analysis(parameters, n, a_start, a_type)
                #print(param)
                #print(rmse_error_0)
                #print(a_start)
            else:
                rmse_error_0, average_error_0 = diff_attack_analysis(parameters, n, a_start, a_type)
            tmpArray_1.append(rmse_error_0)
            tmpArray_2.append(average_error_0)
        
        rmse.append(np.mean(tmpArray_1))
        average.append(np.mean(tmpArray_2))
        
        tmpArray_1 = []
        tmpArray_2 = []
       
    return rmse, average
 


parameters = parameter.parameters

rmse_error[0], average_error[0] = find_attack(0, 5, 1, "additive")
rmse_error[1], average_error[1] = find_attack(0, 10, 1, "additive")
rmse_error[2], average_error[2] = find_attack(0, 15, 1, "additive")
rmse_error[3], average_error[3] = find_attack(0, 20, 1, "additive")
rmse_error[4], average_error[4] = find_attack(0, 25, 1, "additive")


rmse_error[5], average_error[5] = find_attack(0, 5, 1, "deductive") 
rmse_error[6], average_error[6] = find_attack(0, 10, 1, "deductive") 
rmse_error[7], average_error[7] = find_attack(0, 15, 1, "deductive") 
rmse_error[8], average_error[8] = find_attack(0, 20, 1, "deductive") 
rmse_error[9], average_error[9] = find_attack(0, 25, 1, "deductive") 

rmse_error[10], average_error[10] = find_attack(0, 5, 1, "differential")
rmse_error[11], average_error[11] = find_attack(0, 10, 1, "differential") 
rmse_error[12], average_error[12] = find_attack(0, 15, 1, "differential") 
rmse_error[13], average_error[13] = find_attack(0, 20, 1, "differential") 
rmse_error[14], average_error[14] = find_attack(0, 25, 1, "differential")  

rmse_error[15], average_error[15] = find_attack(0.10, 0, 0, "additive")
rmse_error[16], average_error[16] = find_attack(0.20, 0, 0, "additive")
rmse_error[17], average_error[17] = find_attack(0.30, 0, 0, "additive")
rmse_error[18], average_error[18] = find_attack(0.40, 0, 0, "additive")
rmse_error[19], average_error[19] = find_attack(0.50, 0, 0, "additive")

rmse_error[20], average_error[20] = find_attack(0.10, 0, 0, "deductive")
rmse_error[21], average_error[21] = find_attack(0.20, 0, 0, "deductive") 
rmse_error[22], average_error[22] = find_attack(0.30, 0, 0, "deductive") 
rmse_error[23], average_error[23] = find_attack(0.40, 0, 0, "deductive") 
rmse_error[24], average_error[24] = find_attack(0.50, 0, 0, "deductive")
  
rmse_error[25], average_error[25] = find_attack(0.10, 0, 0, "differential")   
rmse_error[26], average_error[26] = find_attack(0.20, 0, 0, "differential")    
rmse_error[27], average_error[27] = find_attack(0.30, 0, 0, "differential")    
rmse_error[28], average_error[28] = find_attack(0.40, 0, 0, "differential")    
rmse_error[29], average_error[29] = find_attack(0.50, 0, 0, "differential")     
      
 
print (rmse_error)
print (average_error)







