import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
import readData
np.set_printoptions(threshold=np.nan)

days = readData.days
flow = np.array(readData.flow)
flowList = np.array(readData.flowList)
time = np.array(readData.time)
postMile = np.array(readData.postMile)
lanes = np.array(readData.lanes)

flow = (flow - np.min(flow))/(np.max(flow) - np.min(flow))
flowArray = []

for i, val in enumerate(flow):
    flowArray.append(np.array(flow[i].reshape(24*12, 136)))
    
print(np.shape(flow))

flowArray = np.asarray(flowArray)

point = 65
flowAtPoint = flowArray[:, :, point:point+1]

print(np.shape(flowAtPoint))
array = flowArray[:, 100:150:, 5]
print(np.shape(array))

diff = [[], [], [], [], [], [], [], [], [], [], [], []]
attack = [[], [], [], [], [], [], [], [], [], [], [], []]
attack_type = [[], [], [], [], [], [], [], [], [], [], [], []]

meanFlow = []

for i in range(0, 7):
    print(np.mean(flowAtPoint[i:28:7, :, : ], axis=0))
    meanFlow.append(np.squeeze(np.mean(flowAtPoint[i:28:7, :, : ], axis=0)))

print(meanFlow)
print(np.shape(meanFlow))
         
for i in range(0, 288 - len(diff)):
    for j in range(0, len(diff)):
        diff[j].append( np.absolute(np.mean(flowAtPoint[:, i:i+j+1, :]) - np.mean(flowAtPoint[:, i+j+1:i+j+1 + j+1, :])) )
        
        
#for i in range(0, len(diff)):
    #print(np.std(diff[i]))
    

#plt.plot(flowAtPoint[5, 0:288, :])
#plt.show()

#arr = np.mean(flowAtPoint[:, 0:288, :], axis=0)
#print(arr)
#plt.plot(arr)
#plt.show()


#for i in range(0, 30):
    #plt.plot(flowAtPoint[i, :, :])
    #plt.axis([0, 300, 0, 1])
    #plt.show()

flowWithAttack = flowAtPoint[5, 0:288, :] 
flowWithAttack[150:200] = flowWithAttack[150:200] + 2*random.uniform(0.05, 0.10)
#flowWithAttack[50:90] = flowWithAttack[50:90] - 2*random.uniform(0.10, 0.20)
flowWithAttack[200:250] = flowWithAttack[200:250] - 2*random.uniform(0.10, 0.20)

threshhold = [max(i) for i in diff]
print(threshhold)
for i in range(0, 288 -len(diff)):
    for j in range(0, len(diff)):
        if (abs( np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])) > threshhold[j]):
            attack[j].append(i+j+1)
            attack_type[j].append(np.sign(np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])))
            
            
for i in range(0, len(attack)):   
    print(attack[i])
    print(attack_type[i])
    plt.scatter(attack[i], attack[i])
    plt.grid()
    plt.show()

#print(flowWithAttack[140:190])

plt.plot(flowAtPoint[5, 0:288, :] )
plt.axis([0, 300, 0, 1])
plt.show()

plt.plot(flowWithAttack)
plt.axis([0, 300, 0, 1])
plt.show()


         

