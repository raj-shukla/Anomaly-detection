import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
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
flowAtPoint = flowArray[5, :, point:point+10]
print(np.shape(flowAtPoint))

X = flowAtPoint
pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)

X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)


np.random.seed(42)
noisy = flowAtPoint
noisy[150:200, :] = noisy[150:200, :] + 0.1

plt.plot(X)
plt.show()

plt.plot(noisy)
plt.show()

pca = PCA(0.99).fit(noisy)
print(pca.n_components_)

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)

plt.plot(filtered)
plt.show()

#pca_1 = PCA().fit(X)

#plt.plot(np.cumsum(pca_1.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.show()

#plt.plot(X)
#plt.show()

#plt.plot(X_pca)
#plt.show()


