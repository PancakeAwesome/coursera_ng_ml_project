# 本文是降维
# load data
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('ex7data1.mat')
X = data['X']

print(X.shape)
print(X[:5])

plt.plot(X[:, 0], X[:, 1], 'bo')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# feature normalize 特征归一化


def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros_like(mu)

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X_norm - mu) / sigma

    return X_norm, mu, sigma

# complete the pca


def pca(data_normal):
    m, n = data_normal.shape
    # sigma = np.cov(data_normal)
    sigma = data_normal.T.dot(data_normal) / m
    U, S, V = np.linalg.svd(sigma)
    return U, S, V

# 第一步：归一化
X_norm, mu, sigma = featureNormalize(X)
# 第二步得到U矩阵
U, S, V = pca(X_norm)
print("you should expect to see -0.707107 -0.707107)\n")
print(U[0, 0], U[1, 0])

# 显示归一化可视化效果
plt.plot(X_norm[:,0],X_norm[:,1],'bo')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 降维2->1
def projectData(x,u,k):
    z = np.zeros((x.shape[0],k))
    u_reduce = u[:,:k] #get the first k line
    z = x.dot(u_reduce) #[m,n]*[n,k] = [m,k]
    return z
# 还原 1-->2
def recoverData(z,u,k):
    x_rec = np.zeros((z.shape[0],u.shape[0]))
    u_reduce = u[:,:k]
    x_rec = z.dot(u_reduce.T) #[m,k]*[k,n] = [m,n]
    return x_rec

Z = projectData(X_norm,U,k=1)  #n-->k
print ("(this value should be about 1.481274)")
print (Z[0])
X_rec = recoverData(Z,U,k=1) # get back k--->n
print ("(this value should be about  -1.047419 -1.047419)")
print (X_rec[0,0],X_rec[0,1])

#可视化
plt.plot(X_norm[:,0],X_norm[:,1],'bo')
plt.plot(X_rec[:,0],X_rec[:,1],'rx')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()