# 本文实现逻辑回归
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 从txt文件中导入数据
def load_data(filename):
    data = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(',')
        col_num = len(lineArr)
        temp = []
        for i in range(col_num):
            temp.append(float(lineArr[i]))
        data.append(temp)
    return np.array(data)

data = load_data('ex2data1.txt')
print(data.shape)
print(data[:5])

X = data[:, :-1]
y = data[:, -1:]
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

# 可视化数据集
label0 = np.where(y.ravel() == 0)
plt.scatter(X[label0,0],X[label0,1],marker='x',color = 'r',label = 'Not admitted')
label1 = np.where(y.ravel() == 1)
plt.scatter(X[label1,0],X[label1,1],marker='o',color = 'b',label = 'Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc = 'upper left')
# plt.show()

# 计算cost和梯度
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def out(x, w):
    return sigmoid(np.dot(x, w)) # 这里的theta是列向量
def compute_cost(X_train, y_train, theta):
    m = X_train.shape[0]
    J = 0
    theta = theta.reshape(-1,1) # 将theta转换成列向量
    grad = np.zeros((X_train.shape[1], 1))
    h = out(X_train, theta)
    J = -1 * np.sum(y_train * np.log(h) + (1- y_train) * np.log(1 - h)) / m 
    grad = X_train.T.dot(h - y_train) / m
    grad = grad.ravel() # 将grad列向量摊平成array

    return J,grad

# 测试
m = X.shape[0]
one = np.ones((m, 1))
X = np.hstack((one, data[:, :-1]))
W = np.zeros((X.shape[1],1))

cost,grad = compute_cost(X, y, W)
print ('compute with w=[0,0,0]')
print ('Expected cost (approx):0.693...')
print (cost)
print ('Expected gradients (approx):[-0.1,-12,-11]')
print (grad)

cost1,grad1 = compute_cost(X,y,np.array([[-24],[0.2],[0.2]]))
print ('compute with w=[-24,0.2,0.2]')
print ('Expected cost (approx):0.218....')
print (cost1)
print ('Expected gradients (approx): [0.04,2.566,0.646]')
print (grad1)

#这里使用了最优算法，不是选择梯度下降法，例如你可以选择BFGS等
params = np.zeros((X.shape[1],1)).ravel()
args = (X,y)
def f(params,*args):
    X_train,y_train = args
    m,n = X_train.shape
    J = 0
    theta = params.reshape((n,1))
    h = out(X_train,theta)
    J = -1*np.sum(y_train*np.log(h) + (1-y_train)*np.log((1-h))) / m
    
    return J

def gradf(params,*args):
    X_train,y_train = args
    m,n = X_train.shape
    theta = params.reshape(-1,1)
    h = out(X_train,theta)
    grad = np.zeros((X_train.shape[1],1))
    grad = X_train.T.dot((h-y_train)) / m
    g = grad.ravel()
    return g

#res = optimize.minimize(f,x0=init_theta,args=args,method='BFGS',jac=gradf,\
#                        options={'gtol': 1e-6, 'disp': True})
res = optimize.fmin_cg(f,x0=params,fprime=gradf,args=args,maxiter=500)
print(res)

#可视化一下线性的决策边界
label = np.array(y)
index_0 = np.where(label.ravel()==0)
plt.scatter(X[index_0,1],X[index_0,2],marker='x'\
            ,color = 'b',label = 'Not admitted',s = 15)
index_1 =np.where(label.ravel()==1)
plt.scatter(X[index_1,1],X[index_1,2],marker='o',\
            color = 'r',label = 'Admitted',s = 15)

#show the decision boundary
x1 = np.arange(20,100,0.5)
x2 = (- res[0] - res[1]*x1) / res[2]
plt.plot(x1,x2,color = 'black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')
# plt.show()

# 预测
def predict(X, theta):
    h = out(X, theta)
    y_predict = np.where(h >=0.5, 1.0, 0)

    return y_predict

# test
prob = out(np.array([[1,45,85]]), res)
print( "For a student with scores 45 and \
85, we predict an admission ")
print ("Expected value: 0.775 +/- 0.002")

print (prob)

p = predict(X,res)
print ("Expected accuracy (approx): 89.0")
print (np.mean(p==y.ravel()))