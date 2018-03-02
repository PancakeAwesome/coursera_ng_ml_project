# 本文是绘制学习曲线，学会误差分析,从而进行模型诊断和选择
# Loading and Visualizing Data
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import optimize

data = sio.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
print(X.shape, Xval.shape, Xtest.shape)

plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# 正则化用于线性回归


def linearRegCostFunction(X, y, theta, reg):
    J = 0
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_1 = theta[1:, :]
    grad = np.zeros_like(theta)
    J = 0.5 * np.sum((X.dot(theta) - y)**2) / m + \
        reg * 0.5 * np.sum(theta_1**2) / m
    grad = X.T.dot(X.dot(theta) - y) / m
    grad[1:, :] += reg * theta_1 / m
    return J, grad
m = X.shape[0]
XX = np.hstack((np.ones((m, 1)), X))
XXval = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
XXtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))
# 测试一下
init_theta = np.array([[1], [1]])
J, grad = linearRegCostFunction(XX, y, init_theta, 1.0)
print('Cost at theta = [1 ; 1]: (this value should be about 303.993192)')
print(J)
print ('Gradient at theta = [1 ; 1](this\
 value should be about [-15.303016; 598.250744])')
print(grad)

# 拟合参数，最优化


def f(params, *args):
    X, y, reg = args
    m, n = X.shape
    J = 0
    theta = params.reshape((n, 1))
    theta_1 = theta[1:, :]
    J = 0.5 * np.sum((X.dot(theta) - y)**2) / m + \
        reg * 0.5 * np.sum(theta_1**2) / m
    return J


def gradf(params, *args):
    X, y, reg = args
    m, n = X.shape
    theta = params.reshape((n, 1))
    theta_1 = theta[1:, :]
    grad = np.zeros_like(theta)
    grad = X.T.dot(X.dot(theta) - y) / m
    grad[1:, :] += reg * theta_1 / m
    g = grad.ravel()
    return g


# Train linear regression with lambda = 0
# 用优化算法去训练
def train(X, y, reg):
    args = (X, y, reg)
    init_theta = np.zeros((X.shape[1], 1))
    params = init_theta.ravel()
    res = optimize.fmin_cg(f, x0=params, fprime=gradf, args=args, maxiter=500)
    return res
res = train(XX, y, 0)
print('result = ', res)
# 可视化一下训练出来的参数res,也即theta权重参数
# 可以看出效果一般，因为是一个直线
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.plot(X, XX.dot(res), '-')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# 定义一下学习曲线，观察数据集大小m的变化对训练误差和验证误差的影响


def learnCurve(X, y, Xval, yval, reg):
    m = X.shape[0]
    err_train = []
    err_val = []
    for i in range(m):  # increase the data size :1--->m(m=12)
        best_theta = train(X[:i + 1, :], y[0:i + 1], reg)
        err_t, g1 = linearRegCostFunction(
            X[0:i + 1, :], y[0:i + 1], best_theta, reg)
        err_v, g2 = linearRegCostFunction(Xval, yval, best_theta, reg)
        err_train.append(err_t)
        err_val.append(err_v)
    return err_train, err_val

err_train, err_val = learnCurve(XX, y, XXval, yval, 0)
print(err_train[:5])
print(err_val[:5])
plt.plot(err_train, 'b', linestyle='-', label='err_train')
plt.plot(err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper left')
plt.show()

# 从上面的图形看出，模型属于“高偏差”，那么增大数据集没有用，只能够增加模型复杂度，改进模型
# 所以增加多项式或者高次项特征


def ployFeatures(X, p=8):
    X_poly = np.zeros((X.shape[0], p))
    '''X is a vector'''
    for i in range(p):
        X_poly[:, i] = X.T**(i + 1)
    return X_poly

# 因为有高次项，所以进行特征均值归一化，进行特征缩放


def featureNormalize(x):
    mu = np.mean(x, axis=0)
    xx = x - mu
    sigma = np.std(x, axis=0)
    x_norm = xx / sigma
    return x_norm, mu, sigma

X_ploy = ployFeatures(X, p=8)
X_ploy, mu, sigma = featureNormalize(X_ploy)
X_ploy = np.hstack((np.ones((X_ploy.shape[0], 1)), X_ploy))
print(mu)
print(sigma)
print(X_ploy[1, :])

X_ploy_test = ployFeatures(Xtest, p=8)
X_ploy_test = (X_ploy_test - mu) / sigma  # 都是用训练集的mu,和sigma去缩放
X_ploy_test = np.hstack((np.ones((Xtest.shape[0], 1)), X_ploy_test))

X_ploy_val = ployFeatures(Xval, p=8)
X_ploy_val = (X_ploy_val - mu) / sigma
X_ploy_val = np.hstack((np.ones((Xval.shape[0], 1)), X_ploy_val))

print(X_ploy.shape, X_ploy_test.shape, X_ploy_val.shape)

# 正则化强度=0,等于没有用正则化，容易过拟合
res1 = train(X_ploy, y, 0)
print(res1)
# 我们可视化一下拟合的曲线,会发现过拟合了，所以需要选择合适的reg,正则化强度这个超参数


def plotFit(mu, sigma, theta, p):
    x = np.linspace(-50, 45).reshape(-1, 1)
    x_ploy = ployFeatures(x, p)
    x_ploy = x_ploy - mu
    x_ploy = x_ploy / sigma
    x_ploy = np.hstack((np.ones((x.shape[0], 1)), x_ploy))
    plt.plot(x, x_ploy.dot(theta), '--', color='black')

    pass
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(mu, sigma, res1, p=8)

plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

# 在这之前还是绘制一下这个模型的学习曲线，看看模型方差和偏差怎么样
err_train, err_val = learnCurve(X_ploy, y, X_ploy_val, yval, 0)
print(err_train[:5])
print(err_val[:5])

plt.plot(err_train, 'b', linestyle='-', label='err_train')
plt.plot(err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('Number of training examples(m)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper left')
plt.show()

# 从上图可以看出，模型属于高方差，过拟合了，所以需要加入正则化,reg就不能等于0了
# 选择多少合适，就需要交叉验证来决定了


def validationCurve(X, y, Xval, yval):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1,
                  0.3, 1.0, 3.0, 10.0]  # 1,3,10-->1.0,3.0,10.0
    err_train = []
    err_val = []
    for reg in lambda_vec:  # choice the reg
        best_theta = train(X, y, reg)
        err_t, g1 = linearRegCostFunction(X, y, best_theta, reg)
        err_v, g2 = linearRegCostFunction(Xval, yval, best_theta, reg)
        err_train.append(err_t)
        err_val.append(err_v)
    return lambda_vec, err_train, err_val

lambda_vec, err_train, err_val = validationCurve(X_ploy, y, X_ploy_val, yval)
plt.plot(lambda_vec, err_train, 'b', linestyle='-', label='err_train')
plt.plot(lambda_vec, err_val, 'r', linestyle='-', label='err_val')
plt.xlabel('the lamda(reg)')
plt.ylabel('error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper left')
plt.show()

# 打印对比一下err_train和err_val
print('reg\terr_train\terr_val')
for i in range(len(lambda_vec)):
    print('%f\t%f\t%f' % (lambda_vec[i], err_train[i], err_val[i]))
# so good reg is 0.3,结果会平滑一点，不会太过拟合
res1 = train(X_ploy, y, reg=0.3)
print(res1)
# 可视化一下拟合的曲面,发现更加平滑了，不会过拟合


def plotFit(mu, sigma, theta, p):
    x = np.linspace(-50, 45).reshape(-1, 1)
    x_ploy = ployFeatures(x, p)
    x_ploy = x_ploy - mu
    x_ploy = x_ploy / sigma
    x_ploy = np.hstack((np.ones((x.shape[0], 1)), x_ploy))
    plt.plot(x, x_ploy.dot(theta), '--', color='black')

    pass

plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(mu, sigma, res1, p=8)

plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
