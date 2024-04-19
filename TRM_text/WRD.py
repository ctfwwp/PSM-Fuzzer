#word Rotator's Distance(W2X)
##WRD由上下界
import numpy as np
from scipy.optimize import linprog
key = {'0': -8, '1': -7, '2': -6, '3': -5, '4': -4, '5': -3, '6': -2, '7': -1, '8': 1, '9': 2, 'a':3, 'b':4, 'c':5,
       'd':6, 'e': 7, 'f': 8}
def wasserstein_distance(p, q, D):
    '''
    通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shap e=[m,n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    '''
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = np.array(D)
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun
def word_rotator_distance(x, y):
    """W2X（Word Rotator's Distance）的参考实现
    """
    x, y = np.array(x), np.array(y)
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)

def word_rotator_similarity(x, y):
    """1 - W2X
    x.shape=[m,d], y.shape=[n,d]
    """
    return 1 - word_rotator_distance(x, y)
def change_data(x,y):
         str0 = []
         l = len(x)
         for i in range(0, l, 2):
            str1 = []
            str1.append(key[x[i]])
            str1.append(key[x[i + 1]])
            str0.append(str1)

         str2 = []
         l = len(y)
         for i in range(0, l, 2):
            str3 = []
            str3.append(key[y[i]])
            str3.append(key[y[i + 1]])
            str2.append(str3)
         return str0,str2
def similarity(a,b):
    x,y= change_data(a,b)
    return word_rotator_similarity(x,y)
