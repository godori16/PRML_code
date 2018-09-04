# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:36:47 2018

@author: TJ
"""


import numpy as np
import matplotlib.pyplot as plt
from BasisFunction import BasisFunction

# Sample Generation
x_domain = np.linspace(0, 1, 101)
y_true = np.sin(2*np.pi*x_domain)

N_train = 10
x_train = np.linspace(0,1,N_train)
sig = 0.15
y_train = np.sin(2*np.pi*x_train) + sig*np.random.normal(size=N_train)


## Maximum Likelihood Method
M = 9
beta = 1           # beta is the inverse variance, called precision, of target value
alpha = 0.1        # alpha is the precision of parameters

Psi = BasisFunction(x_train, 'polynomial', M)
Dim = Psi.shape[1]
w_Bay_cov = np.linalg.inv(alpha*np.eye(M+1) + beta*Psi.transpose().dot(Psi))
w_Bay_mean = beta*w_Bay_cov.dot(Psi.transpose()).dot(y_train)

beta_new = beta +1
alpha_new = 0
i=1
toler = 0.0001
while np.abs(beta-beta_new) > toler:
    if i != 1:
        beta = beta_new
        alpha = alpha_new
    
    lamda, v = np.linalg.eig(beta*Psi.transpose().dot(Psi))
    gamma = np.sum( lamda / (alpha + lamda) )
    
    alpha_new = gamma/(w_Bay_mean.transpose().dot(w_Bay_mean))
    beta_new = 1/(1/(N_train-gamma)*sum((y_train-Psi.dot(w_Bay_mean))**2))
    
    w_Bay_cov = np.linalg.inv(alpha_new*np.eye(Dim) + beta_new*Psi.transpose().dot(Psi))
    w_Bay_mean = beta_new*w_Bay_cov.dot(Psi.transpose()).dot(y_train)
    
    i += 1

# Predictive distribution
Psi_pred = BasisFunction(x_domain, 'polynomial', M)
y_Bay_pred = Psi_pred.dot(w_Bay_mean)
y_Bay_var = np.diag(1/beta + Psi_pred.dot(w_Bay_cov).dot(Psi_pred.transpose()))
y_Bay_sig = np.sqrt(y_Bay_var)

plt.plot(x_domain, y_true, 'g')
plt.plot(x_train, y_train, 'bo')
plt.plot(x_domain, y_Bay_pred, 'r')
plt.plot(x_domain, y_Bay_pred + y_Bay_sig, 'r--', x_domain, y_Bay_pred - y_Bay_sig, 'r--')
plt.show()