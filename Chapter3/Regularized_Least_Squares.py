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

plt.plot(x_domain, y_true, 'g')
plt.plot(x_train, y_train, 'bo')
plt.show()

## Maximum Likelihood Method
regul = 0.0005
M = 5

Psi = BasisFunction(x_train, 'polynomial', M)
w_regul = np.linalg.inv(regul*np.eye(M+1) + Psi.transpose().dot(Psi)).dot(Psi.transpose()).dot(y_train)
y_regul_pred = np.polyval(np.flipud(w_regul), x_domain)
var_regul = 1/N_train*np.sum((y_train-np.polyval(np.flipud(w_regul), x_train).transpose())**2)
sig_regul = np.sqrt(var_regul)

plt.plot(x_domain, y_true, 'g')
plt.plot(x_train, y_train, 'bo')
plt.plot(x_domain, y_regul_pred, 'r')
plt.plot(x_domain, y_regul_pred+sig_regul, 'r--', x_domain, y_regul_pred-sig_regul, 'r--')
plt.show()