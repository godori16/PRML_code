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
M = 5
Psi = BasisFunction(x_train, 'polynomial', M)
Psi.shape

w_ML = np.linalg.inv(Psi.transpose().dot(Psi)).dot(Psi.transpose()).dot(y_train)
y_MLpred = np.polyval(np.flipud(w_ML), x_domain)
var_ML = 1/N_train*np.sum((y_train-np.polyval(np.flipud(w_ML), x_train).transpose())**2)
sig_ML = np.sqrt(var_ML)

plt.plot(x_domain, y_true, 'g')
plt.plot(x_train, y_train, 'bo')
plt.plot(x_domain, y_MLpred, 'r')
plt.plot(x_domain, y_MLpred+sig_ML, 'r--', x_domain, y_MLpred-sig_ML, 'r--')
plt.show()