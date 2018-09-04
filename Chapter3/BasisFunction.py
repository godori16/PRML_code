# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:47:16 2018

@author: TJ
"""

import numpy.matlib as matlib
import numpy as np

def BasisFunction(x, basis_name='polynomial', *options):
    if basis_name == 'polynomial':
        basis_function = np.power(matlib.repmat(x, options[0]+1, 1).transpose(),
                                  matlib.repmat(np.arange(0, options[0]+1), x.shape[0], 1)
                                  )
    return basis_function