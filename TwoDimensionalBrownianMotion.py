# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:13:32 2020

@author: Enda
Two-Dimensional Brownian Motion
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update(matplotlib.rcParamsDefault)
rcParams.update({'font.size': 8})


csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Helvetica'}
#%matplotlib inline


n = 5000
x = np.cumsum(np.random.randn(n))
y = np.cumsum(np.random.randn(n))

# We add 10 intermediary points between two
# successive points. We interpolate x and y.
k = 10
x2 = np.interp(np.arange(n * k), np.arange(n) * k, x)
y2 = np.interp(np.arange(n * k), np.arange(n) * k, y)
fig, ax = plt.subplots(1, 1, figsize=(4,3))
# Now, we draw our points with a gradient of colors.
ax.scatter(x2, y2, c=range(n * k), linewidths=0,
           marker='o', s=3, cmap=plt.cm.jet,)
#ax.grid()
ax.axis('equal')
#ax.set_xlabel('Number of Steps', **hfont)
set_size(4,3)
plt.savefig('BM2.pdf')
plt.show()
