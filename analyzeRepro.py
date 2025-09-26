# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:35:58 2025

@author: crvan
"""

import numpy as np
import  matplotlib.pyplot as plt

results_dir = r'C:\data'

with open(results_dir + r'\repro_0\posy.csv', 'r') as f_in:
    ypos = np.loadtxt(f_in, delimiter = ',')


plt.figure(1)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Rx.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])

plt.xlabel("y position[mm]")
plt.ylabel("rx [rad]")
plt.show()

plt.figure(2)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Ry.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])

plt.xlabel("y position[mm]")
plt.ylabel("ry [rad]")
plt.show()

plt.figure(3)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Z.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])

plt.xlabel("y position[mm]")
plt.ylabel("Z [m]")
plt.show()


plt.figure(4)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Rx.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])
plt.xlim(-5e-3, 5e-3)
plt.xlabel("y position[mm]")
plt.ylabel("rx [rad]")
plt.show()

plt.figure(5)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Ry.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])
plt.xlim(-5e-3, 5e-3)
plt.xlabel("y position[mm]")
plt.ylabel("ry [rad]")
plt.show()

plt.figure(6)

data = np.zeros((ypos.shape[0], ypos.shape[1],10))
for i in range(10):
    with open(results_dir + r'\repro_%i\Z.csv' %i, 'r') as f_in:
        data[:,:,i] = np.loadtxt(f_in, delimiter = ',')
        
    plt.plot(ypos[:,0], data[:,ypos.shape[1]//2,i])
plt.xlim(-5e-3, 5e-3)
plt.xlabel("y position[mm]")
plt.ylabel("Z [m]")
plt.show()

