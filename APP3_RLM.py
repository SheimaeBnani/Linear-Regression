# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 12:10:05 2021

@author: PC
"""

import numpy as np 
import math
import scipy.stats
from scipy.stats import t
from scipy import sqrt
from statistics import variance, mean

X = np.array([
            [1, 100, 100],
            [1, 104, 99],
            [1, 106, 110],
            [1, 111, 126],
            [1, 111, 113],
            [1, 115, 103],
            [1, 120, 102],
            [1, 124, 103],
            [1, 126, 98]
            ]
            )
Y = np.array([
            [100],
            [106],
            [107],
            [120],
            [111],
            [116],
            [123],
            [133],
            [137]
            ]
            )
print ("Transposée de X : ")
print(np.transpose(X))

print ("La matrice TX : ")
TX = np.dot(np.transpose(X), X)
print('TX = \n', TX)

print ("Anverse de TX : ")
print (np.linalg.inv(TX))

print ("La matrice TY : ")
TY = np.dot(np.transpose(X), Y)
print('TY = \n', TY)

print("L'estimation ponctuelle β : ")
β = np.dot(np.linalg.inv(TX), TY)
print('β = \n', β)

print ("L'estimateur de Y : ")
Y_1 = np.dot(X, β)
print('Y_1 = \n', Y_1)
print("Le résidu : ")
ε = Y - Y_1
print('ε = \n', ε)

print("L'estimation de variance des residus : ")
ε_2 = [i**2 for i in ε]    
print(sum(ε_2)/6)

print("Test global de signification du modèle au risque α = 5% : ")
SCE = 0
for i in range(9):
    SCE += (Y_1[i]  - Y.mean())**2
print("SCE = ", SCE)

SCR = sum(ε_2)
print("SCR = ", SCR)

n = 9
p = 2
F_obs = (SCE/p)/(SCR/(n-p-1))
print("F_obs = ", F_obs)


