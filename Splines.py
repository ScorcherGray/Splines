#Homework 7
import numpy as np
import matplotlib.pyplot as mpl
t = [0.9, 1.3, 1.9, 2.1] #The starting data points
y = [1.3, 1.5, 1.85, 2.1]
#Number 2: The matrix for a polynomial
# Given n data points the matrix must be nxn
M = [[t[b]**n for n in range(len(t))] for b in range(len(t))]
Minv = np.linalg.inv(M) #Inverse of matrix M
Y = np.matmul(Minv,y) #The list of coefficients is the matrix multiplication of M inverse with y
# The points to plot are then the matrix product of M * Y
P = np.matmul(M, Y)
mpl.plot(t,P, 'bo')
mpl.plot(t,y, 'rv')
mpl.show()
#Number 3: Do the splines
h = np.zeros(len(t)-1) #prepping h and b equations
b = np.zeros(len(t)-1)
for i in range(len(t)-1): #Create the h and b equation values
    h[i] = t[i+1] - t[i]
    b[i] = 1/h[i] * (y[i+1]-y[i])

v = np.zeros(len(t) - 1) #Prep  v and u equations
u = np.zeros(len(t) - 1)
for i in range(1, len(t)-1): #Create v and u values
    v[i] = 2*(h[i-1] + h[i])
    u[i] = 6*(b[i] - b[i-1])

A = [[v[1], h[1]], [h[1], v[2]]] #The tri-diagonal matrix.
#print (A)
inv = np.linalg.inv(A) #Inverse of matrix A
#print(inv, np.dot(inv, A))
U = [[u[b]] for b in range(1, len(u))]  #Matrix U made up of u values
Z = np.matmul(inv, U) #Matrix Z of z values needed for splines
z = np.zeros(len(t))
for i in range(1, len(t)-1):
    z[i] = Z[i-1]
mpl.figure(1)
mpl.plot(t,y, 'ro')
#print(Z)
S = []
X = []
step = 0.01 #Larger step sizes causes overlap with the points
for i in range(len(t)-1):
    x = np.arange(t[i], t[i+1] + step, step)
    temp = (z[i+1]/(6*h[i])) * (x - t[i])**3 + (z[i]/(6*h[i]))*(t[i+1] - x)**3 + \
    (y[i+1]/h[i] - (z[i+1]/6)*h[i]) * (x - t[i]) + (y[i]/h[i] - (h[i]/6)*z[i])*(t[i+1] - x)
    mpl.plot(x, temp)

mpl.show()