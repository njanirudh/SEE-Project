#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel('Results.xlsx',stylesheet="abc")
newdata = data.as_matrix()
x = newdata[2:,0]
y = newdata[2:,1]
theta = newdata[2:,4]


#X_Right
x_mean=np.mean(x)
x_var=np.var(x)
y_mean=np.mean(y)
theta_mean=np.mean(theta)
plt.hist(x[np.newaxis].T,bins=5,density=True,color='y')
x.shape
x_std=np.sqrt(x_var)
x_right=np.linspace(x_mean-3*x_std,x_mean+3*x_std,20)
plt.plot(x_right,stats.norm.pdf(x_right,x_mean,x_std))
plt.title("X-Right")
plt.xlabel('X-Right in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#y
y_mean=np.mean(y)
y_var=np.var(y)
plt.hist(y[np.newaxis].T,bins=5,density=True,color='r')
y_std=np.sqrt(y_var)
y_left=np.linspace(y_mean-3*y_std,y_mean+3*y_std,20)
plt.plot(y_left,stats.norm.pdf(y_left,y_mean,y_std))
plt.title("Y-Right")
plt.xlabel('Y-Right in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#theta
theta_mean=np.mean(theta)
theta_var=np.var(theta)
plt.hist(theta[np.newaxis].T,bins=5,density=True, color='g')
theta_std=np.sqrt(theta_var)
theta_left=np.linspace(theta_mean-3*theta_std,theta_mean+3*theta_std,20)
plt.plot(theta_left,stats.norm.pdf(theta_left,theta_mean,theta_std))
plt.grid()
plt.xlabel('angle in degrees')
plt.ylabel('number of counts')
plt.title("Angle")
plt.show()


data = pd.read_excel('Results.xlsx',sheet_name=1,stylesheet="abc")
newdata = data.as_matrix()
x = newdata[2:,0]
y = newdata[2:,1]
theta = newdata[2:,4]



#X_Left
x_mean=np.mean(x)
x_var=np.var(x)
y_mean=np.mean(y)
theta_mean=np.mean(theta)
plt.hist(x[np.newaxis].T,bins=5,density=True,color='y')
x_std=np.sqrt(x_var)
x_right=np.linspace(x_mean-3*x_std,x_mean+3*x_std,20)
plt.plot(x_right,stats.norm.pdf(x_right,x_mean,x_std))
plt.title("X-Left")
plt.xlabel('X-Left in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#y
y_mean=np.mean(y)
y_var=np.var(y)
plt.hist(y[np.newaxis].T,bins=5,density=True,color='r')
y_std=np.sqrt(y_var)
y_left=np.linspace(y_mean-3*y_std,y_mean+3*y_std,20)
plt.plot(y_left,stats.norm.pdf(y_left,y_mean,y_std))
plt.title("Y-Left")
plt.xlabel('Y-Left in mm')
plt.ylabel('number of counts')
plt.grid()
#plt.xlim([450,600])
plt.show()
#theta
theta_mean=np.mean(theta)
theta_var=np.var(theta)
plt.hist(theta[np.newaxis].T,bins=5,density=True, color='g')
theta_std=np.sqrt(theta_var)
theta_left=np.linspace(theta_mean-3*theta_std,theta_mean+3*theta_std,20)
plt.plot(theta_left,stats.norm.pdf(theta_left,theta_mean,theta_std))
plt.grid()
plt.xlabel('angle in degrees')
plt.ylabel('number of counts')
plt.title("Angle")
plt.show()


data = pd.read_excel('Results.xlsx',sheet_name=2,stylesheet="abc")
newdata = data.as_matrix()
x = newdata[2:,0]
y = newdata[2:,1]
theta = newdata[2:,4]


#X_Start
x_mean=np.mean(x)
x_var=np.var(x)
y_mean=np.mean(y)
theta_mean=np.mean(theta)
plt.hist(x[np.newaxis].T,bins=5,density=True,color='y')
x_std=np.sqrt(x_var)
x_right=np.linspace(x_mean-3*x_std,x_mean+3*x_std,20)
plt.plot(x_right,stats.norm.pdf(x_right,x_mean,x_std))
plt.title("X-Start")
plt.xlabel('X-Start in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#y
y_mean=np.mean(y)
y_var=np.var(y)
plt.hist(y[np.newaxis].T,bins=5,density=True,color='r')
y_std=np.sqrt(y_var)
y_left=np.linspace(y_mean-3*y_std,y_mean+3*y_std,20)
plt.plot(y_left,stats.norm.pdf(y_left,y_mean,y_std))
plt.title("Y-Start")
plt.xlabel('Y-Start in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#theta
theta_mean=np.mean(theta)
theta_var=np.var(theta)
plt.hist(theta[np.newaxis].T,bins=5,density=True, color='g')
theta_std=np.sqrt(theta_var)
theta_left=np.linspace(theta_mean-3*theta_std,theta_mean+3*theta_std,20)
plt.plot(theta_left,stats.norm.pdf(theta_left,theta_mean,theta_std))
plt.grid()
plt.xlabel('angle in degrees')
plt.ylabel('number of counts')
plt.title("Angle")
plt.show()


data = pd.read_excel('Results.xlsx',sheet_name=3,stylesheet="abc")
newdata = data.as_matrix()
x = newdata[2:,0]
y = newdata[2:,1]
theta = newdata[2:,4]


#X_Straight
x_mean=np.mean(x)
x_var=np.var(x)
y_mean=np.mean(y)
theta_mean=np.mean(theta)
plt.hist(x[np.newaxis].T,bins=5,density=True,color='y')
x_std=np.sqrt(x_var)
x_right=np.linspace(x_mean-3*x_std,x_mean+3*x_std,20)
plt.plot(x_right,stats.norm.pdf(x_right,x_mean,x_std))
plt.title("X-Straight")
plt.xlabel('X-Straight in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#y
y_mean=np.mean(y)
y_var=np.var(y)
plt.hist(y[np.newaxis].T,bins=5,density=True,color='r')
y_std=np.sqrt(y_var)
y_left=np.linspace(y_mean-3*y_std,y_mean+3*y_std,20)
plt.plot(y_left,stats.norm.pdf(y_left,y_mean,y_std))
plt.title("Y-Straight")
plt.xlabel('Y-Straight in mm')
plt.ylabel('number of counts')
plt.grid()
plt.show()
#theta
theta_mean=np.mean(theta)
theta_var=np.var(theta)
plt.hist(theta[np.newaxis].T,bins=5,density=True, color='g')
theta_std=np.sqrt(theta_var)
theta_left=np.linspace(theta_mean-3*theta_std,theta_mean+3*theta_std,20)
plt.plot(theta_left,stats.norm.pdf(theta_left,theta_mean,theta_std))
plt.grid()
plt.xlabel('angle in degrees')
plt.ylabel('number of counts')
plt.title("Angle")
plt.show()
