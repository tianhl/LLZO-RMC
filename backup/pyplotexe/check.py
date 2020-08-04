# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:32:59 2020

@author: blues
"""

import pandas as pd
newdata = pd.read_csv('new_293K.txt',header=None, sep='\s+')
olddata = pd.read_csv('old_293K.txt',header=None, sep='\s+')
xdiff = olddata[3]-newdata[3]
xdiff[xdiff>0.5]=xdiff-1
xdiff[xdiff<-0.5]=xdiff+1

ydiff = olddata[4]-newdata[4]
ydiff[ydiff>0.5]=ydiff-1
ydiff[ydiff<-0.5]=ydiff+1

zdiff = olddata[5]-newdata[5]
zdiff[zdiff>0.5]=zdiff-1
zdiff[zdiff<-0.5]=zdiff+1

import matplotlib.pyplot as plt
plt.xticks(fontsize=75)
plt.yticks(fontsize=75)
plt.hist(xdiff)
plt.hist(ydiff)
plt.hist(zdiff)
