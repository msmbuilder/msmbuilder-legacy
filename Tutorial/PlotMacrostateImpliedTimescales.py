#!/usr/bin/env python

from numpy import *
from matplotlib import *
from matplotlib.pyplot import *

Unit = "ps"
NumStates = 4
DataDir = "./Macro4/"
LagTime = 1

x,y = LagTime*loadtxt(DataDir+"/ImpliedTimescales.dat").transpose()

plot(x,y,'o')

title("Implied Timescales")
xlabel("Lagtime")
ylabel("Implied Timescale")

K = loadtxt(DataDir+"/Rate.dat")
lam,ev = linalg.eig(K)
lam = sort(abs(lam))
for i in range(1,NumStates):
	plot(x,LagTime/lam[i]*(x*0+1),'g-')


plot([],[],'g-',label="SCRE")
plot([],[],'bo',label="Fixed Lagtime")
xlabel("Lagtime [%s]"%Unit)
ylabel("Implied Timescale [%s]"%Unit)
legend(loc=0,numpoints=1)

axis([0,26,0,26])

show()
