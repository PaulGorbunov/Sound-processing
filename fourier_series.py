'''
representing sound vawe with furier series
'''
import soundfile as sf
import math
import pylab
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
import numpy as np

import pickle

s_inter = 0
e_inter = 20

data, samplerate = sf.read('dormir.wav') 
delt = 1000*((e_inter-s_inter)//10)+1
ys = [f[0] if f[0] > f[1] else f[1] for f in data][s_inter:e_inter]
xs = [e for e in range(len(ys))]
funct = lambda x: ys[x]
a = []
b = []
for i in range(1,delt+1):
    a.append(sum([funct(d)*math.cos(i*d) for d in range(len(xs))])/math.pi)
    b.append(sum([funct(d)*math.sin(i*d) for d in range(len(xs))])/math.pi)

#fast
n_ys = []
for e in xs :
    ser  = 0
    for u in range(1,delt+1):
        ser += a[u-1]*math.cos(u*e) + b[u-1]*math.sin(u*e)
    n_ys.append(ser/(delt/math.pi))

'''
#use xes in plotting
eq  = ""
for u in range(1,delt+1):
    eq += str(a[u-1])+"*cos("+str(u)+"*x0)+"+str(b[u-1])+"*sin("+str(u)+"*x0)+"
eque = parse_expr("("+eq[:len(eq)-1]+")/"+str(delt/math.pi))
x0  = Symbol("x0")
xes = np.arange(0.,float(len(xs)-1),0.2)
n_ys = [eque.subs(x0,t).evalf() for t in xes]
'''
pylab.plot(xs,ys,"g-",xs,n_ys,"r-")
pylab.show()
