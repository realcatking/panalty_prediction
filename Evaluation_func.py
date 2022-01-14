import math
from math import log

def term_score(targ,pred,term_num):
    s = 0
    i = 0
    for t in targ:
        p = pred[i]
        a = abs(t-p)
        if(a==0):
            s = s+1
        if(0<a<=1):
            s = s + 0.75
        if(1<a<=2):
            s = s+0.5
        if(2<a<=3):
            s = s + 0.25
        if (a>3):
            s=s
        i = i+1
    s = s/i
    return s


def Accuarcy_zero(targ,pred):
    num = 0
    i = 0
    for t in targ:
        p = pred[i]
        if t==p:
            num+=1
        i+=1
    return num/i


def Accuarcy_one(targ,pred):
    num = 0
    i = 0
    for t in targ:
        p = pred[i]
        if t-1<=p<=t+1:
            num += 1
        i += 1
    return num/i