# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:51:45 2019

@author: Enda
"""

from matplotlib import pyplot as plt
import math as m
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

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


csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Helvetica'}

TINY=1e-12

#Parameters
N=3
method='Euler'

#Global Data Structures

#Main
K=np.zeros([N,N])
Kt=np.zeros([N,N])
beta=np.zeros([N,N])
Sig=np.zeros([N,N])
Diagrt=np.zeros([N,N])

Y0=np.zeros([N])

Theta=np.zeros([N])
Thetat=np.zeros([N])
alpha=np.zeros([N])
kap=np.zeros([N])
mu=np.zeros([N])
sig=np.zeros([N])
lam=np.zeros([N])
deltay=np.zeros([N])
r=np.zeros([N])
Y=np.zeros([N])

fntrl=np.zeros([N+1])
yntrl=np.zeros([N+1])

#ntrlfunc
KtTB=np.zeros([N])
SigTB=np.zeros([N])
SigTBsq=np.zeros([N])


T=8.0 #Bond Maturity


#Double-Check the initial values of the state variables
Y0[0]=0.11   #Short Rate
Y0[1]=0.039   #Mean Rate (?)
Y0[2]=0.015  #Volatility (?)
kap[0]=1.4298
kap[1]=0.01694
kap[2]=0.0351
mu[0]=0.04374
mu[1]=0.00253
mu[2]=0.003209
sig[0]=0.16049 
sig[1]=0.1054 
sig[2]=0.0496
lam[0]=-0.2468
lam[1]=0.03411
lam[2]=-0.1569

delta0=0.0
deltay[0]=1.0
deltay[1]=1.0
deltay[2]=1.0

beta[0,0]=1.0   #  beta_0
beta[1,0]=0.0
beta[2,0]=0.0
beta[0,1]=0.0   #  beta_1
beta[1,1]=1.0
beta[2,1]=0.0
beta[0,2]=0.0   # beta_2
beta[1,2]=0.0
beta[2,2]=1.0

for i in range(N):
    K[i,i]=kap[i]
    Sig[i,i]=sig[i]
    Kt[i,i]=kap[i]+lam[i]
    Theta[i]=mu[i]
    Thetat[i]=Theta[i]*K[i,i]/Kt[i,i]
    
nntot=10 #number of logarithmically spaced resolutions
nnmax=1000 #max number of points
nnmin=100 #min number of points
lnnmax=np.log10(nnmax)
lnnmin=np.log10(nnmin)
dlnn=(lnnmax-lnnmin)/nntot
errs=np.zeros([nntot+1,3])
#ytm_exact=0.097471410 #Default for T=10
ytm_exact=0.11281821728731579
means=np.zeros([0])
tmpsum=0
tmpsuma=np.zeros([nntot+1])
npatha=np.zeros([nntot+1])
svari=np.zeros([nntot+1])



for counter in range(0,nntot+1):
#for counter in range(1):
    if counter==-1:
        lnn=lnnmax+dlnn
    else:
        inn=counter
        lnn=lnnmin+inn*dlnn
        
    nn=int(round(10**lnn))
    print(counter, 'nn',nn)
    
    dt=T/nn
    Psum=0.
    
    npath=nn #No of paths
    npatha[counter]=npath
    Pi=np.zeros([0])
    tmpsuma[counter]=0

    
    for ipath in range(npath):
        #if ipath%(int(round(npath/10)))==0:
            #print(ipath)
        t=0.
        rint=0.
        Y=Y0.copy()
        #x1=([])
        y=[0.164] #Sum of initial state vars
        tmp1=[Y[0]]
        tmp2=[Y[1]]
        tmp3=[Y[2]]
########################################################################

        for k in range(nn):
            #MC on Y
            Ytrunc=np.maximum(Y,0)

            for j in range(N):
                Diagrt[j,j]=np.sqrt(alpha[j]+beta[:,j].dot(Ytrunc))
            dt1=dt if T-(t+dt)>TINY else T-t #roundoff correction
            dWt=np.sqrt(dt1)*np.random.standard_normal(3) 
            dY=Kt.dot(Thetat-Ytrunc)*dt1+Sig.dot(Diagrt).dot(dWt)
            #print('dY',dY)
            Y+=dY
########################################################################

            #print('Y',Y)
            tmp1.append(Y[0])
            tmp2.append(Y[1])
            tmp3.append(Y[2])
            tmp=Y[0]+Y[1]+Y[2]
            if counter>=0:
                x1=np.arange(nn+1) #Number of steps on x-axis
   
            t+=dt1
            
            r1=delta0+deltay.dot(Y)
            rint+=r1*dt1

########################################################################

###Saving Plots of Paths for Each Factor###        
        if counter>=0:
            plt.figure(1)
            plt.grid(1)
            plt.title('Possible Short Rate Paths',**hfont)
            plt.xlabel('Steps',**hfont)
            plt.plot(x1,tmp1)
            set_size(4.3,2)
            if ipath==npath-1:
                plt.savefig('figure01a.pgf')
                
            plt.figure(2)
            plt.grid(1)
            plt.plot(x1,tmp2)
            plt.title('Possible Mean Rate Paths',**hfont)
            plt.xlabel('Steps',**hfont)
            set_size(4.3,2)
            if ipath==npath-1: #Save figure after plotting last path
                plt.savefig('figure02a.pgf')

            plt.figure(3)
            plt.grid(1)
            plt.plot(x1,tmp3)
            plt.title('Possible Volatility Paths',**hfont)
            plt.xlabel('Steps',**hfont)
            set_size(4.3,2)
            if ipath==npath-1: #Save figure after plotting last path
                plt.savefig('figure03a.pgf')
                
########################################################################
           
            
        P1=np.exp(-rint)
        Pi=np.append(Pi, P1) #Array of bond prices from each path
        Psum+=P1 
        
    plt.show()
    Pave=Psum/npath
    
###Finding the Standard Deviation:###
    means=np.append(means, Pave) 
    #Saving the sample mean for each iteration of npaths
    for j in range(npath):
        tmpsuma[counter]+=((Pi[j]-means[counter])**2)
    svari[counter]=(1/(npath-1))*tmpsuma[counter]
    
    ytmntrl=-np.log(Pave)/T
    
    if counter==-1:
       ytm_exact=0.12078385242439155
    else:
        errs[counter,:]=nn,npath,np.abs(ytmntrl-ytm_exact)
        print('nn {:d} npath {:d} ytm {:.16f} err {:.4e}'
              .format(nn,npath,ytmntrl,np.abs(ytmntrl-ytm_exact)))
        print('UB Sample Variance:',svari[counter],'Paths:',npath
              ,'Std Err:',(np.sqrt(svari[counter])/np.sqrt(npath))
              ,'\nSample Mean:',means[counter]
              , 'Standard Deviation:',np.sqrt(svari[counter]))
        print('95% Confidence Interval (BP):'
              ,means[counter]-1.96*(np.sqrt(svari[counter])/np.sqrt(npath))
              ,means[counter]+1.96*(np.sqrt(svari[counter])/np.sqrt(npath)))
        print('99% Confidence Interval (BP):'
              ,means[counter]-2.58*(np.sqrt(svari[counter])/np.sqrt(npath))
              ,means[counter]+2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))
        print('95% Confidence Interval (YTM):'
              ,-np.log(means[counter]
                       +1.96*(np.sqrt(svari[counter])/np.sqrt(npath)))/T
              ,-np.log(means[counter]
                       -1.96*(np.sqrt(svari[counter])/np.sqrt(npath)))/T)
        print('99% Confidence Interval (YTM):'
              ,-np.log(means[counter]
                       +2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))/T
              ,-np.log(means[counter]
                       -2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))/T, '\n')
    
plt.ylabel('error')
plt.xlabel('nn')
plt.loglog(errs[:,0],errs[:,2],'bo-')
plt.loglog(errs[:,0],5e-1*errs[:,0]**-1,'--')
plt.loglog(errs[:,0],1e-1*errs[:,0]**-0.5,'--')
