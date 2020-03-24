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
N=2
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

Y0[0]=0.11 #Short-Rate
Y0[1]=0.054 #Mean-rate
kap[0]=0.7298
kap[1]=0.021185
mu[0]=0.04013
mu[1]=0.022543 
sig[0]=0.16885 
sig[1]=0.054415 
lam[0]=-0.01731
lam[1]=-0.044041 

delta0=0.0
deltay[0]=1.0
deltay[1]=1.0

beta[0,0]=1.0 #beta_0
beta[1,0]=0.0
beta[0,1]=0.0 #beta_1
beta[1,1]=1.0

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
ytm_exact=0.11130203518039194
means=np.zeros([0])
tmpsum=0
tmpsuma=np.zeros([nntot+1])
tmpsumb=np.zeros([nntot+1])
npatha=np.zeros([nntot+1])
svari=np.zeros([nntot+1])
svaria=np.zeros([nntot+1])
ytmi=np.zeros([nntot+1])


for counter in range(0,nntot+1): 
    if counter==-1:
        lnn=lnnmax+dlnn
    else:
        inn=counter
        lnn=lnnmin+inn*dlnn
        
    nn=int(round(10**lnn))
    print(counter, 'nn',nn)
   
    
    nsteps=800
    dt=T/nsteps
    Psum=0.

    npath=nn #No of paths
    npatha[counter]=npath
    Pi=np.zeros([0])
    ytmj=np.zeros([0])
    tmpsuma[counter]=0
    tmpsumb[counter]=0



    
    for ipath in range(npath):

        t=0.
        rint=0.
        Y=Y0.copy()
        #x1=([])
        y=[0.164]
        tmp1=[Y[0]]
        tmp2=[Y[1]]
########################################################################

        for k in range(nsteps):
            #MC on Y
            Ytrunc=np.maximum(Y,0) #Stops value from going less than 0

            for j in range(N):
                Diagrt[j,j]=np.sqrt(alpha[j]+beta[:,j].dot(Ytrunc))
            dt1=dt if T-(t+dt)>TINY else T-t #roundoff correction
            dWt=np.sqrt(dt1)*np.random.standard_normal(2) 
            dY=Kt.dot(Thetat-Ytrunc)*dt1+Sig.dot(Diagrt).dot(dWt)
            Y+=dY
#########################################################################

            #print('Y',Y)
            tmp1.append(Y[0])
            tmp2.append(Y[1])
            tmp=Y[0]+Y[1]
            if counter>=0:
                x1=np.arange(nsteps+1) #Number of steps on x-axis
                #print('x1',x1)
            y.append(tmp) #List of adjusted rates
            #print('y',y)
            #plt.plot(npath,y)
            #plt.show()
            ###########
            t+=dt1
            
            r1=delta0+deltay.dot(Y)
            rint+=r1*dt1
            #plt.plot(npath,y)
            #plt.show(
#########################################################################

###Saving Plots of Paths for Each Factor###        
        if counter>=0 and counter<=3:
            #plt.plot(x1,y)
            
            plt.figure(2)
            plt.grid(1)
            plt.plot(x1,tmp2)
            plt.title('Possible Mean Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            set_size(4.3,2)
            if ipath==npath-1: #Save figure after plotting last path
                plt.savefig('MC2.pgf')

            plt.figure(1)
            plt.grid(1)
            plt.title('Possible Short Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            plt.plot(x1,tmp1)
            set_size(4.3,2)
            if ipath==npath-1:
                plt.savefig('MC1.pgf')
        
#########################################################################
                
        P1=np.exp(-rint)
        Pi=np.append(Pi, P1) 
        #Array of bond prices from each path
        ytmj=np.append(ytmj, (-np.log(P1)/T)) 
        #Array of ytms from each bond price

        Psum+=P1

    plt.show(2) 
    Pave=Psum/npath


###Finding the Standard Deviation of BP:###
    means=np.append(means, Pave) 
    #Saving the sample mean for each iteration of npaths
    for j in range(npath):
        tmpsuma[counter]+=((Pi[j]-means[counter])**2)
    svari[counter]=(1/(npath-1))*tmpsuma[counter]
    #sample variance of BP
    
    ytmntrl=-np.log(Pave)/T

###Finding the Standard Deviation of YTM:###
    ytmi[counter]=-np.log(means[counter])/T #All YTM means
    for j in range(npath):
        tmpsumb[counter]+=(((ytmj[j])-ytmi[counter])**2) 
    svaria[counter]=(1/(npath-1))*tmpsumb[counter] #sample variance of YTM
    
#########################################################################
    if counter==-1:
        ytm_exact=ytmntrl
        #ytm_exact=0.11130203518039194 #new using ODE result
    else:
        errs[counter,:]=nn,npath,np.abs(ytmntrl-ytm_exact)
        print('nn {:d} npath {:d} ytm {:.16f} err {:.4e}'.format(nn,
              npath,ytmntrl,np.abs(ytmntrl-ytm_exact)))
        print('Paths:',npath,'Step Size:',T/nsteps,'\nSample Mean:'
              ,means[counter],'Std Err:'
              ,(np.sqrt(svari[counter])/np.sqrt(npath))
              , '\nUB Sample Variance:',svari[counter]
              ,'Standard Deviation:',np.sqrt(svari[counter]))
        print('95% Confidence Interval (BP):'
              ,means[counter]-1.96*(np.sqrt(svari[counter])/np.sqrt(npath))
              ,means[counter]+1.96*(np.sqrt(svari[counter])/np.sqrt(npath)))
        print('99% Confidence Interval (BP):'
              ,means[counter]-2.58*(np.sqrt(svari[counter])/np.sqrt(npath))
              ,means[counter]+2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))
        print('95% Confidence Interval (YTM):'
              ,-np.log(means[counter]+1.96*(np.sqrt(svari[counter])/
              np.sqrt(npath)))/T,-np.log(means[counter]-1.96
              *(np.sqrt(svari[counter])/np.sqrt(npath)))/T)
        print('99% Confidence Interval (YTM):',
              -np.log(means[counter]
              +2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))/T
              ,-np.log(means[counter]-2.58*(np.sqrt(svari[counter])
              /np.sqrt(npath)))/T, '\n')


tmpsum=0

##############################################################################

###Error Plot###
plt.ylabel('Error')
plt.xlabel('Number of Paths')
plt.grid(1)
plt.loglog(errs[:,0],errs[:,2],'bo-')
plt.loglog(errs[:,0],5e-1*errs[:,0]**-1,'--')
plt.loglog(errs[:,0],1e-1*errs[:,0]**-0.5,'--')
plt.savefig('Error-NA.pgf')









