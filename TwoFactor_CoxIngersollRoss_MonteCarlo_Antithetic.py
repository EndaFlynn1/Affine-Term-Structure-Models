# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:51:45 2019

@author: Enda
"""

from matplotlib import pyplot as plt
import math as m
import numpy as np
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
Diagrt1=np.zeros([N,N])
Diagrt2=np.zeros([N,N])



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
Y1=np.zeros([N])
Y2=np.zeros([N])


fntrl=np.zeros([N+1])
yntrl=np.zeros([N+1])

#ntrlfunc
KtTB=np.zeros([N])
SigTB=np.zeros([N])
SigTBsq=np.zeros([N])

T=8 #Bond Maturity

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

#ytm_exact=9.788581e-02 ##not used

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
nnmax=500 #max number of paths
nnmin=5 #min number of paths

lnnmax=np.log10(nnmax)
lnnmin=np.log10(nnmin)
dlnn=(lnnmax-lnnmin)/nntot
errs=np.zeros([nntot+1,3])
ytm_exact=0.11130203518039194 #exact ODE solution 

means=np.zeros([0])
meansna=np.zeros([0])
meansa=np.zeros([0])

tmpsum=0
tmpsuma=np.zeros([nntot+1])
tmpsumb=np.zeros([nntot+1])
tmpsum1=np.zeros([nntot+1])
tmpsum2=np.zeros([nntot+1])
npatha=np.zeros([nntot+1])
svari=np.zeros([nntot+1])
svari1=np.zeros([nntot+1])
svari2=np.zeros([nntot+1])
svaria=np.zeros([nntot+1])

covar=np.zeros([nntot+1])
corr=np.zeros([nntot+1])

ytmi=np.zeros([nntot+1])


for counter in range(0,nntot+1): 
    #change back to -1 for MC 'exact' solution
    if counter==-1:
        lnn=lnnmax+dlnn
    else:
        inn=counter
        lnn=lnnmin+inn*dlnn
        
    nn=int(round(10**lnn))
    print(counter, 'nn',nn)
    
    nstep=800 #no. of steps here
    dt=T/nstep
    Psum=0.
    Psum1=0.
    Psum2=0.

    Psuma=0.

    npath=nn #No of paths
    npatha[counter]=npath

    Pi=np.zeros([0])
    Pia=np.zeros([0])
    Pib=np.zeros([0])
    Pic=np.zeros([0])


    ytmj=np.zeros([0])
    tmpsuma[counter]=0
    tmpsumb[counter]=0
    tmpsum1[counter]=0
    tmpsum2[counter]=0
    
    for ipath in range(npath):

        t=0.
        rint=0.
        rint1=0.
        rint2=0.

        Y1=Y0.copy()
        Y2=Y0.copy()

        y=[0.164]
        tmp1=[Y1[0]] #Factor 1 of Non-Antithetic Sample
        tmp2=[Y1[1]] #Factor 2 of Non-Antithetic Sample
        tmp1a=[Y2[0]] #Factor 1 of Antithetic Sample
        tmp2a=[Y2[1]] #Factor 2 of Antithetic Sample

########################################################################
########(8)
        for k in range(nstep):
            #MC on Y1 and Y2
            Ytrunc1=np.maximum(Y1,0) #Stops value from going less than 0
            Ytrunc2=np.maximum(Y2,0) #Stops value from going less than 0

            for j in range(N):
                Diagrt1[j,j]=np.sqrt(alpha[j]+beta[:,j].dot(Ytrunc1))
                Diagrt2[j,j]=np.sqrt(alpha[j]+beta[:,j].dot(Ytrunc2))

            dt1=dt if T-(t+dt)>TINY else T-t #roundoff correction
            dWt1=np.sqrt(dt1)*np.random.standard_normal(2) 
            dWt2=-dWt1 #Antithetic sample            
            dY1=Kt.dot(Thetat-Ytrunc1)*dt1+Sig.dot(Diagrt1).dot(dWt1)
            Y1+=dY1
            
            
            dY2=Kt.dot(Thetat-Ytrunc2)*dt1+Sig.dot(Diagrt2).dot(dWt2) 
            #antithetic 
            Y2+=dY2

            
########################################################################
############(12) (nn loop)
            tmp1.append(Y1[0]) #list of short rates at each time step
            tmp2.append(Y1[1])
            tmp1a.append(Y2[0]) 
            #list of short rates at each time step (antithetic)
            tmp2a.append(Y2[1]) 


            tmp=Y1[0]+Y1[1]
            if counter>=0:
                x1=np.arange(nstep+1) #Number of steps on x-axis
                #print('x1',x1)
            t+=dt1
            
            r1=delta0+deltay.dot(Y1)
            rint1+=r1*dt1
            
                        
            r2=delta0+deltay.dot(Y2)
            rint2+=r2*dt1


########################################################################
########(npath loop)
###Saving Plots of Paths for Each Factor###        
        if counter==0: #Only saving plots from first iteration
            plt.figure(1)
            plt.grid(1)
            plt.title('Possible Short Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            plt.plot(x1,tmp1)
            set_size(4.3,2)
            if ipath==npath-1:
                plt.savefig('MC1.pgf')
            
            plt.figure(2)
            plt.grid(1)
            plt.plot(x1,tmp2)
            plt.title('Possible Mean Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            set_size(4.3,2)
            if ipath==npath-1: #Save figure after plotting last path
                plt.savefig('MC2.pgf')
###Antithetic Plots
            plt.figure(3)
            plt.grid(1)
            plt.title('Possible Short Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            plt.plot(x1,tmp1a)
            set_size(4.3,2)
            if ipath==npath-1:
                plt.savefig('MC3.pgf')

            plt.figure(4)
            plt.grid(1)
            plt.title('Possible Mean Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            plt.plot(x1,tmp2a)
            set_size(4.3,2)
            if ipath==npath-1:
                plt.savefig('MC4.pgf')
###Combined Plots
            plt.figure(5)
            plt.grid(1)
            plt.title('Antithetic and Non-Antithetic Rate Paths',**hfont)
            plt.xlabel('Number of Steps',**hfont)
            plt.plot(x1,tmp2,color='red', alpha=1, linewidth=0.5)
            plt.plot(x1,tmp2a,'--',color='blue', alpha=1,linewidth=0.5)
            #plt.legend(loc="upper right")
            set_size(4.3,2)
            if ipath==npath-1:
                plt.plot(x1,tmp2,color='red', alpha=1, 
                         label='Non-Antithetic', linewidth=0.5)
                plt.plot(x1,tmp2a,'--',color='blue', alpha=1,
                         label='Antithetic',linewidth=0.5)
                plt.legend(loc="upper left")
                plt.savefig('MC5.pgf')                
       
            
        
########################################################################
########(npath loop)                
        P1=np.exp(-rint1)
        P2=np.exp(-rint2)
        
        
        Pi=np.append(Pi, P1) 
        Pi=np.append(Pi, P2) 
        #Array of all bond prices from each path
        Pia=np.append(Pia, P1) 
        #Array of bond prices from path 1
        Pib=np.append(Pib,P2) 
        #Array of bond prices from path 2
        Pic=np.append(Pic, (P1+P2)/2) 
        #Array of pairwise averages as individual samples
        #ytmj=np.append(ytmj, (-np.log(P1)/T)) #Array of ytms from each bond price

        Psum1+=P1
        Psum2+=P2
        Psuma+=((P1+P2)/2)

########################################################################
####(counter loop)
    plt.show(2) #Shows figures after completion of all paths.
    Pave1=Psum1/npath
    Pave2=Psum2/npath
    Pavea=Psuma/npath 
    #Average bond price using pairwise averages as individual samples
    #(pg107 Jaeckel)

########################################################################

####(counter loop)
###Finding the Correlation and Covariance of Antithetic and Non-Antithetic:
    meansna=np.append(meansna, Pave1)
    #Saving the sample mean for each iteration of npaths
    meansa=np.append(meansa, Pave2) 
    #Saving the sample mean for each iteration of npaths

    for j in range(npath):
        tmpsum1[counter]+=((Pia[j]-meansna[counter])**2)
    svari1[counter]=(1/((npath)-1))*tmpsum1[counter] 
    #sample variance of BP

    for j in range(npath):
        tmpsum2[counter]+=((Pib[j]-meansa[counter])**2)
    svari2[counter]=(1/((npath)-1))*tmpsum2[counter] 
    #sample variance of BP

    for j in range(npath):
        tmpsumb[counter]+=((Pia[j]-meansna[counter])*(Pib[j]-meansa[counter])) 
        #Numerator of covariance
    covar[counter]=(1/((npath)-1))*tmpsumb[counter] #covariance
    
    corr[counter]=(covar[counter]
    /((np.sqrt(svari1[counter]))*(np.sqrt(svari2[counter])))) 
    #Finding the Correlation
    

###Finding the Sample Variance of BP:###
    means=np.append(means, Pavea) 
    #Saving the sample mean for each iteration of npaths
    for j in range(npath):
        tmpsuma[counter]+=((Pic[j]-means[counter])**2)
    svari[counter]=(1/((npath)-1))*tmpsuma[counter] 
    #sample variance of BP
    
    
    ytmntrl=-np.log(Pavea)/T

    
#########################################################################
    if counter==-1:
        ytm_exact=ytmntrl
        #ytm_exact=0.11130203518039194 #new using ODE result
    else:
        errs[counter,:]=nn,npath,np.abs(ytmntrl-ytm_exact)
        print('nn {:d} npath {:d} ytm {:.16f} err {:.4e}'
              .format(nstep,npath,ytmntrl,np.abs(ytmntrl-ytm_exact)))
        print('Paths:',npath,'Step Size:',T/nstep,'\nSample Mean:'
              ,means[counter],
              'Std Err:',(np.sqrt(svari[counter])/np.sqrt(npath)), 
              '\nUB Sample Variance:',svari[counter],
              'Standard Deviation:',np.sqrt(svari[counter]))
        print('95% Confidence Interval (BP):',
              means[counter]-1.96*(np.sqrt(svari[counter])/np.sqrt(npath))
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
               -2.58*(np.sqrt(svari[counter])/np.sqrt(npath)))/T)
        print('Covariance:', covar[counter],
              'Correlation:', corr[counter],'\n')

#########################################################################

###Error Plot###
plt.ylabel('Error')
plt.xlabel('Number of Paths')
plt.grid(1)
plt.loglog(errs[:,0],errs[:,2],'bo-')
plt.loglog(errs[:,0],5e-1*errs[:,0]**-1,'--')
plt.loglog(errs[:,0],1e-1*errs[:,0]**-0.5,'--')




















