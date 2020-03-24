from matplotlib import pyplot as plt
import math as m
import numpy as np
from scipy.integrate import solve_ivp

#parameters
N=3
method='RK45'
# RK23 or RK45: non-stiff#
# method='Radau'# Radau or BDF or LSODA: stiff

#Function to Set Size of Plot Axes
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
########################################################################

#global data structures#

# main
K=np.zeros([N,N])
Kt=np.zeros([N,N])
beta=np.zeros([N,N])
Sig=np.zeros([N,N])

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
# A, B[0:N-1]#
# ntrlfunc
KtTB=np.zeros([N])
SigTB=np.zeros([N])
SigTBsq=np.zeros([N])
T=8  # bond maturity


Y[0]=0.11   #Short Rate
Y[1]=0.039   #Mean Rate 
Y[2]=0.015  #Volatility
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
    K[ i,i]=kap[i]
    Sig[ i,i]=sig[i]
    Kt[i,i]=kap[i]+lam[i]
    Theta[ i]=mu[i]
    Thetat[i]=Theta[i]*K[i,i]/Kt[i,i]
    
########################################################################

###Functions 17 and 18 from Huang and Yu that are to be integrated.###
def ntrlfunc(t,y):  # A=y[0], B[0:N-1]=y[1:N]
    for i in range(N):
        KtTB[i]=0.0
        SigTB[i]=0.0
        for j in range(N):
            KtTB[i]+=Kt[j,i]*y[i+1] # KtT B
            SigTB[i]+=Sig[j,i]*y[i+1]   # SigT B

    for i in range(N):
        SigTBsq[i]=SigTB[i]*SigTB[i]    # ( [SigT B]_j )^2

    fntrl[0]=-delta0;
    for j in range(N):
        fntrl[0]+=-Thetat[j]*KtTB[j]+0.5*SigTBsq[j]*alpha[j]    #  y[0] 

    for i in range(N):
        fntrl[i+1]=-KtTB[i]+deltay[i]
        for j in range(N):
            fntrl[i+1]+=-0.5*SigTBsq[j]*beta[i,j]   #  y[1:N] 

    return fntrl
########################################################################
 
ntol=30
tolmax=1e-3
tolmin=1e-12
ltolmax=np.log10(tolmax)
ltolmin=np.log10(tolmin)
dltol=(ltolmax-ltolmin)/ntol
errs=np.zeros([ntol+1,3])
for counter in range(-1,ntol+1):
    if counter==-1: # reference solution
        ltol=ltolmin-1
    else:
        itol=ntol-counter
        ltol=ltolmin+itol*dltol
    tol=10**ltol
    #print(counter,'tol',tol)

    rtol=tol
    atol=tol
    # Initialise A, B[0:N-1]
    for i in range(N+1):
        yntrl[i]=0.0
        #   ABCD
        #  [0,    1,.....,N,     N+1,   N+2,........,2N+1]
    ytmntrl=delta0
    for j in range(N):
        ytmntrl+=deltay[j]*Y[j] #Before EQ15, r(t)=d0+dT_y.y(t)
    # start risk ntrl integration

    t=0.0
    ti=T
    #integrates using equations 17 and 18 from Huang & Yu
    sol=solve_ivp(fun=ntrlfunc, t_span=(t,ti), rtol=rtol, atol=atol
                  ,y0=yntrl, method=method)

    #pick out terminal values
    t=sol.t[-1]
    yntrl=sol.y.T[-1]
    yntrlall=sol.y #All Solution Values
    At=-yntrlall[0]
    Bt1=yntrlall[1]
    Bt2=yntrlall[2]
    Bt3=yntrlall[3]
    ######
    tmp1=sol.y.T
    #####
    #print(range(N))
    ytmntrl=-yntrl[0] 
    for j in range(N):
        ytmntrl+=yntrl[j+1]*Y[j]   # ytm eqn 16
        #print(ytmntrl)
    ytmntrl/=t
    P=np.exp(-ytmntrl*8)

    #print(ytmntrl)
    ####Setup for Yield Curve Plot###
    t1=[]
    t1=sol.t[:] #All time values (5417 vars)

    if counter==-1: #reference solution
        ytm_exact=ytmntrl
    else:
        errs[counter,:]=tol,sol.nfev,np.abs(ytmntrl-ytm_exact)
        print('tol{:.4e}  t{:.4e}  ytm{:.16e}  err{:.4e}'
              .format(tol,t, ytmntrl,np.abs(ytmntrl-ytm_exact)))

#### Setting up The Yield Curve: ###
yields1=np.zeros([0]) #Creating NumPy array of zeros
yields1=(Y[0]+Y[1]+Y[2]) #NumPy array of bond yields

for l in range(len(At)):
    if l>0:
        yields1=np.append(yields1,((At[l]/(t1[l])+(Bt1[l]/t1[l])*Y[0]
                        +(Bt2[l]/t1[l])*Y[1]+(Bt3[l]/t1[l])*Y[2]))) 
        #Appending yields from formula.
#########################################################################
        
###Plotting the Yield Curve###
plt.plot(sol.t,yields1,label="$y(\\tau)$")
plt.legend(loc="upper right")
plt.grid()
csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Helvetica'}
plt.xlabel('Time to Maturity: $ \\tau$ (in years)',**hfont)
#plt.ylabel('Yield to Maturity: $ y(\\tau)$',**hfont)
plt.title('December 4, 1980',**hfont)
set_size(4.3,2)
plt.savefig('yc3.pgf')
plt.show()
#########################################################################

###Plot of Error###
plt.ylabel('error')
plt.xlabel('tol')
plt.loglog(errs[:,0],errs[:,2],'bo-')
plt.show()
#########################################################################

###Plot of the 4 Factors###
plt.plot(sol.t,sol.y[0],label="$A(\\tau)$")
plt.plot(sol.t,sol.y[1],label="$B_1(\\tau)$")
plt.plot(sol.t,sol.y[2],label="$B_2(\\tau)$")
plt.plot(sol.t,sol.y[3],label="$B_3(\\tau)$")
plt.legend(loc="upper left")


csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Helvetica'}
plt.grid()
#plt.xlabel('',**hfont)
#plt.ylabel('Yield to Maturity: $ y(\\tau)$',**hfont)
plt.title('Factors Influencing the Yield Curve Shape',**hfont)
plt.xlabel('Time to Maturity: $ \\tau$ (in years)',**hfont)
#plt.ylabel('',**hfont)
set_size(4.3,2)
plt.savefig('4factors.pgf')
plt.show()

#########################################################################

###Plot of Actual Bond Yields from Chen 2004###
tactual=np.array([0,0.5,0.75,1,1.25,1.5,2,2.5,3,3.25,3.5,3.75,4,4.5,4.75,5
                  ,5.5,6,6.75,7,7.5])
actualyields=np.array([[.14,.145,.158],
                       [.15,.153,.152],
                       [.15,.145,.147],
                       [.139,.14,.142],
                       [.138,.14,.139],
                       [.136,.135,.137],
                       [.13,.131,.129],
                       [.128,.129,.129],
                       [.13,.125,.126],
                       [.128,.126,.128],
                       [.127, .127, .127],
                       [.1265,.1265,.127],
                       [.126,.126,.126],
                       [.125,.125,.125],
                       [.124,.124,.124],
                       [.125,.125,.125],
                       [.125,.124,.125],
                       [.124,.124,.124],
                       [.122,.122,.12],
                       [.12,.119,.12],
                       [.116,.116,.116]],dtype=object)

    
plt.scatter(np.repeat(tactual, actualyields.shape[1]), actualyields.flat
            ,label="Actual Yields")
plt.grid()
plt.title('December 4, 1980',**hfont)
plt.xlabel('Time to Maturity: $ \\tau$ (in years)',**hfont)
#plt.ylabel('',**hfont)
set_size(4.3,2)
plt.legend(loc="upper right")
plt.savefig('figure1.pgf')
plt.show()
#######################################################################
    
###Plot of Actual Yields and Yield Curve:###    
plt.scatter(np.repeat(tactual, actualyields.shape[1]), actualyields.flat
            ,color='orange',label="Actual Yields")
plt.plot(sol.t,yields1,label="$y(\\tau)$")
plt.grid()
plt.title('December 4, 1980',**hfont)
plt.xlabel('Time to Maturity: $ \\tau$ (in years)',**hfont)

set_size(4.3,2)
plt.legend(loc="upper right")
plt.savefig('yc3.pgf')
plt.show()
########################################################################

