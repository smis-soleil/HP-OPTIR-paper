# -*- coding: utf-8 -*-
"""
@author: BD

Code for the fit of OPTIR spectra of hBN at ambient conditions.
The OPTIR signal is described as I_OPTIR = I0*A(nu), where A(nu) = 1 - T(nu) - R(nu), T and R = transmittance and reflectance of the system.
T and R are computed from the Fresnel's coefficients for a 2-interface system (thin dielectric slab).

The program returns: 
- a plot of the data + best-fit curve
- a txt file with the best-fit parameters 
- a txt file with the x, y (OPTIR), y (best-fit curve) data
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import lmfit 

##-----------------------------Function definition------------------------------------------------------------------------
#%% Defining the functions for the refractive index (isotropic case) and the Fresnel coefficients

#Definition of N from epsilon for hBN (function from Segura et el., JPCC, 2019)

def N(x, eps_inf, eps_0, omega_TO, gamma_0):
    return np.sqrt(eps_inf + (eps_0 - eps_inf)*(omega_TO**2)/(omega_TO**2 - x**2 - complex(1j)*gamma_0*x))

def n(x, eps_inf, eps_0, omega_TO, gamma_0):
    return N(x, eps_inf, eps_0, omega_TO, gamma_0).real

def k(x, eps_inf, eps_0, omega_TO, gamma_0):
    return N(x, eps_inf, eps_0, omega_TO, gamma_0).imag

#connection between eps_0 and omega_LO 

def omega_LO(eps_inf, eps_0, omega_TO):
    return omega_TO*np.sqrt(eps_0/eps_inf)

#Fresnel's coefficient from Tanner, normal incidence (theta_i = 0) and two-interface case

def r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1):
    return (n1 - N(x, eps_inf, eps_0, omega_TO, gamma_0))/(n1 + N(x, eps_inf, eps_0, omega_TO, gamma_0))

def r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3):
    return ( N(x, eps_inf, eps_0, omega_TO, gamma_0) - n3 )/( N(x, eps_inf, eps_0, omega_TO, gamma_0) + n3 )

def t_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1):
    return ( 2*n1 )/( n1 + N(x, eps_inf, eps_0, omega_TO, gamma_0)  )

def t_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3):
        return ( 2*N(x, eps_inf, eps_0, omega_TO, gamma_0) )/( N(x, eps_inf, eps_0, omega_TO, gamma_0) + n3  )
    
def delta(x, eps_inf, eps_0, omega_TO, gamma_0, d):
    return 2*np.pi*N(x, eps_inf, eps_0, omega_TO, gamma_0)*x*d
    
def r_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return ( r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1) + r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(2*complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )/( 1 + r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1)*r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(2*complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )

def t_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return ( t_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1)*t_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )/( 1 + r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1)*r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(2*complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )

def R_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return np.abs(r_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d))**2

def T_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return n3/n1*np.abs(t_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d))**2

def A_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return ( 1 - R_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d) - T_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d) )

#Expected behaviour for eps_inf and eps_0 with pressure (initial parameter in the fitting procedure)

def eps_inf_P(P):
    eps_inf_P0 = 4.95 #from Segura et al (exp.)
    rate = 0.047 #from Segura et al (exp.)
    return eps_inf_P0 + rate*P 

def eps_0_P(P):
    eps_0 = 6.96
    rate = 0.062 #from the fit of the data of Segura et al for eps_0
    return eps_0 + rate*P

#%% definition of the fit functions, initial parameters for the fit 

#function for the fit: A_123 rescaled by a constant factor I0, to be determined from the data
def A_123_I0(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d, I0):
    return I0*A_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d)

def A_3l_fit(params, x, data):
    eps_inf = params['eps_inf'].value
    eps_0 = params['eps_0'].value
    omega_TO = params['omega_TO'].value
    gamma_0 = params['gamma_0'].value
    n1 = params['n1'].value
    n3 = params['n3'].value
    d = params['d'].value
    I0 = params['I0'].value
    model = A_123_I0(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d, I0)
    return (model - data)

##---------------------------------------Main----------------------------------------------
#Data's folders and names

data_path = 'path' #'path'
data_name = 'name' #'name'

result_path = 'result_path' #'result_path'

#Importing the data 
raw_data = np.loadtxt('%s\\%s.txt' %(data_path, data_name), skiprows=1, encoding='utf-8') 

#%% Fitting the OPTIR spectrum

xdata = np.array(raw_data[:, 0])
ydata = np.array(raw_data[:, 1])
y_max = max(ydata)
y_min = min(ydata)
y_diff = y_max-y_min

#Parameters intialised to realistic values
params = lmfit.Parameters()
params['eps_inf'] = lmfit.Parameter(name='eps_inf', value = 4.96)  
params['eps_0'] = lmfit.Parameter(name='eps_0',  value= 6.96)  
params['omega_TO'] = lmfit.Parameter(name='omega_TO', value = 1360) 
params['gamma_0'] = lmfit.Parameter(name='gamma_0', value = 7.3) 
params['n1'] = lmfit.Parameter(name='n1', value= 1, vary = False) #n=1 for air, n = 2.38 for diamond
params['n3'] = lmfit.Parameter(name='n3', value= 2.38, vary = False) #n = 2.38 for diamond, n = 1.51 for NaCl
params['d'] = lmfit.Parameter(name='d', value = 0.000194) 
params['I0'] = lmfit.Parameter(name='I0', value = 50)

result = lmfit.minimize(A_3l_fit, params, args=(xdata, ydata))
fit_params = np.array([result.params['eps_inf'].value, result.params['eps_0'].value, result.params['omega_TO'].value, result.params['gamma_0'].value, result.params['n1'].value, result.params['n3'].value, result.params['d'].value, result.params['I0'].value])

#%% Plotting the results 

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

#plotting the data and the corresponding best-fit curve
ax.plot(raw_data[:, 0], (raw_data[:, 1]), color = 'blue', linewidth = 2, label = "P$_{amb}$") 
ax.plot(raw_data[:, 0], (A_123_I0(raw_data[:, 0], *fit_params)), color= 'red') 

xmin = 1000
xmax = 1800
ax.set_xlim(xmin, xmax)
ax.set_xlabel('ω (cm$^{-1}$)')
ax.set_ylabel('OPTIR intensity (normalized)')
ax.legend(loc  = 'upper left') 

plt.show()
plt.savefig('%s\\Plot_fit_%s.png' %(result_path, data_name), dpi = 240)

#%% Writing the output into a file
df = pd.DataFrame(fit_params) #columns=["P(GPa)","eps_inf","eps_0","omega_TO","gamma_0","n1","n3","d","I0"]
np.savetxt('%s\\Fit_results_%s.txt' %(result_path, data_name), df.values, fmt='%f')

#%% Saving the file for the collective plot (P + OPTIR spectra + fit)
plot_data = zip(xdata, ydata, A_123_I0(xdata, *fit_params))
df = pd.DataFrame(plot_data, columns=["xdata","ydata","fit"])
np.savetxt('%s\\Plot_fit__%s.txt'  %(result_path, data_name), df.values, fmt='%f')
    