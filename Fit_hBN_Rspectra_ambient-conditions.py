# -*- coding: utf-8 -*-
"""
@author: BDAlo 

Code for the fit of IR reflectance spectra of hBN at ambient conditions.
The reflectance R(nu) is computed from the Fresnel's coefficients for a 2-interface system (thin dielectric slab).

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
    
def delta(x, eps_inf, eps_0, omega_TO, gamma_0, d):
    return 2*np.pi*N(x, eps_inf, eps_0, omega_TO, gamma_0)*x*d
    
def r_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return ( r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1) + r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(2*complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )/( 1 + r_12(x, eps_inf, eps_0, omega_TO, gamma_0, n1)*r_23(x, eps_inf, eps_0, omega_TO, gamma_0, n3)*np.exp(2*complex(1j)*delta(x, eps_inf, eps_0, omega_TO, gamma_0, d)) )

def R_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d):
    return np.abs(r_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d))**2

#%% definition of the fit functions and initial parameters for the fit 

def R_3l_fit(params, x, data):
    eps_inf = params['eps_inf'].value
    eps_0 = params['eps_0'].value
    omega_TO = params['omega_TO'].value
    gamma_0 = params['gamma_0'].value
    n1 = params['n1'].value
    n3 = params['n3'].value
    d = params['d'].value
    model = R_123(x, eps_inf, eps_0, omega_TO, gamma_0, n1, n3, d)
    return (model - data)


##---------------------------------------Main----------------------------------------------
#Data's folders and names

data_path = r'C:\Users\dalo\OneDrive - SYNCHROTRON SOLEIL\Drive - SOLEIL\Articoli\hBN@HP_OPTIR\Data+codes_folder\Data Fig1' #'path'
data_name = '1p9µm-thick_hBN_R_spectrum' #'name'

result_path = r'C:\Users\dalo\OneDrive - SYNCHROTRON SOLEIL\Drive - SOLEIL\Articoli\hBN@HP_OPTIR\Data+codes_folder\Codes\test' #'result_path'

#Importing the data 
raw_data = np.loadtxt('%s\\%s.txt' %(data_path, data_name), skiprows=0, encoding='utf-8') 

#%% Fitting the R spectrum

xdata = np.array(raw_data[:, 0])
ydata = np.array(raw_data[:, 1])
y_max = max(ydata)
y_min = min(ydata)
y_diff = y_max-y_min

params = lmfit.Parameters()
params['eps_inf'] = lmfit.Parameter(name='eps_inf', value = 4.95)  # min=eps_inf_P(P[i])-0.05, max=eps_inf_P(P[i])+0.05
params['eps_0'] = lmfit.Parameter(name='eps_0',  value= 6.96)  # min=1.8, max=10   min = 6.4, max = 8.8 
params['omega_TO'] = lmfit.Parameter(name='omega_TO', value = 1365) #vary = False
params['gamma_0'] = lmfit.Parameter(name='gamma_0', value = 8.78) #4 cm-1,  min = 7.2, max = 7.6 vary = False
params['n1'] = lmfit.Parameter(name='n1', value= 1, vary = False) #, min=1.8, max=3.0
params['n3'] = lmfit.Parameter(name='n3', value= 2.38, vary = False)
params['d'] = lmfit.Parameter(name='d', value = 0.000200) #min=0.0001, max= 0.00064

result = lmfit.minimize(R_3l_fit, params, args=(xdata, ydata))
fit_params = np.array([result.params['eps_inf'].value, result.params['eps_0'].value, result.params['omega_TO'].value, result.params['gamma_0'].value, result.params['n1'].value, result.params['n3'].value, result.params['d'].value]) 

#%% Plotting the results 

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)

#plotting the data with the fit superimposed 
ax.plot(raw_data[:, 0], (raw_data[:, 1]), color = 'blue', linewidth = 2, label = "P$_{amb}$") 
ax.plot(raw_data[:, 0], (R_123(raw_data[:, 0], *fit_params)), color= 'red') 

ax.set_xlabel('ω (cm$^{-1}$)')
ax.set_ylabel('OPTIR intensity (normalized)')
ax.legend(loc  = 'upper left') #bbox_to_anchor=(1.0, 1.0)

plt.show()
plt.savefig('%s\Plot_fit_%s.png' %(result_path, data_name), dpi = 240)

#%% Writing the output into a file
df = pd.DataFrame(fit_params) #columns=["P(GPa)","eps_inf","eps_0","omega_TO","gamma_0","n1","n3","d","I0"]
np.savetxt('%s\Fit_results_%s.txt' %(result_path, data_name), df.values, fmt='%f')

#Writing a file with the data for the plot (data + best-fit curve)
plot_data = zip(xdata, ydata, R_123(xdata, *fit_params))
df = pd.DataFrame(plot_data, columns=["xdata","ydata","fit"])

np.savetxt('%s\Plot_fit_%s.txt' %(result_path, data_name), df.values, fmt='%f')
    

