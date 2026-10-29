"""
@author: Beatrice D'Alò - Synchrotron SOLEIL

This script opens a .ptir file containing high-pressure OPTIR spectra of the considered sample (here, hBN) and performs a fit of each spectrum through the lmfit package.
The OPTIR signal is described as I_OPTIR = I0*A(nu), where A(nu) = 1 - T(nu) - R(nu), T and R = transmittance and reflectance of the system.
T and R are computed from the Fresnel's coefficients for a 2-interface system (thin dielectric slab).

The program returns: 
- a plot of the data + best-fit curves (normalised and stacked for clarity) 
- a plot of the best-fit parameters
- a txt file with the best-fit parameters 
- txt files with the best-fit curves (one file for each high-pressure spectrum)
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import lmfit 

##---------------------------DEFINITIONS---------------------------------

##Functions to read the data (and the relative meta data) from the ptir file
#This function returns the labels of each map/spectrum inside the PTIR file
def data_label(h5):
    return list(h5['MEASUREMENTS'].keys())

#This function returns the name of each map/spectrum inside the PTIR file
#Note: decoding the name in the UTF-8 format is needed
def data_name(h5):
    data_label = list(h5['MEASUREMENTS'].keys())
    N_data = len(h5['MEASUREMENTS'])
    data_name = ['0']*N_data #Creating a list with zeros of dimension N, to host the name
    for i in range(0, N_data):
        data_name[i] = h5['MEASUREMENTS/%s' %data_label[i]].attrs['Label'].decode('utf8')
        #print('Name of data n.%i: %s' %(i, data_name[i]))
    return data_name

#This function finds the pressure value from the map/spectrum name
def pressure_value(data_names):
    match = re.search(r'P(\d+(?:[a-zA-Z]\d+)?)', data_names)
    if match:
        p_value = match.group(1)
        #print('p = %s' %p_value)
    if 'p' in p_value:
        p = float(p_value.replace('p', '.'))
    else: 
        p = float(p_value)
    #print('p = %.1f' %p)
    return p

#This function reads the pressure values from the txt file with measurement parameters of each map, and returns an array with the pressures
#Notice: the path refers to the txt file with the parameters, not to the h5 file
def pressure_values_dataset(h5, path):
    #data_label = list(h5['MEASUREMENTS'].keys())
    data_names = data_name(h5)
    N_data = len(h5['MEASUREMENTS'])
    P = np.zeros(N_data)
    for i in range(0, N_data):
        params = np.loadtxt('%s\%s_params.txt' %(path, data_names[i]), skiprows=1)
        P[i] = params[0]
    return P

#Reading the meta-data: this function reads and returns an array with the parameters at which the O-PTIR spectrum were acquired
#In order: P, IR power, probe power, gain, DC values, detector sum voltage values, x start, x increment
def params_dataset(h5, path):
    data_names = data_name(h5)
    N_data = len(h5['MEASUREMENTS'])
    params = np.zeros(N_data, 8)
    for i in range(0, N_data):
        data_params = np.loadtxt(r'%s\%s_params.txt' %(path, data_names[i]), skiprows=1)
        params[i, 0] = data_params[0] #P
        params[i, 1] = data_params[1] #IR power
        params[i, 2] = data_params[2] #probe power
        params[i, 3] = data_params[3] #gain
        params[i, 4] = data_params[4] #DCvalue
        params[i, 5] = data_params[5] #detector sum voltage values
        params[i, 6] = data_params[6] #x start
        params[i, 7] = data_params[7] #x increment
    return params

##Functions for the refractive index and the Fresnel coefficients
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

#Fresnel coefficient from Tanner, normal incidence (theta_i = 0) and two-interface case

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

#function for the fit: A_123 rescaled by a constant factor I0, determined from the data

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

#Expected behaviour for eps_inf and eps_0 with pressure (to initialize the fit parameters)
def eps_inf_P(P):
    eps_inf_P0 = 4.85 # 4.95 from Segura et al (exp.)
    rate = 0.047 #from Segura et al (exp.)
    return eps_inf_P0 + rate*P 

def eps_0_P(P):
    eps_0 = 7.0 # 6.96 from Segura et al (exp.)
    rate = 0.062 #from the fit of the data of Segura et al for eps_0
    return eps_0 + rate*P
##-----------------------------------------MAIN------------------------------------------
#Opening the PTIR file (containing the DC or OPTIR maps only) into a h5py File object (h5)
#data's folders
path_0 = 'data_path'
path_txt = 'txt_data_path'
#results' folder
path = 'results_path'
path_fit_curves = 'results_path'
#Name of the .ptir file
file_name = r'HP_380nm-thick_hBN_OPTIRspectra'
#name of the analysed sample
sample_name = 'sample_name' #'380nm-thick_hBN'
f = r'%s\%s.ptir' %(path_0, file_name)
h5 = h5py.File(f, 'r')
print('Reading the O-PTIR spectra file: ok')

#Counting the number of OPTIR spectra into the file
N_data = len(h5['MEASUREMENTS'])

#Selecting the name of the OPTIR or DC spectra as written in the PTIR file
spectrum_name = data_name(h5)

#Extracting the pressure values 
P = pressure_values_dataset(h5, path_txt)

#creating a numpy array for the infos on the spectra (P, max, min) and for the fit results
stats = np.zeros((N_data, 3))
stats[:, 0] = P
n_fit_params = 9
fit_res = np.full((N_data, n_fit_params), np.nan)
fit_res[:, 0] = P
#array to store the indeces of the P0 pressures
n_P0 = 3 
P0_ind = np.full((n_P0), np.nan)

#fitting procedure
params = lmfit.Parameters()

fig, ax = plt.subplots()
fig.set_size_inches(4.8, 8.2)

step = 0.8
cmap = plt.get_cmap('hsv') #winter
colors = [cmap(i) for i in np.linspace(0, 1, N_data)]
p = 0

#sorting the P values to perform the fitting procedure from lower to higher pressure values (important to initialize each fit to its previous value)
sorted_ind = np.argsort(P) #array with the indeces values ordered following the pressure values
for i in sorted_ind:
    print('%i, Index %i, P=%.1f' %(i, sorted_ind[i], P[i]))

for i in sorted_ind:
    if ((P[i] == 0) and ('closed' in spectrum_name[i])):   #first fit at 0 GPa --> selecting the right spectrum by its name in the .ptir file
        print('%i, Index %i, P=%.1f' %(i, sorted_ind[i], P[i]))
        P0_ind[2] = i
        print("Index for closed cell: %i" %i)
        data = np.loadtxt('%s\%s.txt' %(path_txt, spectrum_name[i]))
        x = data[:, 0]
        y = data[:, 1]
        #print(x)
        #print(y)
        stats[i, 1] = np.max(y)
        stats[i, 2] = np.min(y)
        ax.plot(x, (y-stats[i, 2])/(stats[i, 1]-stats[i, 2]) + step*p, color= colors[p], label = 'P = %.1f' %P[i]) #plotting the data (normalized)
        params['eps_inf'] = lmfit.Parameter(name='eps_inf', value = eps_inf_P(P[i]))  
        params['eps_0'] = lmfit.Parameter(name='eps_0',  value= 6.96, vary = False)  
        params['omega_TO'] = lmfit.Parameter(name='omega_TO', value = 1352) 
        params['gamma_0'] = lmfit.Parameter(name='gamma_0', value = 9.49, vary = False) 
        params['n1'] = lmfit.Parameter(name='n1', value= 2.38, vary=False) #refractive index of diamond in the chosen IR range
        params['n3'] = lmfit.Parameter(name='n3', value= 1.51, vary = False) #refractive index of NaCl in the chosen IR range
        params['d'] = lmfit.Parameter(name='d', value = 0.0000380, vary = False) 
        params['I0'] = lmfit.Parameter(name='I0', value = 72, min = 0)
        #fitting
        result = lmfit.minimize(A_3l_fit, params, args=(x, y))
        fit_params = np.array([result.params['eps_inf'].value, result.params['eps_0'].value, result.params['omega_TO'].value, result.params['gamma_0'].value, result.params['n1'].value, result.params['n3'].value, result.params['d'].value, result.params['I0'].value])
        fit_res[i,1:] = fit_params
        #plotting the results
        ax.plot(x, step*p + A_123_I0(x, *fit_params)/stats[i, 1], color= 'green') #plotting the data (normalized)
        np.savetxt('%s\%s_best-fit_curve.txt' %(path_fit_curves, spectrum_name[i]), np.c_[x, A_123_I0(x, *fit_params)], fmt='%f', header = 'x(cm-1)  A(nu) (adim.)')  
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('OPTIR intensity (normalized and shifted)')
        ax.legend(loc  = 'upper left') #bbox_to_anchor=(1.0, 1.0)
        ax.legend()
        p += 1
        print("Fit ok")
    elif ((P[i] < 12) and ('back' not in spectrum_name[i]) and (P[i] != 0)): # proceeding with the fits for P > 0 GPa. At each pressure, the fit parameters are initialized to the best-fit ones of the previous pressure 
        print('%i, Index %i, P=%.1f' %(i, sorted_ind[i], P[i]))
        data = np.loadtxt('%s\%s.txt' %(path_txt, spectrum_name[i]))
        x = data[:, 0]
        y = data[:, 1]
        #print(x)
        #print(y)
        stats[i, 1] = np.max(y)
        stats[i, 2] = np.min(y)
        ax.plot(x, (y-stats[i, 2])/(stats[i, 1]-stats[i, 2]) + step*p, color= colors[p], label = 'P = %.1f' %P[i]) #plotting the data (normalized)  
        #setting the fitting parameters
        params['eps_inf'] = lmfit.Parameter(name='eps_inf', value = eps_inf_P(P[i]), min = eps_inf_P(P[i])-0.05, max = 6.36)  
        params['eps_0'] = lmfit.Parameter(name='eps_0',  value =  eps_0_P(P[i]), vary = False)  
        params['omega_TO'] = lmfit.Parameter(name='omega_TO', value = fit_params[2]) 
        params['gamma_0'] = lmfit.Parameter(name='gamma_0', value = 9.49, vary = False) 
        params['n1'] = lmfit.Parameter(name='n1', value= 2.38, vary=False) 
        params['n3'] = lmfit.Parameter(name='n3', value= 1.51, vary = False)
        params['d'] = lmfit.Parameter(name='d', value = fit_params[6], vary = False) 
        params['I0'] = lmfit.Parameter(name='I0',  value = fit_params[7])
        #fitting
        result = lmfit.minimize(A_3l_fit, params, args=(x, y))
        fit_params = np.array([result.params['eps_inf'].value, result.params['eps_0'].value, result.params['omega_TO'].value, result.params['gamma_0'].value, result.params['n1'].value, result.params['n3'].value, result.params['d'].value, result.params['I0'].value])
        fit_res[i,1:] = fit_params
        #plotting the results
        ax.plot(x, step*p + A_123_I0(x, *fit_params)/stats[i, 1], color= 'blue') #plotting the data (normalized)  
        np.savetxt('%s\%s_best-fit_curve.txt' %(path_fit_curves, spectrum_name[i]), np.c_[x, A_123_I0(x, *fit_params)], fmt='%f', header = 'x(cm-1)  A(nu) (adim.)')
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('OPTIR intensity (normalized and shifted)')
        ax.legend(loc  = 'upper left') #bbox_to_anchor=(1.0, 1.0)
        ax.legend()
        p += 1
        print("Fit ok")

params_name = ['$\epsilon_\infty$', '$\epsilon_0$', '$\omega_{TO}$', '$\gamma_0$', '$n_1$', '$n_3$', 'd', '$I_0$']
np.savetxt('%s\Fit_results_HP_%s.txt' %(path, sample_name), fit_res, fmt='%f', header = '%s' %params_name)
fig.savefig('%s\Plot_fit_HP_%s.png' %(path, sample_name), dpi = 240)

#plotting the fit results

fig_params, ax_params = plt.subplots(2, 4)
fig_params.set_size_inches(14, 8)

ax_params[0][0].scatter(fit_res[:,0], fit_res[:,1], color= 'blue', label = '$\epsilon_\infty$')
ax_params[0][1].scatter(fit_res[:,0], fit_res[:,2], color= 'blue', label = '$\epsilon_0$')
ax_params[0][2].scatter(fit_res[:,0], fit_res[:,3], color= 'blue', label = '$\omega_{TO}$')
ax_params[0][3].scatter(fit_res[:,0], fit_res[:,4], color= 'blue', label = '$\gamma_0$')
ax_params[1][0].scatter(fit_res[:,0], fit_res[:,5], color= 'blue', label = '$n_1$')
ax_params[1][1].scatter(fit_res[:,0], fit_res[:,6], color= 'blue', label = '$n_3$')
ax_params[1][2].scatter(fit_res[:,0], fit_res[:,7], color= 'blue', label = 'd')
ax_params[1][3].scatter(fit_res[:,0], fit_res[:,8], color= 'blue', label = '$I_0$')

for i in range(0, 8):
    ax_params[i//4][i%4].set_xlabel('P (GPa)')
    ax_params[i//4][i%4].set_ylabel('%s' %params_name[i]) #fontsize = '8'

fig_params.savefig('%s\Plot_fit_HP_%s_params.png' %(path, sample_name), dpi = 240)

plt.show()
