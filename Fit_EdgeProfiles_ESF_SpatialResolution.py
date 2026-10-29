"""
@author: BDAlo 

This python script reads the linescan and fits it thorugh a sigmoidal function to obtain the edge spread function (ESF).
It computes its derivative (the line spread function - LSF) and the FWHM of the LSF.

The program gives in output: 
- a plot of the linescan + best-fit sigmoidal curve
- a plot of the linescan + LSF + a light-shaded area which indicates the FWHM of the LSF
- a txt file with the best-fit parameters
The best-fit parameters and the FWHM of the LSF are written on the terminal.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.ticker as mticker
import lmfit

##------------------------------
#Defining functions

#from m to µm  (useful for linescans from the AFM topography)s
def m_to_µm(m):
    return m*10**(6)

#sigmoidal function 
def sigmoidal_function(x, a, b, r, x_0):
    return a + b/(1 + np.exp(r*(-x+x_0)))

#Fitting function for the sigmoidal function
def sigmoidal_fit(params, x, data):
    a = params['a'].value
    b = params['b'].value
    r = params['r'].value
    x_0 = params['x_0'].value
    model = sigmoidal_function(x, a, b, r, x_0)
    return (model - data)

def sigmoidal_der_function(x, b, r, x_0):
    return (b*r*np.exp(-r*(x-x_0)))/(1+np.exp(-r*(x-x_0)))**2

def xFWHM_1(r, x0):
    return (x0 - np.log(3+2*np.sqrt(2))/r)

def xFWHM_2(r, x0):
    return (x0 - np.log(3-2*np.sqrt(2))/r)

def FWHM(r):
    return 1/r*np.log((3+2*np.sqrt(2))/(3-2*np.sqrt(2)))

##------------------------------MAIN-----------------------------------------------
#Reading the files

#Folders' paths
path_txt = 'path_txt'
path_fig = 'path_fig'
path_fig_save = 'path_fig_folder'
path_results = 'path_results_folder'
#name of the files with the linescans
name_ls1 = 'ls1'
name_ls2 = 'ls2'

#loadine the linescans (i.e., two of them)
ls_1 = np.loadtxt('%s\\%s' %(path_txt, name_ls1), skiprows=3, encoding='utf-8') #Linescan from image 1
ls_2 = np.loadtxt('%s\\%s' %(path_txt, name_ls2), skiprows=3, encoding='utf-8') #Linescan from image 2 

##-----------------------------------------------------------
#General fig params

plt.rcParams.update({
    "font.size": 10,          # base font size
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.titlesize": 12
})

color_1 = 'dimgrey' 
color_2 = 'red'

color_1 = 'dimgrey' #'silver'
color_2 = 'mediumseagreen'#'red'

linestyle_1 = '-'
linestyle_2 = '-'

panel_w = 2.8 
panel_h = 1.102
# Margins in inches
left_margin = 0.5
right_margin = 0.5
top_margin = 0.5
bottom_margin = 0.5

fig_w = panel_w + left_margin + right_margin
fig_h = panel_h + top_margin + bottom_margin

h = [Size.Fixed(left_margin), Size.Fixed(panel_w), Size.Fixed(right_margin)]
v = [Size.Fixed(bottom_margin), Size.Fixed(panel_h), Size.Fixed(top_margin)]

#--------------------------------
#Fitting the data

#fitting procedure
params = lmfit.Parameters()

#single fit
data_list = [ls_1, ls_2]
name_list = ['ls1', 'ls2']
yname_list = ['Height (µm)', 'I$_{OPTIR}$ (mV)'] #example of y labels
color_list = ['%s' %color_1, '%s' %color_2]
b0_list = np.array([1.4, 1.0]) #list of parameters for fit initialization
x0_list = np.array([5.0, 10]) #list of parameters for fit initialization

for i in range(0, 2):
    #creating the figure
    fig_fit= plt.figure(figsize=(fig_w, fig_h))
    divider = Divider(fig_fit, (0, 0, 1, 1), h, v, aspect=False)
    ax_fit = fig_fit.add_axes(divider.get_position())
    ax_fit.set_axes_locator(divider.new_locator(nx=1, ny=1))
    #reading the data set
    data = data_list[i]
    x = data[:, 0]
    y = data[:, 1]
    y_max = np.max(y)
    y_min = np.min(y)
    #ax_fit.plot(x, (y-y_min)/(y_max-y_min), color= '%s' %color_HP) #plotting the data (normalized)
    ax_fit.plot(x, y, color= '%s' %color_2) #plotting the data 
    #fit of the data
    params['a'] = lmfit.Parameter(name='a', value = 0)  # 
    params['b'] = lmfit.Parameter(name='b',  value= b0_list[i])  # 
    params['r'] = lmfit.Parameter(name='r',  value= 1, min = 0)
    params['x_0'] = lmfit.Parameter(name='x_0', value = x0_list[i]) #vary = False
    result = lmfit.minimize(sigmoidal_fit, params, args=(x, y))
    fit_params = np.array([result.params['a'].value, result.params['b'].value, result.params['r'].value, result.params['x_0'].value])
    a, b, r, x0 = fit_params[0], fit_params[1], fit_params[2], fit_params[3]
    np.savetxt('%s\%s_best-fit_params.txt' %(path_results, name_list[i]), np.c_[*fit_params, FWHM(fit_params[2])], fmt='%f', header = 'a  b   r   x_0 ')
    print('For %s: a = %.2f, b = %.2f, r = %.2f, x_0 = %.2f' %(name_list[i], *fit_params))
    print('FWHM = %.2f' %(FWHM(fit_params[2])))
    ax_fit.plot(x, sigmoidal_function(x, *fit_params), color= 'blue') #plotting the best-fit curve
    ax_fit.set_xlabel('Distance (µm)')
    ax_fit.set_ylabel('%s profile' %name_list[i])
    plt.savefig('%s\%s_ls_fit.png' %(path_fig, name_list[i]), dpi = 360)
    #Plotting the data + LSF + shaded area between the FWHM 
    fig_plot= plt.figure(figsize=(fig_w, fig_h))
    ax_plot = fig_plot.add_axes(divider.get_position())
    ax_plot.set_axes_locator(divider.new_locator(nx=1, ny=1))
    #plotting
    ax_plot.plot(x - x0, y - a, color= '%s' %color_list[i]) #plot of the data
    sigmoidal_der_max = b*r/4
    ax_plot.plot(x-x0, sigmoidal_der_function(x-x0, b, r, 0)/sigmoidal_der_max*b, color= 'grey', alpha = 0.6, linestyle='dashed', linewidth=0.8) 
    ax_plot.axvspan(xFWHM_1(r, x0) - x0, xFWHM_2(r, x0) -x0, color='%s' %color_list[i], edgecolor = 'None', linewidth=0, alpha=0.2)
    x_max = 18.2 #-2, 26
    x_min = -x_max
    ax_plot.set_xlim(x_min, x_max)
    ax_plot.set_xlabel('Distance (µm)')
    ax_plot.set_ylabel('%s' %yname_list[i])
    ax_plot.tick_params(axis='both', which='both', direction='in')
    ax_plot.xaxis.set_major_locator(mticker.MultipleLocator(5.0))
    plt.savefig('%s\%s_ls_plot.png' %(path_fig_save, name_list[i]), dpi = 360)

plt.show()


