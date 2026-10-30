# -*- coding: utf-8 -*-
"""
This script fits the data through a linear baseline + peak model (possiblity to increase the number of peaks). 
The program reads the data file and selects a x range of the data where it performs the fit.

It gives in output:
    - a figure with the data in the selected range, the initial fit curves and the best-fit ones,
    - a figure with the data + the best-fit curve,
    - a window asking if you want to save the fit results. 
    If yes --> the program produces a .txt file with the best-fit results and a .png image with the data + best-fit curve, and you can procede by fitting the higher pressure spectra.
    If no --> the program stops.
    The best-fit values from the last succefull fit are stored in a file. You can re-start the fitting from the last non-successfull fit.
    - a .txt file ('fit_results_%icm-1.txt' %peak) where the best-fit parameters at each pressure are stored.
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel, LorentzianModel
import tkinter as tk
from tkinter import messagebox
import os
import re
import pickle

np.set_printoptions(suppress=True)

#-----------------------Definitions-----------------------------------

#selecting the x range to fit in the files
def select_range(fname):
    x, y = np.loadtxt(fname, delimiter="\t",unpack=True)
    y = -np.log10(y) #conversion into absorbance
    x_min = 1600 #1420 #low-frequency limit in cm-1 of the range to keep
    x_max = 1632 #1450 #high-frequency limit in cm-1 of the range to keep
    ind=np.where((x > x_min) & (x < x_max)) #array containing the data_x array indexes corresponding to the low and high limits
    i_min=ind[0][0]
    i_max=ind[0][-1]
    x = x[i_min:i_max]
    y = y[i_min:i_max]
    return x, y

#This function finds the pressure value from the spectrum name
def P_from_filename(data_names):
    match = re.search(r'(\d+(?:[a-zA-Z]\d+)?)', data_names)
    if match:
        p_value = match.group(1)
        print('p = %s' %p_value)
    if 'p' in p_value:
        p = float(p_value.replace('p', '.'))
        print('p numeric = %s' %p)
    else: 
        p = float(p_value)
    #print('p = %.1f' %p)
    return p

#This function orders the files with respect to the pressure values 
def sorted_files(directory):
    #counts the files
    files = [f for f in os.listdir(directory) if (('.DPT' in f) and ('OPD' not in f))]
    n = len(files)
    print('The file number is %i' %n)
    print(files)
    #takes the pressure values
    p = np.zeros(n)
    i=0
    for f in files:
        p[i] = P_from_filename(f)
        i+=1
    files_sorted = sorted(files, key=P_from_filename)
    print (files_sorted)
    return files_sorted

#returns the index of the array at or below 'value'
def index_of(arrval, value):
    if value < min(arrval):
        return 0
    return max(np.where(arrval <= value)[0])

#fit function
def fit(directory, file, dest_dir):
    x, y = select_range(directory + file)
    P = float(P_from_filename(directory + file))
    #linear baseline
    bkg = LinearModel(prefix = 'baseline_')                     # two Parameters: intercept and slope     
    bkg.set_param_hint('intercept', value = y.min(), vary=False) #set_param_hint allow to use bounds and fix/vary the parameter
    pars = bkg.make_params(slope=0) #parameters values can be initialized here, in set_param_hint or in the Model() directly
    #1 peak fit (possiblity to add other ones, if needed)
    peak1 = LorentzianModel(prefix='L1_')   #three Parameters: amplitude, center, and sigma. Parameters fwhm (=2sigma) and height are also included.
    #peak2 = LorentzianModel(prefix='L2_')   
    pars += peak1.make_params()             
    pars.update(peak1.guess(y, x=x))
    pars['L1_center'].set(value = 1614, min=1580, max=1632) #parameters to initialize the fit
    pars['L1_sigma'].set(10, min=3, vary = True) #parameters to initialize the fit
    """
    pars += peak2.make_params()
    pars.update(peak2.guess(y, x=x))
    pars['L2_center'].set(value = 1460, min=1440, max=1480)    # x[np.argmax(y)] can be used to catch the wavenumber of the maximum
    pars['L2_sigma'].set(10, min=3)
    pars['L2_amplitude'].set(value = .5, min=0)
    """
    mod = peak1 + bkg # + peak2
    os.chdir(dest_dir) 
    #"""  #Taking the previous best-fit parameters to initialize the fit (not possible at the first fit)
    pickle_file = 'best_params.pk'
    with open(pickle_file, 'rb') as pf:
        best_old_pars = pickle.load(pf)
    #"""
    print("\nbest_pp = ", best_old_pars) #best_old_pars or #pars
    print("\npars = ", pars)    
    bkg_y_init = bkg.eval(best_old_pars, x = x) #best_old_pars or #pars
    out = mod.fit(y, best_old_pars, x=x, fit_kws={'xtol':10e-3}) #best_old_pars or #pars
    comps = out.eval_components()
    best_pars = out.params
    best_pars_values = out.best_values
    #print(best_pars)
    print('\nchisqrd = ', out.chisqr)
    centers = [best_pars_values.get('L1_center')] #, best_pars_values.get('L2_center')
    fwhms  = [best_pars['L1_fwhm'].value] #, best_pars['L2_fwhm'].value
    amplitudes  = [best_pars_values.get('L1_amplitude')] #, best_pars_values.get('L2_amplitude'), 
    rounded_centers = [round(item,2) for item in centers]
    rounded_fwhm = [round(item,2) for item in fwhms]             #For a Lorentzian fwhm = 2 * sigma
    rounded_amplitudes = [round(item,2) for item in amplitudes]  
    print("centers: ", rounded_centers,"\nfwhms:", rounded_fwhm, "\namplitudes:", rounded_amplitudes)

    cwa = np.concatenate((rounded_centers, rounded_fwhm, rounded_amplitudes))

    Pcwa = np.insert(np.array(cwa), 0, P)
    print(Pcwa)

    #plotting the results
    fig_comps = plt.figure()
    plt.title("Fit at %1.1f GPa" %P)
    plt.plot(x, y, 'bo', label='Data')
    plt.plot(x, out.init_fit, 'm-', label='Initial guess')
    plt.plot(x, out.best_fit, 'r-', label='Fit')
    plt.plot( x, comps['baseline_'], label = 'baseline')
    plt.plot( x, comps['L1_'], label = "L1")
#    plt.plot( x, comps['L2_'],  label = "L2")
    plt.plot(x, bkg_y_init, 'c-', label = 'Init bkg')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()

    fig_fit = plt.figure()
    plt.title("Fit at %1.1f GPa" %P)
    plt.plot(x, y, 'bo', label='Data')
    plt.plot(x, out.best_fit, 'r-', label='Fit')
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.legend()
    plt.show()

    fit_curve = np.column_stack((x, out.best_fit))

    #ask if the fit has been successfull
    if tk.messagebox.askokcancel("Check", "Do you want to save the results?"):
        pickle_file = 'best_params.pk'
        with open(pickle_file, 'wb') as pf:
            pickle.dump(best_pars, pf)
        np.savetxt(file.split(".")[0] + '_fit.dat', fit_curve, fmt='%s', delimiter=' ')
        fig_fit.savefig(file.split(".")[0] + '_fit.png')
        with open('fit_results_%icm-1.txt' %peak, "a") as results_file: #prevents overwriting of the previous data in %fit_results_%icm-1.txt
            np.savetxt(results_file, [Pcwa], fmt='%1.2f', delimiter=' ', newline='\n')
    else:
        global fit_success
        fit_success = False
    
    return 

fit_success = True

root = tk.Tk()

#---------------------------Main-----------------------------------

#loading the data
directory= "D:/Dati/Dati chrysene@HP/Chrysene_HP_FTIR/Chrysene_MIR_vs_P_selected/"
#peak analysed (for the file name)
peak = 1598

dest_dir = directory + '/fit_results/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir, exist_ok=True) 

if not os.path.exists(directory + '/fnames_%icm-1.txt' %peak):
    with open(directory + '/fnames_%icm-1.txt' %peak, 'w') as f:
        f.write('empty \n')

sorted_fname = sorted_files(directory)

#fitting the data
for fname in sorted_fname:
    if P_from_filename(fname) >= 6: #Select the pressure values from which you want to start (or restart) the fit procedure
        print(fname + '\n')
        if (".DPT" in fname) and ('OPD' not in fname): #select the files by their name
            with open(directory + '/fnames_%icm-1.txt' %peak, 'r') as f:
                loaded_fname = f.read()
            if fname != loaded_fname:
                fit(directory, fname, dest_dir)
                if not fit_success:
                    break
                else:
                    with open(directory + '/fnames_%icm-1.txt' %peak, 'a') as f:
                        f.write(fname + '\n')
            else:
                print("data already fitted, next ...\n")
        else:
            print(" ...not a data file\n")

root.destroy()