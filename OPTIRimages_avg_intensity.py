"""
@author: BDAlo 

This script compute the average intensity of the single-wavelength OPTIR images of hBN considering only the crystal area as selected from the corresponding DC images.
A chosen threshold of the DC signal is used to select the sample.

The program returns:
- a plot with the average intensity data 
- a plot of the OPTIR images with the corresponding masks on the sample area, as obtained from the DC images
- a txt file with the statistics (avg, min, max) of each OPTIR image's intensity
"""
import h5py
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from gsffile import write_gsf #read_gsf, 
import re

##---------------------------DEFINITIONS---------------------------------
#Reading the .ptir file

#This function returns the labels of each image/spectrum inside the PTIR file
def data_label(h5):
    return list(h5['MEASUREMENTS'].keys())

#This function returns the name of each image/spectrum inside the PTIR file
#Note: decoding the name in the UTF-8 format is needed
def data_name(h5):
    map_label = list(h5['MEASUREMENTS'].keys())
    N = len(h5['MEASUREMENTS'])
    map_name = ['0']*N #Creating a list with zeros of dimension N, to host the name
    for i in range(0, N):
        map_name[i] = h5['MEASUREMENTS/%s' %map_label[i]].attrs['Label'].decode('utf8')
        #print('Name of map n.%i: %s' %(i, map_name[i]))
    return map_name

#This function receive a .hdf file in input and produces corresponding .txt files for each image. To check the output, it plots the images 
def save_data(h5, path):
    map_label = list(h5['MEASUREMENTS'].keys())
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    fig, ax = plt.subplots(1, N) #Horizontally stacked plots
    for i in range(0, N):
        data = h5['MEASUREMENTS/%s/DATA' %map_label[i]]
        #writing the image data in a gsf file
        obj = GwyContainer()
        obj['/0/data/title'] = '%s' %map_name[i]   
        write_gsf(r'%s\%s.gsf' %(path, map_name[i]), data)
        #np.savetxt(r'%s\%s.txt' %(path, map_name[i]), data, fmt='%f')
        ax[i].imshow(data)
        ax[i].set_title('%s' %map_name)
    plt.show()
    return print('Image saved!')

#This function finds the pressure value from the image/spectrum name
def pressure_value(map_name):
    match = re.search(r'P(\d+(?:[a-zA-Z]\d+)?)', map_name)
    if match:
        p_value = match.group(1)
        print('p = %s' %p_value)
    if 'p' in p_value:
        p = float(p_value.replace('p', '.'))
    else: 
        p = float(p_value)
    print('p = %.1f' %p)
    return p

#This function reads the pressure values from the .txt file with measurement parameters of each image, and returns an array with the pressures
def pressure_values_dataset(h5, path):
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    P = np.zeros(N)
    for i in range(0, N):
        params = np.loadtxt(r'%s\%s_params.txt' %(path, map_name[i]), skiprows=1)
        P[i] = params[0]
    return P

#This function reads and returns an array with the wavelengths at which the OPTIR images were acquired 
def nu_values_dataset(h5, path):
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    nu = np.zeros(N)
    for i in range(0, N):
        params = np.loadtxt(r'%s\%s_params.txt' %(path, map_name[i]), skiprows=1)
        nu[i] = params[1]
    return nu

#This function reads and returns an array with the probe power at which the OPTIR images were acquired 
def probe_power_dataset(h5, path):
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    probe_power = np.zeros(N)
    for i in range(0, N):
        params = np.loadtxt(r'%s\%s_params.txt' %(path, map_name[i]), skiprows=1)
        probe_power[i] = params[3]
    return probe_power

#This function reads and returns an array with the IR power at which the OPTIR images were acquired 
def IR_power_dataset(h5, path):
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    IR_power = np.zeros(N)
    for i in range(0, N):
        params = np.loadtxt(r'%s\%s_params.txt' %(path, map_name[i]), skiprows=1)
        IR_power[i] = params[2]
    return IR_power

#This function reads and returns an array with the IR power at whichthe OPTIR images were acquired 
def gain_dataset(h5, path):
    map_name = data_name(h5)
    N = len(h5['MEASUREMENTS'])
    gain = np.zeros(N)
    for i in range(0, N):
        params = np.loadtxt(r'%s\%s_params.txt' %(path, map_name[i]), skiprows=1)
        gain[i] = params[4]
    return gain



#This function finds the corresponding DC and O-PTIR images and returns a np.array with three columns: (P, i, j) where P is the pressure value
# i is the index of the OPTIR image taken at P and j is the index of the corresopnding DC image

def match_maps(map_name, map_name_DC, P, P_DC):
    match_P = np.zeros((N,3))
    match_P[:, 0] = P
    for i in range(0, N):
        name = map_name[i].replace("OPTIR", "{}")
        map = np.loadtxt('%s/%s.txt' %(path_txt, map_name[i]))
        for j in range(0, N):
            name_DC = map_name_DC[j].replace("DC", "{}")
            DC_map = np.loadtxt('%s/%s.txt' %(path_txt, map_name_DC[j]))
            if name == name_DC and map.shape == DC_map.shape and P[i] == P_DC[j]:
                #print("OPTIR image %i matches with DC image %i" %(i, j))
                match_P[i, 1] = i
                match_P[i, 2] = j
            if (j == N-1)  and (match_P[i, 1] == 0) and (match_P[i, 2] == 0):
                print ('Error! No correspondence found for P = %s GPa, OPTIR image = %s' %(P[i], map_name[i]))
        #else:
        #    print("Match NOT found for OPTIR image %i" %(i))
#print(match_P)
    return match_P

#defining a function to find the sample position from the DC image
def sample_pos(DC_map, DC_lim, i_min, i_max, length):
    #first selection of the sample shape in the DC image (sample pos = 1, 0 elsewhere)
    map_dim = DC_map.shape
    sample_pos = np.zeros(map_dim)
    for j in range(0, map_dim[1]):
        for i in range(0, map_dim[0]):
            if (DC_map[i,j] <= DC_lim):
                sample_pos[i,j] = 1
    #filling the 'voids' in the sample_pos image
    n = length #number of voids in the image (at each row)
    j_min = np.zeros(n, dtype = int) #array to save the initial point of the voids in a single row
    j_max = np.zeros(n, dtype = int) #array to save the last point of the voids in a single row
    for i in range(i_min, i_max):
        k, l = 0, 0
        for j in range(1, map_dim[1]-1):
            if sample_pos[i,j-1] == 0 and sample_pos[i,j] == 1:
                j_min[k] = j
                k += 1
                #print('for j = %i, k = %i'%(j, k))
            if sample_pos[i,j] == 1 and sample_pos[i,j+1] == 0:
                j_max[l] = j
                l += 1
                #print('for j = %i, l = %i'%(j, l))
            if sample_pos[i,j] == 1 and j == min(range(1, map_dim[1]-1)): #takes into account the initial edge of the image = 1
                j_max[l] = j
                l += 1
                #print('for j = %i, l = %i'%(j, l))
            if sample_pos[i,j] == 1 and j == max(range(1, map_dim[1]-1)): #takes into account the edge of the image = 1
                j_max[l] = j
                l += 1
                #print('for j = %i, l = %i'%(j, l))            
        for j in range(j_min[0], j_max[l-1]):
            sample_pos[i,j] = 1
            #print('j_min = %i, j_max = %i' %(j_min[0], j_max[l-1]))
    #plt.imshow(DC_map, cmap='viridis')
    #plt.imshow(sample_pos, cmap = 'Greys', alpha = 0.2)
    #plt.title('Sample position at %.2f GPa' %P)
    return sample_pos

def n_points(sample_pos):
    map_dim = sample_pos.shape
    N_points = 0
    for i in range(0, map_dim[0]):
        for j in range(0, map_dim[1]):
            if sample_pos[i,j] == 1:
                N_points += 1
    return N_points


##-----------------------------------------MAIN------------------------------------------
#Opening the PTIR file (containing the DC or OPTIR images only) into a h5py File object (h5)
path_0 = 'file_path' # 'file_path'
file_name_DC = 'PTIR_DCimages_file_name' # 'PTIR_DCimages_file_name'
f_DC= r'%s\%s.ptir' %(path_0, file_name_DC)
h5_DC = h5py.File(f_DC, 'r')
print('Reading the DC map file: ok')
file_name = 'PTIR_OPTIRimages_file_name'
f = r'%s\%s.ptir' %(path_0, file_name)
h5 = h5py.File(f, 'r')
print('Reading the OPTIR images file: ok')

#sample name for the results' files
sample_name = 'sample name' #'380nm-thick_hBN'
#folder where saving the results
path = 'results_folder_path'

#Counting the number of OPTIR images (should be == number of DC images) into the file
N_DC = len(h5_DC['MEASUREMENTS'])
N = len(h5['MEASUREMENTS'])
print('Number of maps: N=%i' %N)
if N_DC != N:
    print('Error! Number of DC images != number of OPTIR images')
    exit(1)

#Selecting the name of the OPTIR or DC images as written in the PTIR file
map_name_DC = data_name(h5_DC)
map_name = data_name(h5)

#Selecting the pressure and wavelength values of the dataset
path_txt = 'txt_folder_path'
P = pressure_values_dataset(h5, path_txt)
P_DC = pressure_values_dataset(h5_DC, path_txt)
nu = nu_values_dataset(h5, path_txt)
IR_power = IR_power_dataset(h5, path_txt)
probe_power = probe_power_dataset(h5, path_txt)
gain = gain_dataset(h5, path_txt)

#Coupling the corresponding DC and OPTIR maps
match_P = match_maps(map_name, map_name_DC, P, P_DC)
print(match_P)

#creating a numpy array for the map statistics 
stats = np.full((N, 4), np.nan)
stats[:, 0] = P

#sample selection and computation of the avg intensity of the OPTIR images from the corresponding DC images (or directly from the OPTIR ones, if convenient)
#In the for cycle, each image (or range of images) is selected through its name/pressure value, and the covenient DC/OPTIR threshold is used to select the sample area from the bkg
#In genral, images with Probe Power = 74% at HP are selected
fig_map, ax_map = plt.subplots(3,5, constrained_layout=True)

p=0
for i in range(0, N):    
    DC_map = np.loadtxt('%s/%s.txt' %(path_txt, map_name_DC[int(match_P[i, 2])])) #loading of the DC image
    map = np.loadtxt('%s/%s.txt' %(path_txt, map_name[int(match_P[i, 1])]))
    i_min = 0
    i_max = DC_map.shape[0]
    length = i_max
    if (('open' in map_name[i]) and (P[i] == 0) and ('back' not in map_name[i])): #open DAC, ambient conditions
        DC_lim = 3.28 #Value of the DC limit conveniently chosen to have sample/bkg contrast
        print('DC lim = %.1f at P = %.1f, open cell (i = %i)' %(DC_lim, P[i], i))
        sample_pos_P = sample_pos(DC_map, DC_lim, i_min, i_max, length)
        n_points_sample = n_points(sample_pos_P)
        mean = np.mean(map*sample_pos_P)*map.size/n_points_sample
        stats[i, 1] = mean
        stats[i, 2] = np.min(map*sample_pos_P)
        stats[i, 3] = np.max(map*sample_pos_P)
        ax_map[p//5][p%5].imshow(map, cmap='viridis')
        ax_map[p//5][p%5].imshow(sample_pos_P, cmap = 'Greys', alpha = 0.2)
        ax_map[p//5][p%5].set_title('%.1f GPa' %P[i], fontsize = '8')
        ax_map[p//5][p%5].set_xticks([])
        ax_map[p//5][p%5].set_yticks([])
        i_opencell = i
        p += 1
    if (('closed' in map_name[i]) and (P[i] == 0) and ('back' not in map_name[i]) and ('ok' in map_name[i])): #closed DAC
        DC_lim = 5.88 #Value of the DC limit conveniently chosen to have sample/bkg contrast
        print('DC lim = %.1f at P = %.1f, closed cell (i = %i)' %(DC_lim, P[i], i))
        sample_pos_P = sample_pos(DC_map, DC_lim, i_min, i_max, length)
        n_points_sample = n_points(sample_pos_P)
        mean = np.mean(map*sample_pos_P)*map.size/n_points_sample
        stats[i, 1] = mean
        stats[i, 2] = np.min(map*sample_pos_P)
        stats[i, 3] = np.max(map*sample_pos_P)
        ax_map[p//5][p%5].imshow(map, cmap='viridis')
        ax_map[p//5][p%5].imshow(sample_pos_P, cmap = 'Greys', alpha = 0.2)
        ax_map[p//5][p%5].set_title('%.1f GPa' %P[i], fontsize = '8')
        ax_map[p//5][p%5].set_xticks([])
        ax_map[p//5][p%5].set_yticks([])
        i_closedcell = i
        p += 1
    elif ((probe_power[i] == 74) and ('back' not in map_name[i]) and ('no' not in map_name[i]) and (P[i] > 0) and (P[i] < 9.2)):
        DC_lim = (np.min(DC_map)+np.max(DC_map))/2 -0.25 #Value of the DC limit conveniently chosen to have sample/bkg contrast
        print('DC lim = %.1f at P = %.1f' %(DC_lim, P[i]))
        sample_pos_P = sample_pos(DC_map, DC_lim, i_min, i_max, length)
        n_points_sample = n_points(sample_pos_P)
        mean = np.mean(map*sample_pos_P)*map.size/n_points_sample
        stats[i, 1] = mean
        stats[i, 2] = np.min(map*sample_pos_P)
        stats[i, 3] = np.max(map*sample_pos_P)
        ax_map[p//5][p%5].imshow(map, cmap='viridis')
        ax_map[p//5][p%5].imshow(sample_pos_P, cmap = 'Greys', alpha = 0.2)
        ax_map[p//5][p%5].set_title('%.1f GPa' %P[i], fontsize = '8')
        ax_map[p//5][p%5].set_xticks([])
        ax_map[p//5][p%5].set_yticks([])
        p += 1
        #print(p)
        if P[i] == 5.2:
            sample_pos_5p2 = sample_pos_P

for i in range(0, N):
    if ((probe_power[i] == 74) and ('back' not in map_name[i]) and (P[i] > 9.2)):
    #ALTERNATIVA: dare le coordinate in i,j del campione!!
    #DC_lim = 2.88
    #print('Cycle 2: DC lim = %.1f at P = %.1f' %(DC_lim, P[i]))
    #sample_pos_P = sample_pos(DC_map, DC_lim, P[i], i_min, i_max, length)
        map = np.loadtxt('%s/%s.txt' %(path_txt, map_name[int(match_P[i, 1])]))
        i_min = 0
        i_max = DC_map.shape[0]
        length = i_max
        sample_pos_P = sample_pos_5p2
        n_points_sample = n_points(sample_pos_P)
        mean = np.mean(map*sample_pos_P)*map.size/n_points_sample
        stats[i, 1] = mean
        stats[i, 2] = np.min(map*sample_pos_P)
        stats[i, 3] = np.max(map*sample_pos_P)
        ax_map[p//5][p%5].imshow(map, cmap='viridis')
        ax_map[p//5][p%5].imshow(sample_pos_P, cmap = 'Greys', alpha = 0.2)
        ax_map[p//5][p%5].set_title('%.1f GPa' %P[i], fontsize = '8')
        ax_map[p//5][p%5].set_xticks([])
        ax_map[p//5][p%5].set_yticks([])
        #print(p)
        p += 1

plt.savefig('%s\Intensity_images_sample_selection_%s.png' %(path, sample_name), dpi = 360)

#saving data and statistics
data_P = np.column_stack([stats, nu, IR_power, probe_power, gain])
header_text = f'P(GPa)  avg (mV)  min (mV)  max (mV)  nu (cm-1) IR power(%) Probe power(%) Gain(x) \n i opencell = {i_opencell}, i closed cell = {i_closedcell}'
np.savetxt('%s\HP_intensity_stats_%s.txt' %(path, sample_name), data_P, header=header_text, fmt='%f') #\n comments=''

#plotting the results
#avgs
fig, ax = plt.subplots()

#shift = 0.92
#I_max = np.max(stats[:, 1]
ax.scatter(stats[:, 0], stats[:, 1], color = "mediumturquoise", linewidth = 2) # label = "I$_{map}$"
ax.scatter(stats[i_opencell, 0], stats[i_opencell, 1], color = "red", linewidth = 2, label = "open cell")
ax.scatter(stats[i_closedcell, 0], stats[i_closedcell, 1], color = "lawngreen", linewidth = 2, label = "closed cell")

x_min, x_max = -0.4, 12.2
ax.set_xlim(x_min, x_max)
ax.set_yscale('log')
ax.set_title('HP - 380 nm-thick hBN')
ax.set_xlabel('P (GPa)')
ax.set_ylabel('I$_{map}$')
ax.legend()

plt.savefig('%s\Intensity_image_avg_%s.png' %(path, sample_name), dpi = 360)

plt.show()