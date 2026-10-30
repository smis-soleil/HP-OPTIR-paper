This folder contains the python scripts developed for the data analysis of ambient and high-pressure OPTIR and FTIR spectra and images of hBN and chrysene samples. 

In particular:
  - Fit of the OPTIR and reflactance spectra of hBN at ambient conditions: "Fit_hBN_OPTIRspectra_ambient-conditions.py" and   "Fit_hBN_Rspectra_ambient-conditions.py" (Fig. 1 in the manuscript),
  - Fit of the OPTIR aspectra of hBN at high pressures: "Fit_hBN_OPTIRspectra_high-pressure.py" (Fig. 2 in the manuscript),
  - Calculation of the average OPTIR intensity over the hBN/chrysene crystal area: "OPTIRimages_avg_intensity.py" (Fig. 3 in the manuscript, and Fig. S2-S6 in the Supplementary Information - SI),
  - Estimate of the spatial resolution of the image from the FWHM of the line spread function relative to the crystal edge profile: "Fit_EdgeProfiles_ESF_SpatialResolution.py" (Fig. 4 in the manuscript, and Fig. S8 in the SI),
  - Fit of the high-pressure FTIR and OPTIR spectra of chrysene: "Fit_chrysene_FTIR+OPTIRspectra_high-pressure.py" (Fig. S2 in the SI).

A brief overview of each Python script's operations and outputs is provided below.

#----------------------Python script 1--------------------------
"Fit_hBN_OPTIRspectra_ambient-conditions.py"

This python code fits the OPTIR spectra of hBN at ambient conditions.
The OPTIR signal is described as I_OPTIR = I0*A(nu), where A(nu) = 1 - T(nu) - R(nu), T and R = transmittance and reflectance of the system.
T and R are computed from the Fresnel's coefficients for a thin dielectric slab (two-interface system).

It returns: 
- a plot of the data + best-fit curve,
- a .txt file with the best-fit parameters, 
- a .txt file with the x, y (OPTIR), y (best-fit curve) data.

#----------------------Python script 2--------------------------
"Fit_hBN_Rspectra_ambient-conditions.py"

This python code fits the IR reflectance spectra of hBN at ambient conditions.
The reflectance R(nu) is computed from the Fresnel's coefficients for a thin dielectric slab (two-interface system).

The program returns: 
- a plot of the data + best-fit curve,
- a .txt file with the best-fit parameters, 
- a .txt file with the x, y (OPTIR), y (best-fit curve) data.

#----------------------Python script 3--------------------------
"Fit_hBN_OPTIRspectra_high-pressure.py"

This python script opens the .ptir file containing the high-pressure OPTIR spectra of the considered sample (here, hBN) and performs a fit of each spectrum through the lmfit package.
The OPTIR signal is fitted as I_OPTIR = I0*A(nu), where A(nu) = 1 - T(nu) - R(nu), T and R = transmittance and reflectance of the system.
T and R are computed from the Fresnel's coefficients for a 2-interface system (thin dielectric slab).

The program returns: 
- a plot of the data + best-fit curves (normalised and stacked for clarity), 
- a plot of the best-fit parameters,
- a .txt file with the best-fit parameters,
- .txt files with the best-fit curves (one file for each high-pressure spectrum).

#----------------------Python script 4--------------------------
"OPTIRimages_avg_intensity.py"

This python script computes the average intensity of the single-wavelength OPTIR images of hBN considering only the crystal area as selected from the corresponding DC images.
A chosen threshold of the DC signal is used to select the sample.

The program returns:
- a plot with the average intensity data,
- a plot of the OPTIR images with the corresponding masks on the sample area, as obtained from the DC images,
- a .txt file with the statistics (avg, min, max) of each OPTIR image's intensity.

#----------------------Python script 5--------------------------
"Fit_EdgeProfiles_ESF_SpatialResolution.py"

This python script reads a linescan across the crystal edge profile and fits it thorugh a sigmoidal function to obtain the edge spread function (ESF).
It computes its derivative (the line spread function - LSF) and the FWHM of the LSF. The latter represents the spatial resolution of the corresponding image.

The program gives in output: 
- a plot of the linescan + best-fit sigmoidal curve,
- a plot of the linescan + LSF + a light-shaded area which indicates the FWHM of the LSF,
- a .txt file with the best-fit parameters.
  
The best-fit parameters and the FWHM of the LSF are written on the terminal.

#----------------------Python script 6--------------------------
"Fit_chrysene_FTIR+OPTIRspectra_high-pressure.py"

This script fits the data through a linear baseline + peak model (possiblity to increase the number of peaks). 
It reads the data file and selects a x range of the data where it performs the fit.

It gives in output:
  - a figure with the data in the selected range, the initial fit curves and the best-fit ones,
  - a figure with the data + the best-fit curve,
  - a window asking if you want to save the fit results. 
  If yes --> the program produces a .txt file with the best-fit results and a .png image with the data + best-fit curve, and you can procede by fitting the higher pressure spectra.
  If no --> the program stops.
  The best-fit values from the last succefull fit are stored in a file. You can re-start the fitting from the last non-successfull fit.
  - a .txt file ('fit_results_%icm-1.txt' %peak) where the best-fit parameters at each pressure are stored.

