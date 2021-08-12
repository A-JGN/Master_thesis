#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:34:46 2021

@author: Anna Jegen an Daniel Köhn
"""
import glob
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150
import matplotlib.pyplot as plt
import math
import numpy as np
import segyio

def open_shotfile(file_dir_n_names):
    with segyio.su.open(file_dir_n_names,endian='little',ignore_geometry=True) as f: 
        data = f.trace.raw[:]
    return data

def plot_shot(idx,title,row, col, pclip):
   shot = file_dicc['model_data' +str(idx)]
   vmax = pclip * np.max(np.abs(shot))
   vmin = - vmax
   fig.add_subplot(col,row, idx)
   plt.imshow(shot, cmap='Greys', vmin=vmin, vmax=vmax)
   plt.title(title) 
   plt.axis('auto')


## model information:
###path 2 file
path = "model_4/FWI/GRAD_FORM2/dirmute_nocorr/"
##filename
file = "DENISE_MARMOUSI_y.su.shot1.it"
file_dir_n_names = path + file + "*"
font = {'color':  'black',
        'weight': 'normal',
        'size': 20
        }
files =glob.glob(file_dir_n_names, recursive=False)
## create a dictionary with names of file and model type
file_dicc ={}
print(files)
plt.close('all')
fig = plt.figure(figsize=(10,10))
col=2 
row = math.ceil(len(files)/col)
for ifile in files:
    idx = model_type = str(ifile.split('it')[-1])
    file_dicc['file_name' +str(idx)] = str(ifile)
    file_dicc['model_data' +str(idx)] = open_shotfile(str(ifile))
    file_dicc['model_'+str(idx)+'minVal'] = np.min(file_dicc['model_data' +str(idx)])
    file_dicc['model_'+str(idx)+'maxVal'] = np.max(file_dicc['model_data' +str(idx)])
    print(str(idx))
    if idx == "1": 
       title = "adjoint source field, " + "fmax = 2 Hz"
    elif idx == "2":
        title = "adjoint source field, " + "fmax = 5 Hz"
    elif idx == "3":
        title = "adjoint source field, " + "fmax = 10 Hz"
    elif idx == "4":
        title = "adjoint source field, " + "fmax = 20 Hz"
    plot_shot(idx, title, row,col,pclip=0.2)
    plt.tight_layout()
plt.show()
