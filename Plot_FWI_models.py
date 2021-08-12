#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:59:15 2021

@author: Anna and Daniel Köhn
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pylab import rcParams
from matplotlib import rc

import pickle
import glob

##Functions
def open_FWI_model_file(file,grid_params): 
    f = open (file)
    data_type = np.dtype ('float32').newbyteorder ('<') 
    model_data = np.fromfile (f, dtype=data_type) 
    model_data = model_data.reshape(grid_params['NX'],grid_params['NY']) 
    model_data = np.transpose(model_data) 
    model_data = np.flipud(model_data)
    return model_data

def plot_model_data(n,cm, an, title):
    ax=plt.subplot(3, 1, n)
    data = file_dicc['model_data' +str(idx)]
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=False)
    y = np.arange(0, ((grid_params['NY']+1)* grid_params['DH']), grid_params['DH'])
    x = np.arange(0, ((grid_params['NX']+1)* grid_params['DH']), grid_params['DH'])
    #plt.figure(figsize=(15,5), dpi=450); 
    plt.imshow(data, cmap=cm,interpolation='none',extent=[y[0],x[-1]/1000.0,y[0],y[-1]/1000.0], vmin=file_dicc['model_'+str(idx)+'minVal'], vmax=file_dicc['model_'+str(idx)+'maxVal']); 
    plt.title(title);
    ax = plt.gca();
    ax.set_xticklabels(ax.get_xticks(), font)
    ax.set_yticklabels(ax.get_yticks(), font)
    plt.axis('scaled')
    plt.ylabel('Depth [km]', fontdict=font)
    if n==3:
        plt.xlabel('Distance [km]', fontdict=font)
    plt.gca().invert_yaxis()
    cbar=plt.colorbar(aspect=8, pad=0.02)
    cbar.set_label(title, fontdict=font, labelpad=10)
    plt.text(0.1, 0.32,an,fontdict=font,color='white')
    plt.tight_layout()

###Layout stuff
FSize = 20
font = {'color':  'black',
        'weight': 'normal',
        'size': FSize
        }
mpl.rc('xtick', labelsize=FSize) 
mpl.rc('ytick', labelsize=FSize) 
rcParams['figure.figsize'] = 12, 11

##load colourmap
fp = open('model_4/cmap_cm.pkl', 'rb')
my_cmap_cm = pickle.load(fp)
fp.close()

### model information:
path= "model_4/FWI/GRAD_FORM2/dirmute_nocorr/"
#path= "model_4/FWI/dampR/damp20_a09/"
basename ="modelTest_"
stage = 4;
ending = "_stage_" + "%0.*f" %(0,np.fix(stage)) + ".bin"
file_dir_n_names = path + basename + "*" + ending

###define dictionary with the grid parameters od the model
grid_params={ 
    'DH' : 20, 
    'NX' : 500, 
    'NY' : 174 ,
    'dxticks' : float(1),
    'dyticks' : float(0.5)
    }


###create list of all files in selected folder that suit the pattern
files =glob.glob(file_dir_n_names, recursive=False)
## create a dictionary with names of file and model type
file_dicc ={}
print(files)
idx=1
plt.close('all')
plt.figure()
for ifile in files:
    file_dicc['file_name' +str(idx)] = str(ifile)
    model_type = str(str(ifile.split(path+basename)[1]).split(ending)[0])
    file_dicc['model_type' +str(idx)] = model_type
    file_dicc['model_data' +str(idx)] = open_FWI_model_file(file_dicc['file_name' +str(idx)], grid_params)
    file_dicc['model_'+str(idx)+'minVal'] = np.min(file_dicc['model_data' +str(idx)])
    file_dicc['model_'+str(idx)+'maxVal'] = np.max(file_dicc['model_data' +str(idx)])
    if file_dicc['model_type' +str(idx)] == 'vp':
        title = r"$\rm{V_p [m/s]}$"
    if file_dicc['model_type' +str(idx)] == 'vs':
        title = r"$\rm{V_s [m/s]}$"
    if file_dicc['model_type' +str(idx)] == 'rho':
        title = r"$\rm{\rho [kg/m^3]}$"
    plot_model_data(idx, 'magma', str(idx), title)
    idx +=1
plt.show()


