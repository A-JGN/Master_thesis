#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 18:18:58 2021

@author: Anna
"""
from seiscm import bwr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import rcParams
from matplotlib.pyplot import gca
import scipy.ndimage.filters
import scipy.ndimage as ndimage
import segyio
import os

import glob
def mask_array(Array, mask):
    Zaehler=0
    shape=list(Array.shape)
    fmask=mask.flatten()
    farray=Array.flatten()
    for i in fmask:
        if abs(float(i)) < 1:
            farray[Zaehler]*=0
        Zaehler+=1
    return np.reshape(farray, shape)
def load_shotfile(path2file,file_is_su):
    if file_is_su:
       with segyio.su.open(path2file,endian='little', ignore_geometry=True) as srcf:
        exp_shot = srcf.trace.raw[:] 
    if not file_is_su:
       with segyio.open(path2file,ignore_geometry=True) as srcf:
        exp_shot = srcf.trace.raw[:]
    return exp_shot
def load_stack(path2file, file_is_su):
    if file_is_su:
       with segyio.su.open(path2file,endian='little', ignore_geometry=True) as srcf:
        stack = srcf.trace.raw[:]
        dt =(srcf.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
    if not file_is_su:
       with segyio.open(path2file,ignore_geometry=True) as srcf:
        stack = srcf.trace.raw[:]
        dt =(srcf.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
    return stack, dt
def load_attributes(file,sigma, downS):
        rawdatafile=file
        # Import seismic shot gathers & headers
        with segyio.su.open(rawdatafile,endian='little', ignore_geometry=True) as f:
            # Get basic attributes
            Attb= f.trace.raw[:]  # Get all data into memory (could cause on big files)
            ntr, ns = np.shape(Attb)
            Attb = ndimage.median_filter(Attb,sigma)
            Attb = Attb[::,::downS]
        return Attb

def open_FWI_model_file(file,grid_params): 
    f = open(file)
    data_type = np.dtype ('float32').newbyteorder ('<') 
    model_data = np.fromfile (f, dtype=data_type) 
    print("shape:" + str(model_data.shape))
    model_data = model_data.reshape(grid_params['NX'],grid_params['NY']) 
    model_data = np.transpose(model_data) 
    model_data = np.flipud(model_data)
    return model_data

def export_model(model, filename):
    model= np.transpose(model)
    model = np.fliplr(model)
    afile = open(filename, "w")
    a = np.array(model,'float32')
    output_file = open(filename, 'wb')
    a.tofile(output_file)
    output_file.close()

def load_jacobian(path2data, filename, NX, NY, Laplace_B):
    filepos= path2data + filename
    f1 = open(filepos)
    data_type = np.dtype ('float32').newbyteorder ('<')
    RTM = np.fromfile (f1, dtype=data_type)
    RTM = RTM.reshape(NX,NY)
    RTM = np.transpose(RTM)
    RTM = np.flipud(RTM)
    #if Laplace_B == True:
    RTM = scipy.ndimage.filters.laplace(RTM) # suppress low-wavenumber artifacts in image 
    return RTM
def plot_stack_overvp(stack, vp, grid_params, dx_cdp, dz_cdp, title="", colorscheme='magma', pclip=1):
    FSize = 12
    font = {'color':  'black',
        'size': FSize}
    mpl.rc('xtick', labelsize=FSize) 
    mpl.rc('ytick', labelsize=FSize) 
    rcParams['figure.figsize'] = 15, 11
    
    extent = [0.0,grid_params['NX']*grid_params['DX']/1000.0,0.0,grid_params['NY']*grid_params['DX']/1000]
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text', usetex=False)
    plt.axes()
    plt.imshow(vp, cmap=colorscheme, interpolation='nearest', extent=extent, vmin=np.min(vp), vmax=3449.7659)
    cbar= plt.colorbar( shrink=0.4)
    cbar.set_label('m/s', fontdict=font, labelpad=1)
    cmax=7
    cmin=-cmax
    extent_stack=[0.0,(np.shape(stack)[0]*dx_cdp)/1000.0,0.0,(np.shape(stack)[1]*dz_cdp)]
    plt.imshow(np.flipud(stack.T), cmap=plt.cm.gray, alpha=.2, interpolation='bicubic',
                 extent=extent_stack, vmin=cmin, vmax=cmax)
    cbar.set_label('m/s', fontdict=font, labelpad=1)
    a = gca()
    a.set_aspect(2)
    plt.title(title, fontdict=font)
    plt.ylabel('Depth [km]', fontdict=font)
    plt.xlabel('Distance [km]', fontdict=font)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.pause(0.001)
    plt.show()
def plot_jacobian(RTM, vp, NX, NY, DH, title, out_file_name, wfile_B, jacobian_B, colorscheme, pclip=1):
    FSize = 25
    font = {'color':  'black',
        'size': FSize}
    mpl.rc('xtick', labelsize=FSize) 
    mpl.rc('ytick', labelsize=FSize) 
    rcParams['figure.figsize'] = 15, 11
    extent = [400*DH/1000,(400+NX)*DH/1000.0,0.0,NY*DH/1000]
    cmax=130
    cmin=0
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text', usetex=False)
    plt.axes()
    plt.imshow(vp, cmap=colorscheme, interpolation='nearest', extent=extent, vmin=cmin, vmax=cmax)
    cbar= plt.colorbar( shrink=0.45)
    cbar.set_label('m/s', fontdict=font, labelpad=1)
    if jacobian_B == True: 
        plt.imshow(RTM, cmap=plt.cm.gray, alpha=.5, interpolation='bicubic',
                 extent=extent, vmin=cmin, vmax=cmax)
        cbar.set_label('m/s', fontdict=font, labelpad=1)
    a = gca()
    plt.title(title, fontdict=font)
    plt.ylabel('Depth [km]', fontdict=font)
    plt.xlabel('Distance [km]', fontdict=font)
    plt.gca().invert_yaxis()
    #cbar=plt.colorbar()
    #cbar.set_label('dVp[m/s]', fontdict=font, labelpad=1)
    plt.tight_layout()
    if wfile_B == True:
        figname= out_file_name + '.pdf'
        plt.savefig(figname, bbox_inches='tight', format='pdf')
    #plt.savefig('Marmousi_RTM.png', format='png')
    plt.pause(0.001)
    plt.show()
    print('min value ' +str(np.min(RTM)))
    print('max value ' +str(np.max(RTM)))
    
def plot_models(model, title, cmap,vmin, vmax, DX, DT, cbar_label=' '):
    FSize = 12
    NX = model.shape[0]
    NY = model.shape[1]
    font = {'color':  'black',
        'size': FSize}
    mpl.rc('xtick', labelsize=FSize) 
    mpl.rc('ytick', labelsize=FSize) 
    rcParams['figure.figsize'] = 10, 9
    if DX == DT:
        extent = [0.0,NX*DX/1000.0,0.0,NY*DT/1000]
    elif DX!=DT:
        extent = [0.0,NX*DX/1000.0,0.0,NY*DT]
    plt.imshow(np.flipud(model.T), cmap=cmap,  interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax);
    cbar= plt.colorbar(shrink=0.45)
    cbar.ax.locator_params(nbins=5)
    
    cbar.ax.tick_params(labelsize=FSize) 
    #cbar.ax.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title(title, fontdict=font) 
    plt.axis()
    cbar.set_label(cbar_label, fontdict=font, labelpad=1)
    if DX == DT:
        plt.ylabel('Depth [km]', fontdict=font)
    elif DX != DT:
        plt.ylabel('TWT [s]', fontdict=font)
    plt.xlabel('Distance [km]', fontdict=font)
    plt.gca().invert_yaxis()
    plt.pause(0.001)
    
def plot_closeup_model(model,cmap, vmin, vmax, DX, DT, x_min,x_max, z_min, z_max,RTM=0, cbar_label='m/s'):
    FSize = 12
    cmax=8e-6
    cmin=-cmax
    x_min_idx= int(x_min*1000/DX)
    x_max_idx= int(x_max*1000/DX)
    if DX != DT:
        z_min_idx= int(z_min/DT)
        z_max_idx= int(z_max/DT)
        extent = [x_min_idx*DX/1000,x_max_idx*DX/1000.0,z_min_idx*DT,z_max_idx*DT]
    if DX == DT:
        z_min_idx= int(z_min*1000/DT)
        z_max_idx= int(z_max*1000/DT)
        extent = [x_min_idx*DX/1000,x_max_idx*DX/1000.0,z_min_idx*DT/1000,z_max_idx*DT/1000]
        
    font = {'color':  'black',
        'size': FSize}
    mpl.rc('xtick', labelsize=FSize) 
    mpl.rc('ytick', labelsize=FSize) 
    rcParams['figure.figsize'] = 10, 9
    model=np.flipud(model)[z_min_idx:z_max_idx, x_min_idx:x_max_idx]
    plt.imshow(np.flipud(model), cmap=cmap,  interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax);
    cbar= plt.colorbar(shrink=0.4)
    if np.size(RTM) != 1: 
        RTM=np.flipud(RTM)[z_min_idx:z_max_idx, x_min_idx:x_max_idx]
        plt.imshow(np.flipud(RTM), cmap=plt.cm.gray, alpha=.3, interpolation='bicubic',
                 extent=extent, vmin=cmin, vmax=cmax)
        cbar.set_label('m/s', fontdict=font, labelpad=1)
    cbar.ax.tick_params(labelsize=FSize) 
    #cbar.ax.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.axis()
    cbar.set_label(cbar_label, fontdict=font, labelpad=1)
    if DX == DT:
        plt.ylabel('Depth [km]', fontdict=font)
    elif DX != DT:
        plt.ylabel('TWT [s]', fontdict=font)
    plt.xlabel('Distance [km]', fontdict=font)
    plt.gca().invert_yaxis()
    plt.pause(0.001)
          
def simple_plot_models(model, title, cmap, vmin, vmax):
    plt.figure(figsize=(10,5), dpi=100)
    plt.imshow(model.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis('auto')
    plt.pause(0.001)
    
def plot_shot(shot, title, dt, pclip=1.0):
   FSize = 14
   font = {'color':  'black',
        'size': FSize}
   mpl.rc('xtick', labelsize=FSize) 
   mpl.rc('ytick', labelsize=FSize) 
   vmax = pclip * np.max(np.abs(shot))
   vmin = - vmax
   extent=  [0,shot.shape[0],0.0,shot.shape[1]*dt]
   plt.figure(figsize=(8,4), dpi=700)
   plt.imshow(np.flipud(shot.T), cmap=bwr(),extent=extent, vmin=vmin, vmax=vmax)
   plt.gca().invert_yaxis()
   plt.title(title, fontdict=font)
   plt.axis('auto')
   plt.ylabel('TWT [s]', fontdict=font)
   plt.xlabel('traces', fontdict=font)
   plt.pause(0.001)
   
def plot_stack(shot, title, dt,dx, pclip=1.0):
   FSize = 14
   font = {'color':  'black',
        'size': FSize}
   mpl.rc('xtick', labelsize=FSize) 
   mpl.rc('ytick', labelsize=FSize) 
   vmax = pclip * np.max(np.abs(shot))
   vmin = - vmax
   extent=  [83*dx/1000,shot.shape[0]*dx/1000,0.0,shot.shape[1]*dt]
   plt.figure(figsize=(8,4), dpi=700)
   plt.imshow(np.flipud(shot.T), cmap=bwr(),extent=extent, vmin=vmin, vmax=vmax)
   plt.gca().invert_yaxis()
   plt.title(title, fontdict=font)
   plt.axis('auto')
   plt.ylabel('TWT [s]', fontdict=font)
   plt.xlabel('Distance [km]', fontdict=font)
   plt.pause(0.001)

def calc_mean_model_from_files(grid_params, nmodels, path, file_pattern):
    i=0
    sum_model= np.zeros((grid_params['NY'], grid_params['NX'],nmodels))
    for filename in glob.glob(os.path.join(path,file_pattern)):
        with open(filename, 'r') as f:
            data_type = np.dtype ('float32').newbyteorder ('<')
            vp = np.fromfile (f, dtype=data_type)
            vp = vp.reshape(grid_params['NX'],grid_params['NY'])
            vp = np.transpose(vp)
            vp = np.flipud(vp)
        sum_model[:,:,i]=vp[:]
        i+=1
    return np.mean(sum_model,axis=2), np.std(sum_model, axis=2)