#!/usr/bin/env python
# coding: utf-8

# # Processing of Common shot gathers for diffraction FWI 
# *(object oriented) <br>
# Wellenfront Attribute und Stack Section sind mit dem globalen CRS Code 2016 bestimmt worden.

# In[1]:


import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150
import matplotlib.pyplot as plt
import sys
import numpy as np
import segyio
import math
import scipy.ndimage as ndimage


# In[2]:


class dataset:
    def __init__(self, data,ntr, ns, offset, CDP, sourceX, receivX, nCDP, offset_cdp, cdp_traces):
        self.data=data
        self.ntr=ntr
        self.ns=ns
        self.offset=offset
        self.CDP=CDP
        self.SourceX=sourceX
        self.receivX=receivX
        self.ncdp=nCDP
        self.offset_cdp=offset_cdp
        self.cdp_traces=cdp_traces
        
    def load_section(self, path2data, oname, data, identifier, downS):
        rawdatafile=path2data+ data
        # Import seismic shot gathers & headers
        with segyio.su.open(rawdatafile,endian='little', ignore_geometry=True) as f:
            # Get basic attributes
            self.data = f.trace.raw[:]# Get all data into memory (could cause on big files)
            self.ntr, self.ns = np.shape(self.data)
            ##downsample
            self.data = self.data[:,::downS]
            # Get number of traces ntr & number of samples ns
            # Get time sample interval dt and define time axis t
            dt = 1e-6 *(f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL])
            t = np.arange(0.,dt*(self.ns+1),dt)
            t = t[::downS]
            self.ns =self.ns /downS
            # Get offset, CDP and SourceX for each trace
            offset_shot = np.zeros(self.ntr)
            CDP_shot = np.zeros(self.ntr,dtype=np.int64)
            SourceX_shot = np.zeros(self.ntr)
            self.receivX = np.zeros(self.ntr)
            for i in range (self.ntr):
                offset_shot[i] = f.header[i][segyio.TraceField.offset]
                offset_shot[i] = offset_shot[i]
                CDP_shot[i] = (f.header[i][segyio.TraceField.CDP])
                SourceX_shot[i] = f.header[i][segyio.TraceField.SourceX]
                SourceX_shot[i] = SourceX_shot[i]
                self.receivX[i] = f.header[i][segyio.TraceField.GroupX]
                self.receivX[i] = self.receivX[i]
        # Find unique source positions and number of shots
        SourceX = np.unique(SourceX_shot)
        nshot = SourceX.size
        self.SourceX=SourceX_shot
        # Find unique CDP positions and number of cdp gathers
        self.CDP = np.unique(CDP_shot)
        self.ncdp = self.CDP.size
        # Print ntr, ns, nshot, ncdp        
        print(str(identifier),': ntr = ', self.ntr, '; ', 'ns = ', self.ns, '; ', 'nshot = ', nshot, '; ', 'ncdp = ', self.ncdp)
        return CDP_shot, offset_shot,t, SourceX_shot
        
class wavefrontAttbt:
    def __init__(self,Rn, Rnip, alpha, CDP):
        self.Rn=Rn
        self.Rnip=Rnip
        self.ratio=0
        self.alpha=alpha
        self.CDP=CDP
        self.PFZ=0
        self.coherence=0
                
    def load_attributes(self, path2data, file, Attb,sigma, downS):
        rawdatafile=path2data+ file
        # Import seismic shot gathers & headers
        with segyio.su.open(rawdatafile,endian='little', ignore_geometry=True) as f:
            # Get basic attributes
            Attb= f.trace.raw[:]  # Get all data into memory (could cause on big files)
            ntr, ns = np.shape(Attb)
            Attb = ndimage.median_filter(Attb,sigma)
            Attb = Attb[::,::downS]
            # Get number of traces ntr & number of samples ns
            ns = ns /downS
            # Get offset, CDP and SourceX for each trace
            CDP_shot = np.zeros(ntr,dtype=np.int64())
            SourceX_shot = np.zeros(ntr)
            self.receivX = np.zeros(ntr)
            for i in range (ntr):
                CDP_shot[i] = (f.header[i][segyio.TraceField.CDP])
                SourceX_shot[i] = f.header[i][segyio.TraceField.SourceX]
                self.receivX[i] = f.header[i][segyio.TraceField.GroupX]
            cdpsort = np.argsort(CDP_shot)    
            Attb = Attb[cdpsort,:]
            CDP_shot = CDP_shot[cdpsort]
        # Find unique source positions and number of shots
        self.SourceX=SourceX_shot
        # Find unique CDP positions and number of cdp gathers
        self.CDP = np.unique(CDP_shot)
        return Attb
    
    def  Rad_ratio(self):
        shape=list(self.Rn.shape)
        epsilon = 0.0001
        fRn=self.Rn.flatten()
        fRnip=self.Rnip.flatten()
        c=np.exp(-(np.abs(fRn-fRnip)/np.abs(fRn+fRnip+epsilon)))
        self.ratio=np.reshape(c, shape)
    


class cdp_gathers:
    def __init__(self,cdp_traces, offset_cdp, cdpnum , source_cdp, receiv_cdp, Rn, Rnip, alpha,coherence):
        self.cdp_traces=cdp_traces
        self.cdp_traces_diff=0
        self.offset_cdp=offset_cdp
        self.cdpnum= cdpnum
        self.source_cdp= source_cdp
        self.receiv_cdp=receiv_cdp
        self.Rn_cdp=Rn
        self.Rnip_cdp=Rnip
        self.ratio_cdp=0
        self.alpha_cdp=alpha
        self.coherence=coherence
        self.vnmo2_cdp=0
        self.PFZ_cdp=0
        
    def apply_nmo_corr(self, t, max_offset, tolerance, Schalter, C_min, v0, a):
        # estimate number of cdp traces an time samples
        ntr_cdp = self.cdp_traces.shape[0]
        ns = t.size
        # sample interval
        dt = t[1] - t[0]
        # Allocate memory for NMO-corrected CDP gather
        cdp_traces_nmo = np.zeros((ntr_cdp,ns))
        self.cdp_traces_diff = np.zeros((ntr_cdp,ns))
        self.vnmo2_cdp= np.zeros(ns+1)
        # Apply NMO correction to CDP gather
        for j in range(1,ns-1):
            t0 = t[j]
            #compute vnmo**2 based on CRS Parameters
            self.vnmo2_cdp[j]= (2*v0*self.Rnip_cdp[j])/(t0*np.cos(np.deg2rad(self.alpha_cdp[j]))**2)
            for i in range(ntr_cdp):
                # compute time shift [s] no vnmo**2, bc variable is defined as the square of vnmo
                if self.vnmo2_cdp[j] != 0:
                    tshift = np.sqrt(t0**2 + (self.offset_cdp[i]**2/self.vnmo2_cdp[j]))
                if self.vnmo2_cdp[j] == 0:
                    tshift = np.nan
                # discrete time shift [time samples]
                if not math.isinf(tshift) and not math.isnan(tshift):
                    ntshift = (int)(tshift/dt)
                    if(ntshift < ns-1 and self.offset_cdp[i]< max_offset):
                        cdp_traces_nmo[i,j] = self.cdp_traces[i,ntshift]
            if Schalter == 1 and self.ratio_cdp[j] <= 1 + float(tolerance) and self.ratio_cdp[j] >= 1 - float(tolerance): 
                ##Damp diffractions
                cdp_traces_nmo[:,j]= cdp_traces_nmo[:,j]*(abs(np.exp(-a**2 * abs((self.ratio_cdp[j]-1)))-1)/10)
                if  abs(self.PFZ_cdp[j]) > 1.5:
                    cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*(abs(np.exp(-a**2*abs(self.PFZ_cdp[j])))/10)
                if abs(self.alpha_cdp[j]) > 20 and math.isnan(self.alpha_cdp[j]):
                    cdp_traces_nmo[:,j]= cdp_traces_nmo[:,j]*(abs(np.exp(-a**2 * abs((self.ratio_cdp[j]))-1)))
                    if abs(self.coherence[j]) < C_min and not math.isnan(self.PFZ_cdp[j]):
                        cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*(abs(np.exp(-a**2*abs(self.PFZ_cdp[j]-1))-1))
                    if abs(self.coherence[j]) < C_min and math.isnan(self.PFZ_cdp[j]):
                        cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*(abs(np.exp(-a**2*self.coherence[j])-1))
            if Schalter == 2 and self.ratio_cdp[j] <= 1 - float(tolerance):
                ##Damp reflections
                cdp_traces_nmo[:,j]= cdp_traces_nmo[:,j]*(abs(np.exp(-a**2 * abs((self.ratio_cdp[j]-1))))/10)
                if  abs(self.PFZ_cdp[j]) < 2.1:
                    ###je kleiner PFZ desto mehr Dämpfung
                    cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*abs(np.exp(-a**2*abs(self.PFZ_cdp[j]))/10)
                if abs(self.alpha_cdp[j]) > 10 and math.isnan(self.alpha_cdp[j]):
                    ### Dämpfung nimmt bis r=1 ab und danach wieder zu
                    cdp_traces_nmo[:,j]= cdp_traces_nmo[:,j]*(abs(np.exp(-a**2 * abs((self.ratio_cdp[j]-1))))/4)
                if abs(self.coherence[j]) < C_min and not math.isnan(self.PFZ_cdp[j]):
                    #je größer PFZ desto größer Dämpfung
                    cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*(abs(np.exp(-a**2*abs(self.PFZ_cdp[j]))))
                if abs(self.coherence[j]) < C_min and math.isnan(self.PFZ_cdp[j]):
                    cdp_traces_nmo[:,j] = cdp_traces_nmo[:,j]*(abs(np.exp(-a**2*self.coherence[j])-1))
            
            
        return cdp_traces_nmo
    
    def undo_nmo_corr(self, t,max_offset, cdp_traces_nmo, Schalter, a, tolerance):
        ntr_cdp =  cdp_traces_nmo.shape[0]
        ns = t.size
        ## storage place
        # sample interval
        dt = t[1] - t[0]
        for j in range(1,ns-1):
            t0 = t[j]
            for i in range(ntr_cdp):
                # discrete time shift [time samples]
                if self.vnmo2_cdp[j] != 0:
                    tshift = np.sqrt(t0**2 + (self.offset_cdp[i]**2/self.vnmo2_cdp[j]))
                if self.vnmo2_cdp[j] == 0:
                    tshift = np.nan
                if not math.isinf(tshift) and not math.isnan(tshift):
                    ntshift = (int)(tshift/dt)
                    if(ntshift < ns-1 and self.offset_cdp[i] < max_offset):
                        self.cdp_traces_diff[i,ntshift] = cdp_traces_nmo[i,j]
            if Schalter == 1 and self.ratio_cdp[j] <= 1 + float(tolerance) and self.ratio_cdp[j] >= 1 - float(tolerance): 
                ##enhance diffractions
                  cdp_traces_nmo[:,j]= cdp_traces_nmo[:,j]*(abs(np.exp(0.5**2 * abs((j*dt)))))
                
        
        
    def extract_cdp_gather(self,icdp,CDP, CDP_shot, shot, offset_shot, SourceX_shot, receivX):
        # Extract CDP gather of cdp number icdp
        self.cdpnum = CDP[icdp]
        # Estimate no. of traces in CDP gather
        ntr_cdp = np.size(np.where(CDP_shot == self.cdpnum))
        # Allocate memory for CDP gather traces & offsets
        cdp_traces = np.zeros((ntr_cdp,shot.shape[1]))
        offset_cdp = np.zeros(ntr_cdp)
        source_cdp = []
        receiv_cdp = []
        j=0
        for i in range(shot.shape[0]):
            if(CDP_shot[i]==self.cdpnum):
                cdp_traces[j,:] = shot[i,:]
                offset_cdp[j] = offset_shot[i]
                source_cdp.append(SourceX_shot[i])
                receiv_cdp.append(receivX[i])
                j = j + 1
        # Sort data by offset
        self.cdp_traces = cdp_traces[:]
        self.offset_cdp = offset_cdp[:]
        self.source_cdp = source_cdp[:]
        self.receiv_cdp = receiv_cdp[:]
    
    def extract_wfA(self,icdp, CDP, Attribute, Schalter, ncdp):
        # Allocate memory for CDP gather traces & offsets
        cdp_attribute = np.zeros(len(Attribute.T))
        ## assign to object attributes:
        for i in range(1,name.ncdp-1):
            if(CDP[i]==self.cdpnum):
                cdp_attribute = Attribute[i,:]
        if Schalter == 1:
            self.Rn_cdp = cdp_attribute[:]
        if Schalter == 2:
            self.Rnip_cdp = cdp_attribute[:]
        if Schalter == 3:
            self.alpha_cdp = cdp_attribute[:]
        if Schalter == 4:
            self.ratio_cdp = cdp_attribute[:]
        if Schalter == 5:
            self.PFZ_cdp = cdp_attribute[:]
        if Schalter == 6:
            self.coherence = cdp_attribute[:]
    
    def replaceX_w_Nan(self, Attribute, X):
        Zaehler=0
        for i in Attribute:
            if float(i) == X:
                Attribute[Zaehler]=np.nan
            Zaehler+=1

        
class shot_gathers:
    def __init__(self, Snum, ns, ntr_inshot):
        self.data=0
        self.ntr_inshot=ntr_inshot
        self.ns=ns
        self.Snum=Snum
        self.receiv_Sx=0
    
    def extract_shot_gather(self,corr_traces, SourceX, receivX, downS):
        # Allocate memory for CDP gather traces & offsets
        traces = np.zeros((self.ntr_inshot, self.ns))
        receiv_Sx = np.array(np.zeros(self.ntr_inshot))
        j=0
        for i in range(len(SourceX)):
            if(SourceX[i]== self.Snum):
                traces[j,:] = corr_traces[i,:]
                receiv_Sx[j] = receivX[i]
                j = j + 1
        # Sort data by offset
        mark_idx = np.argsort(receiv_Sx)
        self.data = traces[mark_idx,:]
        self.data = resample_fct(self.data,1,downS)
        self.receiv_Sx = receiv_Sx[mark_idx]

        


# In[3]:


def plot_models(model, title, cmap, vmin, vmax):
    plt.figure(figsize=(10,5), dpi=100); plt.imshow(model.T, cmap=cmap, vmin=vmin, vmax=vmax);
    plt.colorbar(); plt.title(title); plt.axis('auto'); plt.pause(0.001)
def plot_shot(shot, title, pclip=1.0):  
    vmax = pclip * np.max(np.abs(shot)); vmin = - vmax; plt.figure(figsize=(10,5), dpi=100); plt.imshow(shot.T, cmap='Greys', vmin=vmin, vmax=vmax); 
    plt.title(title); plt.axis('auto'); plt.pause(0.001)
def resample_fct(array, ndt1, ndt2):
    out=ndimage.zoom(array, (ndt1,ndt2))
    return out
def moving_average(x, wlen):
    mw = ndimage.convolve(x, np.ones((1,wlen)), mode='nearest')/wlen
    mw_x = x- mw
    return mw_x

    

# In[4]:


####---- create and load datasets into various objects####----
basename='diff4_glob'
path2data='model_4/DE/'
sigma=1
###resample to fit seismic data
downS=1
#---------------Wavefront attributes-----------------------------#
wfA=wavefrontAttbt(0,0,0,0)
alphafile= basename + '.angle.su'
wfA.alpha =wfA.load_attributes(path2data,alphafile, 'alpha', sigma, downS)
#wfA.alpha=resample_fct(wfA.alpha,ndt)

Rnfile= basename+'.RN.su'
wfA.Rn =wfA.load_attributes(path2data,Rnfile, 'Rn', sigma, downS)
wfA.Rn= np.array(wfA.Rn, dtype=np.float64)
#wfA.Rn=resample_fct(wfA.Rn,ndt)
#wfA.Rn=median_filter(wfA.Rn, sigma)

Rnipfile= basename+'.Rnip.su'
wfA.Rnip =wfA.load_attributes(path2data,Rnipfile, 'Rnip', sigma, downS)
wfA.Rnip= np.array(wfA.Rnip, dtype=np.float64)
#wfA.Rnip=resample_fct(wfA.Rnip,ndt)
#wfA.Rnip=median_filter(wfA.Rnip,sigma)
PFZfile= basename+'.RelFzone.su'
wfA.PFZ =wfA.load_attributes(path2data,PFZfile, 'PFZ', sigma, downS)
#wfA.PFZ=resample_fct(wfA.PFZ,ndt)
#wfA.PFZ=median_filter(wfA.PFZ, sigma)
#wfA.PFZ= np.array(wfA.PFZ, dtype=np.float64)
coherencefile= basename+'.coher.su'
wfA.coherence =wfA.load_attributes(path2data,coherencefile, 'coherence', sigma, downS)
wfA.coherence= np.array(wfA.coherence, dtype=np.float64)
#wfA.coherence=resample_fct(wfA.coherence,ndt)
#wfA.coherence=median_filter(wfA.coherence, sigma)
wfA.Rad_ratio()
#----------------seismic tt data--------------------------------#
# Define path and filename to create field data object
rawdata='CRS_diffmodel_4_dirmute.su'
dir_mute_shot=dataset(0,0,0,0,0,0,0,0,0,0)
name=dir_mute_shot
CDP_shot, offset_shot,t, SourceX_shot =dir_mute_shot.load_section('model_4/', dir_mute_shot, rawdata,'muted DW shot', downS)


# In[5]:


plot_models(wfA.alpha,'alpha (DE)', 'magma',-40,40)
plot_models(wfA.Rn,'Rn (DE)', 'magma',0,350)
plot_models(wfA.ratio,'Rn/Rnip Ratio (DE)', 'magma',0,1)
plot_models(wfA.PFZ,'Projected Fresnel Zone (DE)', 'magma',0,4)
plot_models(wfA.coherence,'Coherence (DE)', 'magma',0,1)


# In[6]:


zo_section_1 = np.zeros((dir_mute_shot.ncdp,dir_mute_shot.data.shape[1]))
zo_section_2 = np.zeros((dir_mute_shot.ncdp,dir_mute_shot.data.shape[1]))
zo_section_3 = np.zeros((dir_mute_shot.ncdp,dir_mute_shot.data.shape[1]))
max_offset = 10000000.
wlen=2
C_min= 0.2
Dampfungsfaktor=1
v0=2000
ratio_tolerance=0.6
Schalter=1
enhanceDiff=1
cdpmin = 1
cdpmax = dir_mute_shot.ncdp
object_names=[]
for icdp in range(cdpmin, cdpmax):
    cdp_name= 'cdp_' + str(icdp)
    object_names.append(cdp_name) 
objects = {}
icdp= cdpmin
total=0
all_SourceX=[]
all_receivX=[]
all_traces_diff=[]
all_traces_ncorr=[]
Erster_lauf = True
for object_name in object_names:
    objects[object_name] = cdp_gathers(0,0,0,0,0,0,0,0,0)
    objects[object_name].extract_cdp_gather(icdp, dir_mute_shot.CDP, CDP_shot, dir_mute_shot.data, offset_shot, dir_mute_shot.SourceX, dir_mute_shot.receivX)
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.Rn, 1,len(dir_mute_shot.CDP))
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.Rnip, 2,len(dir_mute_shot.CDP))
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.alpha, 3,len(dir_mute_shot.CDP))
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.ratio, 4,len(dir_mute_shot.CDP))
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.PFZ,5,len(dir_mute_shot.CDP))
    objects[object_name].extract_wfA(icdp, dir_mute_shot.CDP, wfA.coherence,6,len(dir_mute_shot.CDP))
    
    cdp_traces_nmo = objects[object_name].apply_nmo_corr(t, max_offset, ratio_tolerance, Schalter, C_min, v0, Dampfungsfaktor)
    cdp_traces_nmo_full = objects[object_name].apply_nmo_corr(t, max_offset, ratio_tolerance,0, C_min, v0, Dampfungsfaktor)
    #cdp_traces_nmo_dR = objects[object_name].apply_nmo_corr(t, max_offset, ratio_tolerance, 2, C_min, v0, Dampfungsfaktor)
    #cdp_traces_nmo_dR = moving_average(cdp_traces_nmo_dR, wlen)
    cdp_traces_nmo_diff = np.zeros((objects[object_name].offset_cdp.size,t.size))
    cdp_traces_nmo_diff = cdp_traces_nmo_full - cdp_traces_nmo
    cdp_traces_nmo_diff = moving_average(cdp_traces_nmo_diff, wlen)
    ###hier werden die spuren de-nmokorrigiert  bei denen Reflektionen gedämpft wurden
    objects[object_name].undo_nmo_corr(t,max_offset, cdp_traces_nmo_diff, enhanceDiff, Dampfungsfaktor, ratio_tolerance)
    if Erster_lauf:
        ## Erstellen der Arrays
        all_traces_ncorr= objects[object_name].cdp_traces[:]
        all_traces_diff= objects[object_name].cdp_traces_diff[:]
        all_SourceX = objects[object_name].source_cdp[:]
        all_receivX = objects[object_name].receiv_cdp[:]
    else:
        ### cat der Arrays über alle cdpgathers
        all_traces_ncorr= np.concatenate((all_traces_ncorr,objects[object_name].cdp_traces), axis=0)
        all_traces_diff = np.concatenate((all_traces_diff,objects[object_name].cdp_traces_diff), axis=0)
        all_SourceX = np.concatenate((all_SourceX,objects[object_name].source_cdp))
        all_receivX = np.concatenate((all_receivX,objects[object_name].receiv_cdp))
    
    stack_1= np.sum(cdp_traces_nmo_diff,0)
    #stack_2=np.sum(cdp_traces_nmo_dR,0)
    #stack_3=np.sum(cdp_traces_nmo,0)
    zo_section_1[icdp,1:dir_mute_shot.data.shape[1]] = stack_1[1:dir_mute_shot.data.shape[1]]
    #zo_section_2[icdp,1:dir_mute_shot.data.shape[1]] = stack_2[1:dir_mute_shot.data.shape[1]]
    #zo_section_3[icdp,1:dir_mute_shot.data.shape[1]] = stack_3[1:dir_mute_shot.data.shape[1]]
    
    icdp+=1
    Erster_lauf= False
    


# In[7]:



#zo_section_1= moving_average(zo_section_1, 10)
plot_shot(zo_section_1, 'CMP zero-offset section,Diffraction Damping (DE)', pclip=0.2)

# In[ ]: Here the processed data is split back into shotfiles and exported as *.sgy


###find unique source positions
nSx = np.unique(dir_mute_shot.SourceX)
mark_idx = nSx.argsort()
ntr_Sx=len(nSx)
nSx = nSx[mark_idx]
shot_names=[]
## set export_processed_dat ==1 to export manipulated data, set ==0 to export raw data
export_processed_data=1
j=1
for iSx in nSx:
    number = int(iSx)
    Sx_name= 'DENISE_MARMOUSI_y.shot' + str(j)
    shot_names.append(Sx_name) 
    j+=1
shots = {}
i= 0
for shot_name in shot_names:
    #print('object name: ' + str(shot_name))
    file = shot_name + '.sgy'
    Snum=nSx[i]
    ntr_inshot = np.size(np.where(all_SourceX == Snum))
    shots[shot_name] = shot_gathers(Snum,dir_mute_shot.data.shape[1], ntr_inshot)
    ## aus den arrays mit den de-nmokorrigierten traces aus allen cdp werden jeweils alle Spuren eines Schusses in gather sortiert
    if export_processed_data == 1:
        shots[shot_name].extract_shot_gather(all_traces_diff[::,1::], all_SourceX, all_receivX, downS)
    if export_processed_data == 0:
        shots[shot_name].extract_shot_gather(all_traces_ncorr[::,::], all_SourceX, all_receivX, downS)
    if export_processed_data != 0 and export_processed_data != 1:
        print('switch export_processed_data set incorrectly must either be 1 or 0')
        sys.exit()
    ### file wird unstrukturiert exportiert
    spec = segyio.spec()
    spec.ns = shots[shot_name].data.shape[1]
    spec.ilines  = None
    spec.dt=dir_mute_shot.data.shape[1]*downS
    spec.tracecount = shots[shot_name].data.shape[0]
    spec.samples = list(range(shots[shot_name].data.shape[1]))
    spec.format  = 5
    spec.enidan = 'little'
    #print(shots[shot_name].data.shape)
    with segyio.create(file, spec) as f:
        f.trace = shots[shot_name].data
    i+=1
    
with segyio.open('DENISE_MARMOUSI_y.shot50.sgy',ignore_geometry=True) as srcf:
     exp_shot = srcf.trace.raw[:]
     
plot_shot(exp_shot, ' created shotfile', pclip=0.2)




# In[ ]:




