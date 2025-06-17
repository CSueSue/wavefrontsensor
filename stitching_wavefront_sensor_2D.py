# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:05:00 2025

@author: crvan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline,SmoothBivariateSpline
import traceback
import sys
# inputs
#-  9mm x 9mm/ 0.5pitch  (19x19x2) array of tilt measurements. (how to deal with NaNs?)
#- XY position of stage.
#- 3D array of tilt values averaged. (NxNx2)
#- 2D array of number of values of the average. (NxN)
#- 2D array of x-positions.(NxN)
#- 2D array of y-positions.(NxN)

# open issues: How to integrate using both Rx and Ry to get z-flatness. 

pitch = 0.5e-3 
size_im = [18, 18]
centerPositionCamera = [9*pitch, 9*pitch]

def initArrays(Nx,Ny, dx,dy, x0, y0):
    global x_pos,y_pos, N_av, tilts_av
    N_ax = int(np.floor((Nx-1)*dx/pitch))+size_im[1]
    N_ay = int(np.floor((Ny-1)*dy/pitch))+size_im[0]
    y_pos, x_pos = np.mgrid[0:N_ay,0:N_ax]*pitch
    x_pos += x0- centerPositionCamera[0]
    y_pos += y0- centerPositionCamera[1]
    N_av = np.ones((N_ay,N_ax),float)*np.nan
    tilts_av = np.ones((N_ay,N_ax,2),float)*np.nan


def stitch(tiltMeasurement, xyStage):
    global  N_av, tilts_av
    try:
        #create position arrays
        y0,x0 = np.mgrid[0:tiltMeasurement.shape[0], 0:tiltMeasurement.shape[1]]*pitch
        y0+= xyStage[1]-centerPositionCamera[1]
        x0+= xyStage[0]-centerPositionCamera[0]
    
        # find matching indices in y_pos, x_pos.
        sIy =len(np.where(np.round(y_pos[:,0],4)<np.round(y0[:,0].min(),4))[0]) 
        eIy = len(np.where(np.round(y0[:,0].max(),4)>=np.round(y_pos[:,0],4))[0])
        sIx = len(np.where(np.round(x_pos[0,:],4)<np.round(x0[0,:].min(),4))[0])
        eIx = len(np.where(np.round(x0[0,:].max(),4)>=np.round(x_pos[0,:],4))[0])
    
        
        

           
        # if nan values at edge use non regular grid interpolation.
        valid = np.logical_not(np.isnan(tiltMeasurement[:,:,0].flatten()))

        if valid.sum() ==valid.shape[0]:
            # no nans.
            fint_Rx = RectBivariateSpline(y0[:,0],x0[0,:],tiltMeasurement[:,:,0], kx=3, ky=3)
            fint_Ry = RectBivariateSpline(y0[:,0],x0[0,:],tiltMeasurement[:,:,1], kx=3, ky=3)
        else:
            # cannot put kx,ky higher than 1 for this interpolation.
            fint_Rx = SmoothBivariateSpline(y0.flatten()[valid],x0.flatten()[valid],tiltMeasurement[:,:,0].flatten()[valid], kx=1, ky=1)
            fint_Ry = SmoothBivariateSpline(y0.flatten()[valid],x0.flatten()[valid],tiltMeasurement[:,:,1].flatten()[valid], kx=1, ky=1)
        
        
        

        
        # interpolate         
        Rx_int = fint_Rx(y_pos[sIy:eIy,0],x_pos[0,sIx:eIx])
        Ry_int = fint_Ry(y_pos[sIy:eIy,0],x_pos[0,sIx:eIx])
        
        # find overlapping sections where N_av >=1.
        bool1 = N_av[sIy:eIy,sIx:eIx].flatten()>=1
        if bool1.sum() == 0:
            Rx_av = Rx_int.flatten().mean() 
            Ry_av = Ry_int.flatten().mean()
        else:
            # calculate average of difference between overlapping points.
            Rx_av = Rx_int.flatten()[bool1].mean() - tilts_av[sIy:eIy,sIx:eIx,0].flatten()[bool1].mean()
            Ry_av = Ry_int.flatten()[bool1].mean() - tilts_av[sIy:eIy,sIx:eIx,1].flatten()[bool1].mean()        
        # subtract average difference
        Rx_int-=Rx_av
        Ry_int-=Ry_av
        
        
        # mask_nans. find all points <0.5*pitch from nans and disregard interpolated point.
        mask = np.ones(Rx_int.shape, int)
        xnan = x0.flatten()[np.logical_not(valid)]
        ynan = y0.flatten()[np.logical_not(valid)]
        if len(xnan)>0:
            for i,y in enumerate(y_pos[sIy:sIy+mask.shape[0],0]):
                for j,x in enumerate(x_pos[0,sIx:sIx+mask.shape[1]]):
                    if np.min((x-xnan)**2+(y-ynan)**2)<(0.5*pitch)**2:
                        mask[i,j]=0
        
        Npoints = np.nansum((N_av[sIy:sIy+mask.shape[0],sIx:sIx+mask.shape[1]],mask),axis=0)
        # to prevent zero division set 0 to Nan.
        Npoints[Npoints==0] = np.nan
        
        tilts_av[sIy:eIy,sIx:eIx,0] = np.nansum([tilts_av[sIy:eIy,sIx:eIx,0]*N_av[sIy:eIy,sIx:eIx], mask*Rx_int],axis=0)/Npoints
        tilts_av[sIy:eIy,sIx:eIx,1] = np.nansum([tilts_av[sIy:eIy,sIx:eIx,1]*N_av[sIy:eIy,sIx:eIx], mask*Ry_int],axis=0)/Npoints
        
        N_av[sIy:sIy+mask.shape[0],sIx:sIx+mask.shape[1]] =Npoints
            
            
    except:
        traceback.print_exc(file = sys.stdout)
        
    
    return 
        
    
    
    











# approach flat surface. 
# sample at dy. 
# add noise. 
# use interpolation to find matching points.
# Subtract average of new points.


def GaussTilts(x,y,mux,muy, sigma,A):
    Rx = -(y-muy)/sigma**2*A*np.exp(-0.5*((x-mux)**2+(y-muy)**2)/sigma**2)
    Ry = (x-mux)/sigma**2*A*np.exp(-0.5*((x-mux)**2+(y-muy)**2)/sigma**2)    
    
    return Rx,Ry



def simulate_300mmSquare():

    dy = 0.6e-3
    dx = 7e-3
    sx = 9e-3
    sy = 9e-3
    sigmaN = 1e-7 # rad
    # gauss
    sigmaG = 6e-3/2.35
    muG = [0, 0]
    A = 200e-9
    # linear function
    alpha = 50e-6/300e-3 # rad/m
    
    y0,x0 = np.mgrid[0:sy:pitch, 0:sx:pitch]
    
    x_s0 = -150e-3
    y_s0 = -150e-3
    
    Nx = int(300e-3/dx)+1
    Ny = int(300e-3/dy)+1
    
    initArrays(Nx, Ny, dx, dy, x_s0, y_s0)
    
    
    
    zn1 = np.zeros((y0.shape[0], x0.shape[1], 2),float)
    stage_x = np.arange(Nx)*dx + x_s0
    stage_y = np.arange(Ny)*dy + y_s0
    # start stitching from the center where there is overlap.
    stage_x = np.hstack((stage_x[Nx//2:],stage_x[:Nx//2][::-1]))
    stage_y = np.hstack((stage_y[Ny//2:],stage_y[:Ny//2][::-1]))
    
    zn1 = np.zeros((y0.shape[0], x0.shape[1], 2),float)
    for j,x_s in enumerate(stage_x):
        x1 = x0+ x_s-centerPositionCamera[0]
        
        for i,y_s in enumerate(stage_y):
            y1=y0+y_s-centerPositionCamera[1]
  
            
            #bool1 = (x1**2+y1**2)>150e-3**2
            
            Rx,Ry = GaussTilts(x1,y1, muG[0],muG[1],sigmaG,A)
            
            zn1[:,:,0] = np.random.randn(y0.shape[0], y0.shape[1])*sigmaN + Rx + alpha*y1 + alpha*x1
            zn1[:,:,1] = np.random.randn(y0.shape[0], y0.shape[1])*sigmaN + Ry + alpha*y1 + alpha*x1
            #zn1[bool1] = np.nan
        
        
            stitch(zn1, [x_s,y_s])
    
    
    
    
    plt.figure(1)
    plt.title("stitched Rx")
    plt.contourf(x_pos, y_pos, tilts_av[:,:,0])
    plt.figure(2)
    plt.title("true Rx")
    
    
    
    Rx,Ry = GaussTilts(x_pos, y_pos, muG[0], muG[1], sigmaG, A)
    plt.contourf(x_pos, y_pos,Rx + alpha*y_pos+alpha*x_pos)
    error_Rx = tilts_av[:,:,0]-Rx - alpha*y_pos-alpha*x_pos
    error_Ry = tilts_av[:,:,1]-Ry - alpha*y_pos-alpha*x_pos
    
    
    fig3 = plt.figure(3)
    ax = plt.gca()
    plt.title("error Rx")
    Ne = 0
    CS = plt.contourf(x_pos[Ne:x_pos.shape[0]-Ne,Ne:x_pos.shape[1]-Ne], y_pos[Ne:y_pos.shape[0]-Ne,Ne:y_pos.shape[1]-Ne],\
                      error_Rx[Ne:error_Rx.shape[0]-Ne,Ne:error_Rx.shape[1]-Ne]-np.nanmean(error_Rx[Ne:error_Rx.shape[0]-Ne,Ne:error_Rx.shape[1]-Ne]),cmap = plt.cm.jet)
    
    cbar = fig3.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    
    fig4 = plt.figure(4)
    ax = plt.gca()
    plt.title("error Ry")
    CS = plt.contourf(x_pos[Ne:x_pos.shape[0]-Ne,Ne:x_pos.shape[1]-Ne], y_pos[Ne:y_pos.shape[0]-Ne,Ne:y_pos.shape[1]-Ne],\
                      error_Ry[Ne:error_Ry.shape[0]-Ne,Ne:error_Ry.shape[1]-Ne]-np.nanmean(error_Ry[Ne:error_Ry.shape[0]-Ne,Ne:error_Ry.shape[1]-Ne]),cmap = plt.cm.jet)

    
    cbar = fig4.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    
    plt.show()


def simulate_300mm_wafer():
    dy = 0.6e-3
    dx = 7e-3
    sx = 9e-3
    sy = 9e-3
    sigmaN = 1e-7 # rad
    # gauss
    sigmaG = 6e-3/2.35
    muG = [0, 0]
    A = 200e-9
    # linear function
    alpha = 50e-6/300e-3 # rad/m
    
    y0,x0 = np.mgrid[0:sy:pitch, 0:sx:pitch]
    
    x_s0 = -150e-3
    y_s0 = -150e-3
    
    Nx = int(300e-3/dx)+1
    Ny = int(300e-3/dy)+1
    
    initArrays(Nx, Ny, dx, dy, x_s0, y_s0)
    
    
    stage_x = np.arange(Nx)*dx + x_s0
    stage_y = np.arange(Ny)*dy + y_s0
    # start stitching from the center where there is overlap.
    stage_x = np.hstack((stage_x[Nx//2:],stage_x[:Nx//2][::-1]))
    stage_y = np.hstack((stage_y[Ny//2:],stage_y[:Ny//2][::-1]))
    
    zn1 = np.zeros((y0.shape[0], x0.shape[1], 2),float)
    for j,x_s in enumerate(stage_x):
        x1 = x0+ x_s-centerPositionCamera[0]
        
        for i,y_s in enumerate(stage_y):
            y1=y0+y_s-centerPositionCamera[1]
            
            bool1 = (x1**2+y1**2)>150e-3**2
            
            Rx,Ry = GaussTilts(x1,y1, muG[0],muG[1],sigmaG,A)
            
            zn1[:,:,0] = np.random.randn(y0.shape[0], y0.shape[1])*sigmaN + Rx + alpha*y1 + alpha*x1
            zn1[:,:,1] = np.random.randn(y0.shape[0], y0.shape[1])*sigmaN + Ry + alpha*y1 + alpha*x1
            zn1[bool1] = np.nan
        
        
            stitch(zn1, [x_s,y_s])
    
    
    
    
    plt.figure(1)
    plt.title("stitched Rx")
    plt.contourf(x_pos, y_pos, tilts_av[:,:,0])
    plt.figure(2)
    plt.title("true Rx")
    
    
    
    Rx,Ry = GaussTilts(x_pos, y_pos, muG[0], muG[1], sigmaG, A)
    plt.contourf(x_pos, y_pos,Rx + alpha*y_pos+alpha*x_pos)
    error_Rx = tilts_av[:,:,0]-Rx - alpha*y_pos-alpha*x_pos
    error_Ry = tilts_av[:,:,1]-Ry - alpha*y_pos-alpha*x_pos
    
    
    fig3 = plt.figure(3)
    ax = plt.gca()
    plt.title("error Rx")
    Ne = 15
    CS = plt.contourf(x_pos[Ne:-Ne,Ne:-Ne], y_pos[Ne:-Ne,Ne:-Ne],\
                      error_Rx[Ne:-Ne,Ne:-Ne]-np.nanmean(error_Rx[Ne:-Ne,Ne:-Ne]),cmap = plt.cm.jet)
    
    cbar = fig3.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    
    fig4 = plt.figure(4)
    ax = plt.gca()
    plt.title("error Ry")
    CS = plt.contourf(x_pos[Ne:-Ne,Ne:-Ne], y_pos[Ne:-Ne,Ne:-Ne],\
                      error_Ry[Ne:-Ne,Ne:-Ne]-np.nanmean(error_Ry[Ne:-Ne,Ne:-Ne]), cmap = plt.cm.jet)
    
    cbar = fig4.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    
    plt.show()


if __name__ == "__main__":
    
    simulate_300mmSquare()
    plt.close('all')
    simulate_300mm_wafer()
