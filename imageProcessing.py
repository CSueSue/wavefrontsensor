""" fits 2D gaussians in the image and fits a grid"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.optimize import least_squares
from scipy.spatial import KDTree, distance
import time
pitch = 0.5e-3 
N_spots = [19, 19]
imsize = [2048, 2040]
spix = 5.5e-6
# diffraction limit of spot size
sigma_spot = 500e-9/(0.5*pitch/47e-3)*1.22/spix
A_spot = 200
FL = 47e-3
Rzimage = 0.0
def maketestimage(): 
    im = np.zeros(imsize, np.float32)
    Yg,Xg = np.mgrid[0:im.shape[0],0:im.shape[1]]
    
    yspots,xspots = np.mgrid[0:N_spots[1], 0:N_spots[0]]*1.0
    xspots = xspots.flatten()
    yspots = yspots.flatten()
    yspots-= N_spots[1]//2
    xspots-= N_spots[0]//2
    yspots*= pitch/spix
    xspots*= pitch/spix
    yspots += imsize[0]//2
    xspots += imsize[1]//2
    

    
    for i in range(xspots.shape[0]):
         
        im+= A_spot*np.exp(-0.5*((Xg-xspots[i])**2+(Yg-yspots[i])**2)/sigma_spot**2)
        
    im_pil = Image.fromarray(im.astype(np.uint8))
    im_pil.save("testimage.png")

def gauss(par, x,y):
    x0 = par[0]
    y0 = par[1]
    sigma = par[2]
    A = par[3]
    offset = par[4]
    return A*np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sigma**2)+offset

def jacobian(par,x,y,I):
    x0 = par[0]
    y0 = par[1]
    sigma = par[2]
    A = par[3]
    #offset = par[4]
    
    # jacobian is a 5 x N matrix
    jac= np.ones((x.shape[0],5), float)
    jac[:,0] = (x-x0)/sigma**2*A*np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sigma**2) 
    jac[:,1] = (y-y0)/sigma**2*A*np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sigma**2)
    jac[:,2] = 1.5*((x-x0)**2+(y-y0)**2)/sigma**3*A*np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sigma**2)
    jac[:,3] = np.exp(-0.5*((x-x0)**2+(y-y0)**2)/sigma**2)
    #jac[4,:] = 1
    return -jac

def errorGauss(par,x,y,I):
    return I-gauss(par,x,y)


def fitGaussian(gray):



    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 8, cv2.CV_32S)
    
    Rspot = 0.4*pitch/spix
    Yg,Xg = np.mgrid[0:gray.shape[0],0:gray.shape[1]]
    # yg = np.arange(gray.shape[0])
    # xg = np.arange(gray.shape[1])
    Yg = Yg.flatten()
    Xg = Xg.flatten()
    gray_fl = gray.flatten()
    
    fit_parms= []
    sse_list = []
    t0 = time.time()
    for i in range(1,num_labels):
        x0 = centroids[i,0]
        y0 = centroids[i,1]
        par_0 = [x0,y0,sigma_spot, A_spot, 0.0]
        
        bool1 = (Xg-x0)**2+(Yg-y0)**2<Rspot**2
        
        
        res = least_squares(errorGauss, par_0,args = (Xg[bool1],Yg[bool1],gray_fl[bool1]))#, jac=jacobian) #using jacobian makes it slower?
        fit_parms.append(res.x)
        sse_list.append(np.sum(errorGauss(par_0, Xg[bool1],Yg[bool1],gray_fl[bool1])**2))
        
    
        # Ix0 = np.where(xg>fit_parms[i-1][0]-Rspot)[0][0]
        # Iy0 = np.where(yg>fit_parms[i-1][1])[0][0]
        # N = int(2*Rspot)
    
        # gplot = gauss(fit_parms[i-1],xg[Ix0:Ix0+N],yg[Iy0])
        # plt.plot(xg[Ix0:Ix0+N],gplot)
        # plt.plot(xg[Ix0:Ix0+N], gray[Iy0,Ix0:Ix0+N])
        # plt.show()
    #    break
    print(time.time()-t0)

    return fit_parms, sse_list

def calculateTilts(parameters):
    # indexing
    parameters = np.array(parameters)
    
    
    # read nominal positions
    with open("nominalPositions.csv","r") as fp:
        pos_nom = np.loadtxt(fp,  delimiter = ',')
        
    # find the closest point and index to nomimal positions. 
    tree = KDTree(parameters[:,:2])
    dist, ii = tree.query(pos_nom)
    
    
    # calculate x,y vector for each point.
    XY = parameters[ii,:2]-pos_nom
    # put distances > 0.25*pitch to nan.
    XY[dist>0.25*pitch/spix,:] = np.nan
    
    
    # calculate tilts. Signs: Ry ~ x , Rx ~ -y. 
    tiltMeasurement = np.zeros((N_spots[0],N_spots[1],2))
    tiltMeasurement[:,:,0] = XY[:,0].reshape(N_spots)*spix/(2*FL)
    tiltMeasurement[:,:,1] = -XY[:,1].reshape(N_spots)*spix/(2*FL)
    return tiltMeasurement

def calibrateNominalPositions(positions):
    # calculate pitch
    dist = distance.cdist(positions,positions,'euclidean').flatten()
    pitch_median = np.median(dist[np.logical_and(dist>0.8*pitch/spix, dist<1.2*pitch/spix)])
    # calculate Rz.
    Rz = np.cov(positions[:,0],positions[:,1])/np.cov(positions[:,0])
    Rz = Rz[0,1]
    
    # rotate to zero.
    Xr = positions[:,0]*np.cos(-Rz) -positions[:,1]*np.sin(-Rz)
    Yr = positions[:,0]*np.sin(-Rz) +positions[:,1]*np.cos(-Rz)

    # divide by pitch to get indices.
    Ix = np.round(Xr/pitch_median).astype(int)
    Ix -= Ix.min()
    Iy = np.round(Yr/pitch_median).astype(int)
    Iy -=Iy.min()
    
    idx_sorted = np.zeros(Ix.shape, int)
    # sort in y-direction, then sort in x -direction.
    Iy_uniq = np.unique(Iy)
    j = 0
    for i,iy in enumerate(Iy_uniq):
        ii = np.where(Iy==iy)[0]
        idx_sorted[j:j+ii.shape[0]] = ii[np.argsort(Ix[ii])]
        j+= ii.shape[0]
    
    
    with open("nominalPositions.csv","w") as fp:
       np.savetxt(fp, positions[idx_sorted,:], delimiter = ',')

if __name__ == "__main__":
    
    im_pil = Image.open("testimage.png")
    #Convert the image to a NumPy array
    gray = np.array(im_pil)


    parameters, sse = fitGaussian(gray)
    
    parameters = np.array(parameters)
    
    
    tiltMeasurement = calculateTilts(parameters)
    

        
        
    #calibrateNominalPositions(parameters[:,:2])