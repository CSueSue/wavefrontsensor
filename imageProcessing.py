""" fits 2D gaussians in the image and fits a grid"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.optimize import least_squares
from scipy.spatial import KDTree, distance
from scipy.interpolate import RectBivariateSpline
import time
import stitching_wavefront_sensor_2D as stitch
import os  
imdir = r'./testimages/'
    

pitch = 0.5e-3 
N_spots = [18, 17] # y, x
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
    yspots-= N_spots[0]//2
    xspots-= N_spots[1]//2
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





def fitRectangle(gray, threshold = 75):

    edge_clearance= 0.5 # fraction of square edge that is not fitted at corners.
    Lcross = 10 
    Sfilter = 9
    subpixel = 0.1
    outlier = 5
    

    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 8, cv2.CV_32S)
    
    positions = centroids[1:,:]
    
    dist = distance.cdist(positions,positions,'euclidean').flatten()
    pitch_median = np.median(dist[np.logical_and(dist>0.8*pitch/spix, dist<1.2*pitch/spix)])
    
    # find closest point for each point and calculate angle.
    dist = distance.cdist(positions,positions,'euclidean')
    dist[dist==0] = 1e4 # remove zeros.
    Imin = np.argmin(dist, axis=0)
    angles = np.arctan((positions[Imin,1]-positions[:,1])/(positions[Imin,0]-positions[:,0]))
        
    angles = np.mod(angles, np.pi/2)
    phi = np.median(angles)
    
    width = np.median(stats[:,2])
    height = np.median(stats[:,3])
    Np = int(width*(1-edge_clearance))
    
    blurred_image = cv2.GaussianBlur(gray, (Sfilter, Sfilter), 0)
    
    ii = np.linspace(0.5*edge_clearance,1-0.5*edge_clearance,Np)
    stats_rect = np.zeros((centroids.shape[0]-1,6))# x,y,w,h,phi, sigma
    ipitch = int(pitch_median)
    
    # plt.figure(1, figsize = (21,21))
    # plt.imshow(gray, cmap = "gray")
    

    
    for i in range(centroids.shape[0]-1):
        # interpolation of sub image.
        Imin = max(int(centroids[i+1,1]-0.5*pitch_median),0)
        Imax = min(ipitch+Imin,gray.shape[0])
        Jmin = max(int(centroids[i+1,0]-0.5*pitch_median),0)
        Jmax = min(ipitch+Jmin,gray.shape[1])
        
        xI = np.arange(Imax-Imin)
        yI = np.arange(Jmax-Jmin)
        
        f_int = RectBivariateSpline(xI, yI ,blurred_image[Imin:Imax,Jmin:Jmax], kx = 3, ky = 3, s =0)
        corners = corner_points_square([centroids[i+1,0], centroids[i+1,1],phi,width])
        
        
        
        edges= np.zeros((4,Np,2))
        mu_edges = np.zeros((4,2))
        for j in range(4):
        
    
    
            Pstart = corners[j,:].copy()
            Pstart[0]-=Jmin
            Pstart[1]-=Imin
            
            vec_dir = corners[j+1,:]-corners[j,:]
            norm_dir = [vec_dir[1]*-1, vec_dir[0]]/np.linalg.norm(vec_dir)
            
            for k in range(Np):
    
                P0 = Pstart+ii[k]*vec_dir
                
                pointsx = P0[0] + np.arange(-Lcross,Lcross+subpixel,subpixel)*norm_dir[0]
                pointsy = P0[1] + np.arange(-Lcross,Lcross+subpixel,subpixel)*norm_dir[1]
                
                dr = ((pointsx[1]-pointsx[0])**2 + ((pointsy[1]-pointsy[0])**2))**0.5
                
                I =f_int(pointsx,pointsy, grid = False)
                
                If = np.argmax(np.diff(I)/dr)
                edges[j,k,:] = np.array([np.mean(pointsx[If:If+2]),np.mean(pointsy[If:If+2])])
        
                # plt.plot(edges[j,:,0]+Jmin, edges[j,:,1]+Imin,'r+')
        
            # rotate with angle -phi.
            r_edge = (edges[j,:,0]*np.cos(-phi)+edges[j,:,1]*-np.sin(-phi))*((j%2)==0) +\
                (edges[j,:,0]*np.sin(-phi)+edges[j,:,1]*np.cos(-phi))*((j%2)==1)
            
            # remove outliers and calculate mean
            bool1 = abs(r_edge-r_edge.mean())<outlier
            mu_edges[j,0]= r_edge[bool1].mean()
            mu_edges[j,1]= r_edge[bool1].std()
            
            edges[j,np.logical_not(bool1),:] = np.nan
        
        x0 = (mu_edges[0,0]+mu_edges[2,0])/2+Jmin
        y0 = (mu_edges[1,0]+mu_edges[3,0])/2+Imin           
        w = mu_edges[0,0]-mu_edges[2,0]
        h = mu_edges[1,0]-mu_edges[3,0]
        sigma = np.sqrt(np.sum(mu_edges[:,1]**2)/mu_edges.shape[0])

        stats_rect[i,0] = x0
        stats_rect[i,1] = y0
        stats_rect[i,2] = w
        stats_rect[i,3] = h
        stats_rect[i,4] = phi
        stats_rect[i,5] = sigma
        
    # plt.plot(stats_rect[:,0], stats_rect[:,1], 'o')
    
    # plt.savefig("test.png")
    # plt.show()
    

    return stats_rect

def fitCentroids(gray, threshold = 75):
   
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, 8, cv2.CV_32S)
    
    positions = centroids[1:,:]

    if centroids.shape[0]<=1:
        return []
    else:
    
        
        
        # find closest point for each point and calculate angle.
        dist = distance.cdist(positions,positions,'euclidean')
        dist[dist==0] = 1e4 # remove zeros.
        Imin = np.argmin(dist, axis=0)
        angles = np.arctan((positions[Imin,1]-positions[:,1])/(positions[Imin,0]-positions[:,0]))
            
        angles = np.mod(angles, np.pi/2)
        angles[angles>np.pi/4]-=np.pi/2
        phi = np.median(angles)
        
        # calculate sum of intensity 
        intensity_sums = np.bincount(labels.flatten(), weights=gray.flatten(), minlength=num_labels)
        
        stats_centroids = np.hstack((centroids[1:,:], stats[1:,2:4], phi*np.ones((centroids.shape[0]-1,1)), intensity_sums[1:, np.newaxis] ))
        return stats_centroids





def match_positions_unique(pos_nom, parameters):
    N = len(pos_nom)
    M = len(parameters)

    # if no parameters at all, return all NaNs
    if M == 0:
        return np.full((N, 2), np.nan)

    # query nearest neighbor for each pos_nom
    tree = KDTree(parameters[:,:2])
    dist, idx = tree.query(pos_nom, k=1)

    # prepare output
    matched = np.full((N, 2), np.nan)
    intensity = np.zeros(N)

    # keep track of which parameter index is already assigned
    best_match = {}  # parameter_idx -> (nom_idx, distance)

    for nom_idx, (param_idx, d) in enumerate(zip(idx, dist)):
        if param_idx not in best_match or d < best_match[param_idx][1]:
            best_match[param_idx] = (nom_idx, d)

    # fill only the best matches
    for param_idx, (nom_idx, d) in best_match.items():
        matched[nom_idx] = parameters[param_idx,:2]
        intensity[nom_idx] = parameters[param_idx,5]

    return matched, intensity


def calculateTilts(parameters):
    # possible bug in indexing below.
    
    if len(parameters)>0 and len(parameters.shape)>1:
        # indexing
        parameters = np.array(parameters)
        
        
        # read nominal positions
        with open("nominalPositions.csv","r") as fp:
            pos_nom = np.loadtxt(fp,  delimiter = ',')[:,:2]
        
#        bool_nan = np.isnan(pos_nom[:,0])
         

            
        XY,_ = match_positions_unique(pos_nom, parameters)
        # calculate x,y vector for each point.
        XY -= pos_nom
    
        
        
        # calculate tilts. Signs: Ry ~ x , Rx ~ -y. 
        tiltMeasurement = np.zeros((N_spots[0],N_spots[1],2))
        tiltMeasurement[:,:,0] = XY[:,0].reshape(N_spots)*spix/(2*FL)
        tiltMeasurement[:,:,1] = -XY[:,1].reshape(N_spots)*spix/(2*FL)
        return tiltMeasurement
    else:
        return np.zeros((N_spots[0],N_spots[1],2))*np.nan

def calibrateNominalPositions(positions, Rz):
    # calculate pitch
    dist = distance.cdist(positions,positions,'euclidean').flatten()
    pitch_median = np.median(dist[np.logical_and(dist>0.8*pitch/spix, dist<1.2*pitch/spix)])

    # rotate to zero.
    Xr = positions[:,0]*np.cos(-Rz) -positions[:,1]*np.sin(-Rz)
    Yr = positions[:,0]*np.sin(-Rz) +positions[:,1]*np.cos(-Rz)

    # divide by pitch to get indices.
    Ix = np.round(Xr/pitch_median).astype(int)
    Ix -= Ix.min()
    Iy = np.round(Yr/pitch_median).astype(int)
    Iy -=Iy.min()
    N_x = 19
    N_y = 19    
    sorted_data = np.zeros((N_y, N_x,2))
    for i in range(N_y):
        for j in range(N_x):
            idx =np.where(np.logical_and(Ix==j, Iy==i))[0]
            if len(idx)>0:

                sorted_data[i,j,:] = positions[idx[0],:]
            else:
                sorted_data[i,j,:] = np.nan
    
    crop_data = sorted_data[0:N_spots[0],1:N_spots[1]+1,:]
    
    #flip results in x&y-direction. 
    crop_data = crop_data[::-1,::-1,:].reshape((N_spots[0]*N_spots[1], 2))
    
    if np.sum(np.isnan(crop_data))>0:
        print("warning: nans in calibration!")
    
    
    with open("nominalPositions.csv","w") as fp:   
       np.savetxt(fp, crop_data, delimiter = ',')

def corner_points_square(parms):
    x0  = parms[0]
    y0  = parms[1]
    Rz  = parms[2]
    W = parms[3]
    corners = np.array([ [0.5*W,-0.5*W], [0.5*W,0.5*W],[-0.5*W,0.5*W],[-0.5*W,-0.5*W],[0.5*W,-0.5*W]])
    # rotate + translate
    xcorners = np.cos(Rz)*corners[:,0] -np.sin(Rz)*corners[:,1]+x0
    ycorners = np.sin(Rz)*corners[:,0] +np.cos(Rz)*corners[:,1]+y0
    
    return np.vstack((xcorners,ycorners)).T


def half_max_positions(intensity):
    intensity = np.asarray(intensity)
    N = len(intensity)
    
    # Find maximum and half maximum
    max_val = np.max(intensity)
    half_max = max_val / 2.0
    
    # Rising edge crossing
    rising_idx = np.where(intensity >= half_max)[0][0]
    x0, y0 = rising_idx - 1, intensity[rising_idx - 1]
    x1, y1 = rising_idx, intensity[rising_idx]
    rising_pos = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    # Falling edge crossing
    falling_idx = np.where(intensity >= half_max)[0][-1]
    if falling_idx == intensity.shape[0]-1:
        falling_pos = falling_idx
    else:    
        x0, y0 = falling_idx, intensity[falling_idx]
        x1, y1 = falling_idx + 1, intensity[falling_idx + 1]
        falling_pos = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)
    
    # Average position
    avg_pos = 0.5 * (rising_pos + falling_pos)
    
    return rising_pos, falling_pos, avg_pos

def calculate_stage_position_images(Istart, x_im, dy_im, N_im, im_dir):
    # read nominal positions
    with open("nominalPositions.csv","r") as fp:
        pos_nom = np.loadtxt(fp,  delimiter = ',')[:,:2]
    
    
    bool_nan = np.isnan(pos_nom[:,0])

    ImiddleRow = np.zeros((N_im, N_spots[1]))  
            
    for i in range(N_im):
        sn = str(1000+i+Istart)[-3:]
        f_im = "/image_%s.csv" % (sn)
        with open(im_dir+f_im,'r') as fp:
            parameters = np.loadtxt(fp, delimiter = ',')
        
        if len(parameters)>1 and len(parameters.shape)>1:
            parameters = remove_outliers(parameters)
        
            XY,I = match_positions_unique(pos_nom, parameters)


            # #I = np.zeros((pos_nom.shape[0]))
            # # calculate x,y vector for each point.
            # I = parameters[idx,5]
            
            # # if fewer parameters than pos_nom, fill unused slots with 0
            # if parameters.shape[0] < pos_nom.shape[0]:
            #     used = set(idx)              # parameters actually used
            #     unused_nom = [j for j in range(pos_nom.shape[0]) if j not in used]

            #     for j in unused_nom:
            #         I[j] = 0
            
            
            ImiddleRow[i,:] = I.reshape((N_spots[0],N_spots[1]))[N_spots[0]//2,:]
    
    # edge detection.
    average_positions = np.zeros(N_im)
    for i in range(ImiddleRow.shape[1]):
        if ImiddleRow[:,i].max()>0:
    
            xr,xf, xa = half_max_positions(ImiddleRow[:,i])
            average_positions[i] = xa
            
    y_pos_center = average_positions[average_positions>0].mean()
    
    y_pos_images = (np.arange(N_im)-y_pos_center)*dy_im
    
    return np.ones(N_im)*x_im, y_pos_images

def remove_outliers(pos_nom):
    # filter points.
    if pos_nom.shape[0]>2:
        s_med_x = np.median(pos_nom[:,2])
        s_med_y = np.median(pos_nom[:,3])
        bool1 = pos_nom[:,2]*pos_nom[:,3]> s_med_x*s_med_y*0.25
        
        pos_nom = pos_nom[bool1,:]
    return pos_nom

def calculate_start_end_images( Istart, N_im, im_dir):
    # read nominal positions
    with open("nominalPositions.csv","r") as fp:
        pos_nom = np.loadtxt(fp,  delimiter = ',')[:,:2]


    Ispots = np.zeros((N_im, N_spots[0]*N_spots[1]))  
            
    for i in range(N_im):
        sn = str(1000+i+Istart)[-3:]
        f_im = "/image_%s.csv" % (sn)
        with open(im_dir+f_im,'r') as fp:
            parameters = np.loadtxt(fp, delimiter = ',')
        
        if len(parameters)>0 and len(parameters.shape)>1:
            
            parameters = remove_outliers(parameters)

            XY,I = match_positions_unique(pos_nom, parameters)
            
            Ispots[i,:] = I
    

    xr = np.zeros(Ispots.shape[1])
    xf = np.zeros(Ispots.shape[1])
    for i in range(Ispots.shape[1]):
        if Ispots[:,i].max()>0:
            
            idx = np.where(Ispots[:,i]>0.95*Ispots[:,i].max())[0]
            xr[i] = idx[0]
            xf[i] = idx[-1]

    
    return xr,xf



def fit_images_test_dir(imdir, N, threshold = 75):
    
    #imdir = r'C:\data\20250905 y-scan'
    
    for i in range(N):
        sn = str(1000+i)[-3:]
        f_im = "/image_%s.png" % (sn)
        im_pil = Image.open(imdir+f_im)
        #Convert the image to a NumPy array
        gray = np.array(im_pil)
    
        #stats_rectangle = fitRectangle(gray)
        stats_rectangle = fitCentroids(gray, threshold)
        
        with open(imdir + f_im.split('.')[0]+ ".csv" , "w") as fp:
            np.savetxt(fp,stats_rectangle, delimiter = ',')

def calibrate_with_first_test_im(f_im):

    with open(f_im,'r') as fp:
        stats_rectangle = np.loadtxt(fp, delimiter = ',')
    
    calibrateNominalPositions(stats_rectangle[:,:2], stats_rectangle[0,4])
    
def calculate_repro():

    
    Rx = np.zeros((19*19, 99))
    Ry = np.zeros((19*19, 99))
    for i in range(1,100):
        f_im = "image_%i.csv" % (1000+i)
        with open(imdir+f_im,'r') as fp:
            stats_rectangle = np.loadtxt(fp, delimiter = ',')
            
        tilts= calculateTilts(stats_rectangle[:,:2])
        
        Rx[:,i-1]= tilts[:,:,0].flatten()
        Ry[:,i-1]= tilts[:,:,1].flatten()
        
    
    # subtract average 
    Rx_mean = np.mean(Rx, axis=0)
    Ry_mean = np.mean(Ry, axis=0)
    Rx = Rx-np.tile(Rx_mean,(Rx.shape[0],1))
    Ry = Ry-np.tile(Ry_mean,(Ry.shape[0],1))
    
    # calculate std.
    sRx = np.std(Rx, axis = 1)
    sRy = np.std(Ry, axis = 1)    
    
    #fig = plt.figure(1)
    fig, ax = plt.subplots()
    im= ax.imshow(sRx.reshape(19,19))
    fig.colorbar(im, ax=ax, label = "Rx repro- std[rad]")
        
    
    fig, ax = plt.subplots()
    im= ax.imshow(sRy.reshape(19,19))
    fig.colorbar(im,ax=ax, label = "Ry repro- std[rad]")
    plt.show()
    
    # calculate average repro number on center of image.
    print("repro Rx[rad] 1sigma", sRx.reshape(19,19)[2:-2,2:-2].flatten().mean())
    print("repro Ry[rad] 1sigma", sRy.reshape(19,19)[2:-2,2:-2].flatten().mean())
    

def analyzeDataset(imdir, dx_im, dy_im, N_x, N_im, threshold, fit =True):
    if fit:
        fit_images_test_dir(imdir,N_im*N_x, threshold)
    sn = str(1000+N_x//2*N_im+N_im//2)[-3:]
    calibrate_with_first_test_im(os.path.join(imdir,'image_%s.csv' % sn))
    
    # calculate profile slopes.
    
    stage_x = np.arange(N_x)[::-1]*dx_im 
    stage_x -= stage_x.mean()
    I = np.argmin(abs(stage_x))
    idx_x = np.hstack((np.arange(I,N_x),np.arange(0,I)[::-1]))

    x,y = calculate_stage_position_images(0,stage_x.min(), dy_im, N_im, imdir)   
    stitch.initArrays(N_x, N_im, dx_im, dy_im, stage_x.min(), y[0])

    for j in idx_x:
       # print(j)
        x,y = calculate_stage_position_images(j*N_im,stage_x[j], dy_im, N_im, imdir)
        Ir,If = calculate_start_end_images( j*N_im, N_im, imdir)
        
 
        
        Rx_im = np.zeros((N_im, N_spots[0]*N_spots[1]))
        Ry_im = np.zeros((N_im, N_spots[0]*N_spots[1]))
        
        # start stitching from center.
        istart = np.where(y<0)[0][-1]
        idx = np.hstack((np.arange(istart,N_im),np.arange(0,istart)[::-1]))
    
        i_edge = 5
        for i in idx:
            sn = str(1000+i+j*N_im)[-3:]
            f_im = "/image_%s.csv" % (sn)
            with open(imdir+f_im,'r') as fp:
                parameters = np.loadtxt(fp, delimiter = ',')
            if len(parameters)>0 and len(parameters.shape)>1:
                parameters = remove_outliers(parameters)
    
            # filter points below 95% intensity    
            bool1 = np.logical_and(i>=Ir+i_edge, i<=If-i_edge)
        
            tilts = calculateTilts(parameters)
        
            # # filter first and last column. (only half bright)
            # tilts[:,0,:] = np.nan
            # tilts[:,-1,:] = np.nan
            
        
            # # filter points below 95% intensity.
            Rx_im[i,:] = tilts[:,:,0].flatten()
            Rx_im[i,np.logical_not(bool1)] = np.nan
            tilts[:,:,0] = Rx_im[i,:].reshape(N_spots[0], N_spots[1])
        
            # # filter points below 95% intensity
            Ry_im[i,:] = tilts[:,:,1].flatten()    
            Ry_im[i,np.logical_not(bool1)] = np.nan
            tilts[:,:,1] = Ry_im[i,:].reshape(N_spots[0], N_spots[1])
        
            stitch.stitch(tilts, (x[i], y[i]))
        
    #calculate_repro()
    
    with open(os.path.join(imdir, "posx.csv"),"w") as fp:   
        np.savetxt(fp, stitch.x_pos, delimiter = ',')
    with open(os.path.join(imdir, "posy.csv"),"w") as fp:   
        np.savetxt(fp, stitch.y_pos, delimiter = ',')
    with open(os.path.join(imdir, "Rx.csv"),"w") as fp:   
        np.savetxt(fp, stitch.tilts_av[:,:,0], delimiter = ',')
    with open(os.path.join(imdir, "Ry.csv"),"w") as fp:   
        np.savetxt(fp, stitch.tilts_av[:,:,1], delimiter = ',')
    z_stitch = stitch.integrateTilts(stitch.tilts_av, pitch)
    with open(os.path.join(imdir, "Z.csv"),"w") as fp:   
        np.savetxt(fp, z_stitch, delimiter = ',')
    
    plt.figure(1)
    plt.plot(Rx_im)
    
    plt.savefig(os.path.join(imdir, "rx_plot.png"), dpi = 150)
    
    plt.figure(2)
    plt.plot(Ry_im)

    plt.savefig(os.path.join(imdir, "ry_plot.png"), dpi = 150)    
    fig3 = plt.figure(3)

    plt.title("stitched Rx")
    CS= plt.contourf(stitch.x_pos, stitch.y_pos, stitch.tilts_av[:,:,0])
    plt.ylim(-25e-3, 25e-3)
    cbar = fig3.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    plt.savefig(os.path.join(imdir, "rx_surface_plot.png"), dpi = 150)
    
    
    fig3 = plt.figure(4)
    plt.ylim(-25e-3, 25e-3)
    plt.title("stitched Ry")
    CS= plt.contourf(stitch.x_pos, stitch.y_pos, stitch.tilts_av[:,:,1])
    cbar = fig3.colorbar(CS)
    cbar.ax.set_ylabel('error[rad]')
    plt.savefig(os.path.join(imdir, "ry_surface_plot.png"), dpi = 150)
    

    fig3 = plt.figure(5)
    plt.ylim(-25e-3, 25e-3)
    plt.title("integrated Z")
    CS= plt.contourf(stitch.x_pos, stitch.y_pos, z_stitch)
    cbar = fig3.colorbar(CS)
    cbar.ax.set_ylabel('Z[m]')
    
    plt.savefig(os.path.join(imdir, "z_surface_plot.png"), dpi = 150)
    
    plt.show()


if __name__ == "__main__":
    
    
    imdir_ = r'C:\data\test004'

    dx_im_ = 5e-3
    dy_im_ = 1e-3
    N_im_ = 60
    N_x_ = 9



    analyzeDataset(imdir_, dx_im_, dy_im_, N_x_, N_im_, 75, fit = False)
    
    
    
    # for i in range(10):
    #     imdir_ = r'C:\data\repro_%i' % i
    
    #     dx_im_ = 0*5e-3
    #     dy_im_ = 1e-3
    #     N_im_ = 60
    #     N_x_ = 1
    
    
    
    #     analyzeDataset(imdir_, dx_im_, dy_im_, N_x_, N_im_, 75, fit = False)
    
    # imdir = '.'
    # f_im = "/Image__2025-09-26__09-02-52.png" 
    # im_pil = Image.open(imdir+f_im)
    # #Convert the image to a NumPy array
    # gray = np.array(im_pil)

    # #stats_rectangle = fitRectangle(gray)
    # stats_rectangle = fitCentroids(gray, 65)
    
    # with open(imdir + f_im.split('.')[0]+ ".csv" , "w") as fp:
    #     np.savetxt(fp,stats_rectangle, delimiter = ',')

    # calibrate_with_first_test_im(imdir +  f_im.split('.')[0]+ ".csv" )
    # #fit_images_test_dir(imdir,N_im*N_x)
    

    # # actually image is 18 x 17. 
    
    
    
    # #plt.show()
    # # for i in range(30):
    # #     plt.figure(1+i)
    # #     plt.plot(Rx[i*10:i*10+10,:].T)
    # # plt.show()
    
    # # plt.plot(Rx_mean)
    # # plt.plot(np.median(Rx,axis=0))
