#ENPM673 Project1
import cv2 as cv
import numpy as np
import copy
import math
from functions import func_opt
import time

# function for Fourier Transform
def ft_2D(input):
    ft0 = np.fft.fft2(input)
    ft1 = np.fft.ifftshift(ft0)
    return ft1    

# function for Inverse Fourier Transform
def ift_2D(input):
    ift0 = np.fft.ifftshift(input)
    ift1 = np.fft.ifft2(ift0)
    return ift1  

# function for homography calculation of AR tag 
def homography_cal1(pc):    
    pc = pc.reshape((4,2))
    pw = [(0,0), (400,0), (400,300), (0,300)]
    A = np.empty([0,9],float)
    for i in range(4):         
        temp = np.array([[pc[i][0], pc[i][1], 1, 0, 0, 0, -pw[i][0]*pc[i][0], -pw[i][0]*pc[i][1], -pw[i][0]], 
                        [0, 0, 0, pc[i][0], pc[i][1], 1, -pw[i][1]*pc[i][0], -pw[i][1]*pc[i][1], -pw[i][1]]])
        A = np.concatenate((A, temp), axis=0)
    _,_,Vt = np.linalg.svd(A)
    V = np.transpose(Vt)
    h = V[:,-1]
    h = h/h[-1]
    H = h.reshape((3,3))
    return H

# function for homography calculation of testudo image 
def homography_cal2(pc):    
    pc = pc.reshape((4,2))
    pw = [(-200,-130), (480,-130), (480,400), (-200,400)]
    A = np.empty([0,9],float)
    for i in range(4):         
        temp = np.array([[pc[i][0], pc[i][1], 1, 0, 0, 0, -pw[i][0]*pc[i][0], -pw[i][0]*pc[i][1], -pw[i][0]], 
                        [0, 0, 0, pc[i][0], pc[i][1], 1, -pw[i][1]*pc[i][0], -pw[i][1]*pc[i][1], -pw[i][1]]])
        A = np.concatenate((A, temp), axis=0)
    _,_,Vt = np.linalg.svd(A)
    V = np.transpose(Vt)
    h = V[:,-1]
    h = h/h[-1]
    H = h.reshape((3,3))
    return H

# function for adjusting the order of centroids so that first 2 points always form a longer side of A4 paper
def adjust_hull(hull):
    hull_temp = (hull.reshape((4,2))).astype(float)
    d1 = (hull_temp[0][0]-hull_temp[1][0])**2 + (hull_temp[0][1]-hull_temp[1][1])**2
    d2 = (hull_temp[1][0]-hull_temp[2][0])**2 + (hull_temp[1][1]-hull_temp[2][1])**2
    if d2 > d1:
        temp = hull[0]
        hull = np.delete(hull,0,0)
        hull = np.concatenate((hull, np.array([temp])), axis=0)
    return hull

# function for decode AR tag
def tag_decode(tag):
    decode = np.empty((4,4))
    row,col = tag.shape
    nrow, ncol = int(row/4), int(col/4)
    for i in range(4):
        for j in range(4):
            # cv.imshow('Video', tag[nrow*i:nrow*(i+1), ncol*j:ncol*(j+1)])
            if np.average(tag[nrow*i:nrow*(i+1), ncol*j:ncol*(j+1)])>128:
                decode[i,j] = 1
            else:
                decode[i,j] = 0
    return(decode)

# function for finding how many times the AR tag need to be rotated
def rotate_tag(decode):
    if decode[-1][-1] == 1:
        shift = 0
    elif decode[0][-1] == 1:
        shift = 1
    elif decode[0][0] == 1:
        shift = 2
    elif decode[-1][0] == 1:
        shift = 3
    else:
        print('orientation decoding error')        
    return shift

# function for decoding id of rotated AR tag
def get_tag_id(decode):
    bit_1 = decode[1][1]
    bit_2 = decode[1][2]
    bit_3 = decode[2][2]
    bit_4 = decode[2][1]
    id = bit_1 + bit_2*2 + bit_3*(2**2) + bit_4*(2**3)
    return id
    
# function for shifting order of centroids so that the calculated homography yields correct orientation
def shift_hull(hull, shift):
    for i in range(shift):
        hull = hull.reshape((4,2))
        temp = hull[0]
        hull = np.delete(hull,0,0)
        hull = np.concatenate((hull, np.array([temp])), axis=0)
    return hull

# function for finding radius of circle around AR tag for filtering centroids
def find_radius(hull):
    hull_temp = (hull.reshape((4,2))).astype(float)
    d1 = (hull_temp[0][0]-hull_temp[1][0])**2 + (hull_temp[0][1]-hull_temp[1][1])**2
    d2 = (hull_temp[1][0]-hull_temp[2][0])**2 + (hull_temp[1][1]-hull_temp[2][1])**2
    radius = 1.3*math.sqrt(d1+d2)/2
    return radius

# function for warping image with Homography, converting camera frame into world frame (not used due to low performance)
def warpHomography_opt(frameC, H, hull, m, n):
    frameW = np.zeros((n,m))
    x_min = frameC.shape[1]
    y_min = frameC.shape[0]
    x_max = 0
    y_max = 0
    hull = hull.reshape((4,2))
    for k in range(4):
        if hull[k][0] > x_max:
            x_max = hull[k][0]
        if hull[k][0] < x_min:
            x_min = hull[k][0]
        if hull[k][1] > y_max:
            y_max = hull[k][1]
        if hull[k][1] < y_min:
            y_min = hull[k][1]
    for i in range(x_min,x_max): # x-axis
        for j in range(y_min,y_max): # y-axis
            x_w = int((H[0,0]*i + H[0,1]*j + H[0,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            y_w = int((H[1,0]*i + H[1,1]*j + H[1,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            if x_w>=0 and x_w<m and y_w>=0 and y_w<n:
                frameW[y_w, x_w] = frameC[j,i]
    return  frameW

# function for warping image with Inverse Homography, converting world frame into camera frame (not used due to low performance)
def warpHomography_inv(frameC, H, m, n):
    frameW = np.zeros((n,m,3))
    for i in range(frameC.shape[1]): # x-axis
        for j in range(frameC.shape[0]): # y-axis
            x_w = int((H[0,0]*i + H[0,1]*j + H[0,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            y_w = int((H[1,0]*i + H[1,1]*j + H[1,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            if x_w>=0 and x_w<m and y_w>=0 and y_w<n:
                frameW[y_w,x_w,0] = int(frameC[j,i,0])
                frameW[y_w,x_w,1] = int(frameC[j,i,1])
                frameW[y_w,x_w,2] = int(frameC[j,i,2])
    return  frameW

# initialize radius value for filtering centroids
radius = 500
# import video and image
capture = cv.VideoCapture('ENPM673/Project1/1tagvideo.mp4') # cheange to your video directory
img = cv.imread('ENPM673/Project1/testudo.png') # cheange to your testudo image directory
# main loop
while True:
        isTrue, frame = capture.read()
        if isTrue == True: 
            # convert video frame to gray scale
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rows, cols = frame_gray.shape
            # threshold to black/white image
            frame_gray_bw = copy.deepcopy(frame_gray)
            frame_gray_bw[frame_gray_bw>=150] = 255
            frame_gray_bw[frame_gray_bw<150] = 0
            # apply Fourier Transform
            ft = ft_2D(frame_gray_bw)
            crow,ccol = int(rows/2) , int(cols/2)
            # apply high pass filter in frequency domain
            ft[crow-60:crow+60, ccol-60:ccol+60] = 0
            # apply Inverse Fourier Transform
            ift = abs(ift_2D(ft))
            # thresholding
            ift[ift>=0.2*ift.max()] = 255
            ift[ift<0.2*ift.max()] = 0
            ift = np.float32(ift)   
            # extract corners in image using Harris detector
            corners = cv.cornerHarris(ift,80,3,0.04)     
            ift_rgb = cv.cvtColor(ift,cv.COLOR_GRAY2RGB)
            ift_rgb[corners>0.1*corners.max()]=[0,255,0]
            _, corners_t = cv.threshold(corners,0.1*corners.max(),255,0) 
            corners_t = np.uint8(corners_t)       
            # find centroid of corner elements      
            _, _, stats, centroids = cv.connectedComponentsWithStats(corners_t)
            # filter undesired centroids considering area of corner elements
            centroids_filtered = np.empty((0,2))
            max_area = 0   
            for i in range(stats.shape[0]):
                if (stats[i, cv.CC_STAT_AREA] < 100000) and (stats[i, cv.CC_STAT_AREA] > 2*radius):
                    centroids_filtered = np.concatenate((centroids_filtered,np.array([centroids[i]])), axis=0)
                    if stats[i, cv.CC_STAT_AREA] > max_area:
                        max_area = stats[i, cv.CC_STAT_AREA]
                        max_node = copy.deepcopy(centroids[i])
                        max_node = np.uint16(max_node)  
            # filter undesired centroids distance from the AR tag
            cv.circle(ift_rgb, (max_node[0], max_node[1]), radius, (255,0,0), 5)         
            centroids_filtered_limited = np.empty((0,2))   
            for i in range(centroids_filtered.shape[0]):
                if (centroids_filtered[i][0]-max_node[0])**2 + (centroids_filtered[i][1]-max_node[1])**2 < radius**2 :
                    centroids_filtered_limited = np.concatenate((centroids_filtered_limited,np.array([centroids_filtered[i]])), axis=0)
            centroids_filtered_limited = np.uint16(centroids_filtered_limited)
            for pix in centroids_filtered_limited:
                cv.circle(ift_rgb, (pix[0], pix[1]), 5, (255,0,0), -1)
            # detect outermost centroids using Convex Hull
            centroids_filtered_limited = np.float32(centroids_filtered_limited)
            hull = cv.convexHull(centroids_filtered_limited)
            hull = np.uint16(hull)    
            # reduce number of centroids to 4 points considering distance from AR tag
            while hull.shape[0] > 4:
                dist_min = math.inf
                for i in range(hull.shape[0]):
                    dist = (float(hull[i][0][0])-max_node[0])**2 + (float(hull[i][0][1])-max_node[1])**2 
                    if dist < dist_min:
                        dist_min = dist
                        min_idx = i
                hull = np.delete(hull,min_idx,0)
            radius = int(find_radius(hull))
            # since we detect corners of A4 paper, adjust the order of centroids so that first 2 points always form a longer side
            hull_adjusted = adjust_hull(hull)
            for pix in hull_adjusted:
                cv.circle(ift_rgb, (pix[0][0], pix[0][1]), 5, (0,0,255), -1) 
            # calculate Homograpy 
            H = homography_cal1(hull_adjusted)
            # warp gray scale image with Homography, converting AR tag into world frame       
            frame_gray_warp = func_opt.warpHomography_optimized(frame_gray, H, hull_adjusted, 400, 300)
            # fill in holes with median filter
            frame_gray_warp = cv.medianBlur(np.uint8(frame_gray_warp),3)
            fw_col, fw_row = frame_gray_warp.shape
            # crop image and get the AR tag
            marker = copy.deepcopy(frame_gray_warp[int(0.34*fw_col):int(0.61*fw_col),int(0.39*fw_row):int(0.59*fw_row)])
            # threshod AR tag
            marker[marker>=0.8*marker.max()] = 255
            marker[marker<0.8*marker.max()] = 0
            # decode AR tag
            decode = tag_decode(marker)
            # get AR tag corners and rotate the AR tag into upright position
            shift = rotate_tag(decode)
            hull_shifted = shift_hull(hull_adjusted, shift)
            # recalculate Homograpy with correct orientation
            H_R = homography_cal1(hull_shifted)
            # repeat the process of decoding, using new Homography with correct orientation
            frame_gray_warp_R = func_opt.warpHomography_optimized(frame_gray, H_R, hull_adjusted, 400, 300)
            frame_gray_warp_R = cv.medianBlur(np.uint8(frame_gray_warp_R),3)
            marker_R = copy.deepcopy(frame_gray_warp_R[int(0.38*fw_col):int(0.66*fw_col),int(0.40*fw_row):int(0.62*fw_row)])
            marker_R[marker_R>=0.8*marker_R.max()] = 255
            marker_R[marker_R<0.8*marker_R.max()] = 0    
            decode_R = tag_decode(marker_R)   
            # decode id of rotated AR tag and print it out
            id = int(get_tag_id(decode_R))
            print('id:', end = ' ')
            print(id)
            print('')
            # adjust Homograpy to fit testudo image into AR tag
            H_R = homography_cal2(hull_shifted)
            # calculate inverse Homography
            H_R_inv = np.linalg.inv(H_R)
            # warp testudo image with inverse Homography, converting testudo image from world frame to camera frame
            testudo = func_opt.warpHomographyInv_optimized(img, H_R_inv, cols, rows)
            # fill in holes with median filter
            testudo = cv.medianBlur(np.uint8(testudo),3)
            # place testudo image on AR tag
            blended = cv.addWeighted(frame, 0.5, testudo, 1, 0.5) 
            # create a new frame for placing 3D cube
            blended_cube = copy.deepcopy(blended)
            for pix in hull_adjusted:
                cv.circle(blended_cube, (pix[0][0], pix[0][1]), 5, (0,255,0), -1)
            # create intrinsic camera parameter matrix K
            K = np.array([[1346.10059534175, 0, 932.163397529403],
                          [0, 1355.93313621175, 654.898679624155],
                          [0, 0, 1]])
            # calculate B tilda matrix
            B = np.matmul(np.linalg.inv(K),H_R_inv)
            if np.linalg.det(B)<0:
                B = -B
            # calculate scale factor
            scale = 1/((np.linalg.norm(np.matmul(np.linalg.inv(K),H_R_inv[:,0]))
                               + np.linalg.norm(np.matmul(np.linalg.inv(K),H_R_inv[:,1])))/2)
            # extract r1 r2 r3 t and form final B matrix
            r1 = scale*np.transpose(np.array([B[:,0]]))
            r2 = scale*np.transpose(np.array([B[:,1]]))
            # inverse z-axis since usual image axis will result in z pointing into image
            r3 = -np.transpose(np.array([np.cross(scale*B[:,0],scale*B[:,1])])) 
            t = scale*np.transpose(np.array([B[:,2]]))
            Bf = np.concatenate((r1,r2,r3,t), axis=1)
            # calculate projection matrix P matrix 
            P = np.matmul(K,Bf)
            # create a 3D cube corners in world frame
            width_x = 300
            width_y = 300
            height = 300
            box = np.transpose(np.array([[0,0,0,1],
                                        [0,width_y,0,1],
                                        [width_x,width_y,0,1],
                                        [width_x,0,0,1],
                                        [0,0,height,1],
                                        [0,width_y,height,1],
                                        [width_x,width_y,height,1],
                                        [width_x,0,height,1]]))
            # convert 3D cube corners from world frame to camera frame using projection matrix
            pts = []
            for i in range(8):
                pix = np.matmul(P,box[:,i])
                cv.circle(blended_cube, (int(pix[0]/pix[2]), int(pix[1]/pix[2])), 5, (255,0,0), -1)
                pts.append((int(pix[0]/pix[2]), int(pix[1]/pix[2])))
            # create edges for 3D cube
            for i in range(4):
                if i == 3:
                    cv.line(blended_cube,pts[i],pts[i-3],(0,255,0),5)
                    cv.line(blended_cube,pts[i+4],pts[i+4-3],(0,0,255),5)
                    cv.line(blended_cube,pts[i],pts[i+4],(255,0,0),5)
                else:
                    cv.line(blended_cube,pts[i],pts[i+1],(0,255,0),5)
                    cv.line(blended_cube,pts[i+4],pts[i+4+1],(0,0,255),5)
                    cv.line(blended_cube,pts[i],pts[i+4],(255,0,0),5)
            # display frame 
                # Peoblem 1a : ift(edges only), ift_rgb(edges with corners(green) and centroids(red&blue))
                # Peoblem 1b : marker(original position), marker_R(rotated position)
                # Peoblem 2a : blended
                # Peoblem 2b : blended_cube
            cv.imshow('Video', blended_cube) # pass different variables above to see solution to each problem
            cv.waitKey(20)
        else:
            break
        