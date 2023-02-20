from pickletools import uint8
from re import A
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from sympy import NonSquareMatrixError, frac
import copy
import math
from functions import func_opt
import time

def ft_2D(input):
    ft0 = np.fft.fft2(input)
    ft1 = np.fft.ifftshift(ft0)
    return ft1    

def ift_2D(input):
    ift0 = np.fft.ifftshift(input)
    ift1 = np.fft.ifft2(ift0)
    return ift1  

def homography_cal(pc):    
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
    # M, mask = cv.findHomography(pc, np.uint16(pw))
    return H

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
    # M, mask = cv.findHomography(pc, np.uint16(pw))
    return H

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

def adjust_hull(hull):
    hull_temp = (hull.reshape((4,2))).astype(float)
    d1 = (hull_temp[0][0]-hull_temp[1][0])**2 + (hull_temp[0][1]-hull_temp[1][1])**2
    d2 = (hull_temp[1][0]-hull_temp[2][0])**2 + (hull_temp[1][1]-hull_temp[2][1])**2
    if d2 > d1:
        temp = hull[0]
        hull = np.delete(hull,0,0)
        hull = np.concatenate((hull, np.array([temp])), axis=0)
    return hull

def shift_hull(hull, shift):
    for i in range(shift):
        hull = hull.reshape((4,2))
        temp = hull[0]
        hull = np.delete(hull,0,0)
        hull = np.concatenate((hull, np.array([temp])), axis=0)
    return hull
    
def warpHomography(frameC, H, size, hull):
    if frameC.shape == (frameC.shape[0], frameC.shape[1], 3):  
        frameW = np.zeros((size[1],size[0],3))
    elif frameC.shape == (frameC.shape[0], frameC.shape[1]):  
        frameW = np.zeros((size[1],size[0]))
    else:
        print('warp error')     
        return None
    x_min = math.inf
    y_min = math.inf
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
            xc = np.transpose(np.array([[i, j, 1]]))
            xw = np.matmul(H,xc)
            xw = xw/xw[-1]
            xw = np.uint16(xw)
            if xw[0]>=0 and xw[0]<size[0] and xw[1]>=0 and xw[1]<size[1]:
                frameW[xw[1], xw[0]] = frameC[xc[1],xc[0]]
    return  cv.medianBlur(np.uint8(frameW),3)

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

def warpHomography_inv(frameC, H, size):
    if frameC.shape == (frameC.shape[0], frameC.shape[1], 3):  
        frameW = np.zeros((size[1],size[0],3))
    elif frameC.shape == (frameC.shape[0], frameC.shape[1]):  
        frameW = np.zeros((size[1],size[0]))
    else:
        print('warp error')     
        return None      
    for i in range(frameC.shape[1]): # x-axis
        for j in range(frameC.shape[0]): # y-axis
            xc = np.transpose(np.array([[i, j, 1]]))
            xw = np.matmul(H,xc)
            xw = xw/xw[-1]
            xw = np.uint16(xw)
            if xw[0]>=0 and xw[0]<size[0] and xw[1]>=0 and xw[1]<size[1]:
                frameW[xw[1], xw[0]] = frameC[xc[1],xc[0]]
    return  cv.medianBlur(np.uint8(frameW),3)
        
capture = cv.VideoCapture('ENPM673/Project1/1tagvideo.mp4')
img = cv.imread('ENPM673/Project1/testudo.png')
while True:
        isTrue, frame = capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if isTrue == True: 
            rows, cols = frame_gray.shape
            ft = ft_2D(frame_gray)
            # cv.imshow('Video', frame_gray)
            # plt.imshow(frame_gray, cmap='gray')
            crow,ccol = int(rows/2) , int(cols/2)
            r = 50
            ft[crow-r:crow+r, ccol-r:ccol+r] = 0
            # plt.imshow(np.log(np.abs(ft)), cmap='gray')
            ift = abs(ift_2D(ft))
            t = 0.3
            ift[ift>=t*ift.max()] = 255
            ift[ift<t*ift.max()] = 0
            ift = np.float32(ift)       
            corners = cv.cornerHarris(ift,80,3,0.04)     
            corners = cv.dilate(corners,None)
            ift_rgb = cv.cvtColor(ift,cv.COLOR_GRAY2RGB)
            # ift_rgb[corners>0.4*corners.max()]=[0,255,0]
            ret, corners_t = cv.threshold(corners,0.1*corners.max(),255,0) 
            corners_t = np.uint8(corners_t)               
            ret, labels, stats, centroids = cv.connectedComponentsWithStats(corners_t)
            # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            # corners = cv.cornerSubPix(ift,np.float32(centroids),(5,5),(-1,-1),criteria)
            ## detect all centroids
            # centroids = np.uint16(centroids)
            # for pix in centroids:
            #     cv.circle(ift_rgb, (pix[0], pix[1]), 5, (0,255,0), -1)
            # detect outermost centroids
            pts = np.float32(centroids)
            hull = cv.convexHull(pts)
            hull = np.uint16(hull)
            hull = hull[0:4]         
            hull_adjusted = adjust_hull(hull)
            H = homography_cal(hull_adjusted)
            for pix in hull_adjusted:
                cv.circle(ift_rgb, (pix[0][0], pix[0][1]), 5, (0,255,0), -1)
                # xc = np.transpose(np.array([[pix[0][0], pix[0][1], 1]]))
                # xw = np.matmul(H,xc)
                # xw = xw/xw[-1]
                # xw = np.uint16(xw)
                # cv.circle(ift_rgb, (xw[0][0], xw[1][0]), 5, (0,0,255), -1)    
            # frame_gray_warp = cv.warpPerspective(frame_gray, H, (400,300))
            # frame_gray_warp = warpHomography_opt(frame_gray, H, hull_adjusted, 400, 300)
            frame_gray_warp = func_opt.warpHomography_optimized(frame_gray, H, hull_adjusted, 400, 300)
            frame_gray_warp = cv.medianBlur(np.uint8(frame_gray_warp),3)
            fw_col, fw_row = frame_gray_warp.shape
            marker = copy.deepcopy(frame_gray_warp[int(0.34*fw_col):int(0.61*fw_col),int(0.39*fw_row):int(0.59*fw_row)])
            tm = 0.8
            marker[marker>=tm*marker.max()] = 255
            marker[marker<tm*marker.max()] = 0
            decode = tag_decode(marker)
            shift = rotate_tag(decode)
            hull_shifted = shift_hull(hull_adjusted, shift)
            H_R = homography_cal2(hull_shifted)
            H_R_inv = np.linalg.inv(H_R)
            # H_R_inv = H_R_inv/H_R_inv[-1][-1]
            # frame_gray_warp_rotated = cv.warpPerspective(frame_gray, H_R, (400,300))
            # testudo = cv.warpPerspective(img, H_R_inv, (cols,rows))
            # testudo = warpHomography_inv(img, H_R_inv, (cols,rows))
            testudo = func_opt.warpHomographyInv_optimized(img, H_R_inv, cols, rows)
            testudo = cv.medianBlur(np.uint8(testudo),3)
            blended = cv.addWeighted(frame, 0.5, testudo, 1, 0.5) # result for question 2.a
            for pix in hull:
                cv.circle(blended, (pix[0][0], pix[0][1]), 5, (0,255,0), -1)
            
            # question 2.b
            K = np.array([[1346.10059534175, 0, 932.163397529403],
                          [0, 1355.93313621175, 654.898679624155],
                          [0, 0, 1]])
            B = np.matmul(np.linalg.inv(K),H_R_inv)
            if np.linalg.det(B)<0:
                B = -B
            scale = 1/((np.linalg.norm(np.matmul(np.linalg.inv(K),H_R_inv[:,0]))
                               + np.linalg.norm(np.matmul(np.linalg.inv(K),H_R_inv[:,1])))/2)
            r1 = scale*np.transpose(np.array([B[:,0]]))
            r2 = scale*np.transpose(np.array([B[:,1]]))
            r3 = -np.transpose(np.array([np.cross(scale*B[:,0],scale*B[:,1])])) # inverse z-axis since usual image axis will result in z pointing into image
            t = scale*np.transpose(np.array([B[:,2]]))
            Bf = np.concatenate((r1,r2,r3,t), axis=1)
            P = np.matmul(K,Bf)
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
            pts = []
            for i in range(8):
                pix = np.matmul(P,box[:,i])
                cv.circle(blended, (int(pix[0]/pix[2]), int(pix[1]/pix[2])), 5, (255,0,0), -1)
                pts.append((int(pix[0]/pix[2]), int(pix[1]/pix[2])))

            for i in range(4):
                if i == 3:
                    cv.line(blended,pts[i],pts[i-3],(0,255,0),5)
                    cv.line(blended,pts[i+4],pts[i+4-3],(0,0,255),5)
                    cv.line(blended,pts[i],pts[i+4],(255,0,0),5)
                else:
                    cv.line(blended,pts[i],pts[i+1],(0,255,0),5)
                    cv.line(blended,pts[i+4],pts[i+4+1],(0,0,255),5)
                    cv.line(blended,pts[i],pts[i+4],(255,0,0),5)

            # pts = np.array([pts])
            # pts = pts.reshape((-1,1,2))
            # cv.polylines(blended,[pts],True,(255,0,0), thickness = 2)
            
            cv.imshow('Video', blended)
            # cv.imshow('Video', ift_rgb)
            cv.waitKey(20)
            # plt.imshow(ift_rgb, cmap='gray')
            # plt.pause(0.01)
        else:
            break
        