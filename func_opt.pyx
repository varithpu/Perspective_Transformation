import cv2 as cv
import numpy as np
cimport cython
import math

DTYPE = np.intc

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int[:, :] warpHomography_optimized(unsigned char[:, :] frameC, double[:,:] H, unsigned short[:,:,:] hull, int m, int n):
    cdef int[:, :] frameW
    cdef int i, j, x_w, y_w
    cdef int x_min = int(frameC.shape[1])
    cdef int y_min = int(frameC.shape[0])
    cdef int x_max = 0
    cdef int y_max = 0
    for k in range(4):
        if hull[k][0][0] > x_max:
            x_max = int(hull[k][0][0])
        if hull[k][0][0] < x_min:
            x_min = int(hull[k][0][0])
        if hull[k][0][1] > y_max:
            y_max = int(hull[k][0][1])
        if hull[k][0][1] < y_min:
            y_min = int(hull[k][0][1])
    frameW = np.zeros((n,m), dtype=DTYPE) 
    for i in range(x_min,x_max): # x-axis
        for j in range(y_min,y_max): # y-axis
            x_w = int((H[0,0]*i + H[0,1]*j + H[0,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            y_w = int((H[1,0]*i + H[1,1]*j + H[1,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            if x_w>=0 and x_w<m and y_w>=0 and y_w<n:
                frameW[y_w,x_w] = int(frameC[j,i])
    return frameW

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int[:, :, :] warpHomographyInv_optimized(unsigned char[:, :, :] frameC, double[:,:] H, int m, int n):
    cdef int[:, :, :] frameW
    cdef int i, j, x_w, y_w
    frameW = np.zeros((n,m,3), dtype=DTYPE) 
    cdef int frameC_s0 = int(frameC.shape[0])
    cdef int frameC_s1 = int(frameC.shape[1])
    for i in range(frameC_s1): # x-axis
        for j in range(frameC_s0): # y-axis
            x_w = int((H[0,0]*i + H[0,1]*j + H[0,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            y_w = int((H[1,0]*i + H[1,1]*j + H[1,2])/(H[2,0]*i + H[2,1]*j + H[2,2]))
            if x_w>=0 and x_w<m and y_w>=0 and y_w<n:
                frameW[y_w,x_w,0] = int(frameC[j,i,0])
                frameW[y_w,x_w,1] = int(frameC[j,i,1])
                frameW[y_w,x_w,2] = int(frameC[j,i,2])
    return frameW
