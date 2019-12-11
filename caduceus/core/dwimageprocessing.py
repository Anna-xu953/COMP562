'''
Created on Jan 30 2018
@author: Jaeil Kim (threeyears@gmail.com)

Functions for DW image processing
'''

import os
import scipy.misc
import numpy as np

import random
import numpy.random
import os.path
import math
import time
from sys import platform

import nibabel as nib
import tensorflow as tf


'''
Merge multiple feature maps to compose output patches of higher resolution (x2)
for sub-pixel convolutional neural network
E.g. 5x5x5x90x8 --> 10x10x10x90
Ref. https://github.com/tetrachrome/subpixel
'''
def TF_ShifePhase(outPatches, HRfactor):
    bsize, a, b, c, g, f = outPatches.get_shape().as_list() # batch size, width, height, depth, gradient volumes, number of feature maps (should be 8)
    r = HRfactor
    bsize = tf.shape(outPatches)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(outPatches, (bsize, a, b, c, g, r, r, r))
    X = tf.transpose(X, (0, 1, 2, 3, 4, 7, 6, 5))  # bsize, a, b, c, g, r, r, r
    X = tf.split(X, a, axis=1)  # a, [bsize, b, c, g, r, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, b, c, g, a*r, r, r
    X = tf.split(X, b, axis=1)  # b, [bsize, c, g, a*r, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, c, g, a*r, b*r, r
    X = tf.split(X, c, axis=1)  # c, [bsize, g, a*r, b*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # bsize, g, a*r, b*r, c*r
    X = tf.reshape(X, (bsize, g, a*r, b*r, c*r))
    X = tf.transpose(X, (0, 2, 3, 4, 1))
    return X

'''
Merge multiple feature maps to compose output patches of higher resolution (x2)
for sub-pixel convolutional neural network
E.g. 5x5x5x90x8 --> 10x10x10x90
Ref. https://github.com/tetrachrome/subpixel
'''
def ShifePhase(outPatches, HRfactor):
    bsize, a, b, c, g, f = outPatches.shape # batch size, width, height, depth, gradient volumes, number of feature maps (should be 8)
    r = HRfactor
    X = np.split(outPatches, f, axis = 5)
    X = [np.squeeze(x, axis = 5) for x in X]
    print X
    # print [bsize, a, b, c, g, r, r, r]
    # X = np.transpose(outPatches, (0, 4, 5, 3, 2, 1))
    # X = np.reshape(X, [bsize, g, a, b, c, r, r, r])
    # X = np.transpose(X, (0, 1, 2, 3, 4, 5, 6, 7))  # bsize, g, a, b, c, r, r, r
    # print X
    # X = np.split(X, a, axis = 2)
    # X = np.concatenate([np.squeeze(x, axis=2) for x in X], axis=4)  # bsize, b, c, g, a*r, r, r
    # X = np.split(X, b, axis = 2)
    # X = np.concatenate([np.squeeze(x, axis=2) for x in X], axis=4)  # bsize, c, g, a*r, b*r, r
    # X = np.split(X, c, axis = 2)
    # X = np.concatenate([np.squeeze(x, axis=2) for x in X], axis=4)  # bsize, g, a*r, b*r, c*r
    # return np.transpose(X, (0, 2, 3, 4, 1))

'''
Decompose a output patch to a set of sub-patches
for sub-pixel convolutional neural network
E.g. 10x10x10x90 --> 5x5x5x90x8 (2x2x2)
'''
def DecomposeOutPatch(outPatches, HRfactor):
    oshape = outPatches.shape
    dshape = [oshape[0], oshape[4], oshape[1]/2, oshape[2]/2, oshape[3]/2, 8]
    # create array

    # dPatches = np.transpose(outPatches, (0, 4, 1, 2, 3))
    # dPatches = np.reshape(dPatches, dshape)
    # dPatches = np.transpose(dPatches, (0, 2, 3, 4, 1, 5))
    return dPatches

'''
Copy patches to volume data
'''

def PatchToHighResVolume(patches, imgshape, highResShape, width, height, depth, hWidth, hHeight, hDepth, step, skipIdx):
    print 'Start Patch Copy to High Resolution Volume'
    # variables
    halfdim = [int(width/2), int(height/2), int(depth/2)]
    halfoffset = [width%2, height%2, depth%2]

    sOutVol = np.zeros(highResShape)
    # Create arrays
    i = halfdim[0]
    pCount = 0
    while i < imgshape[0] - halfdim[0] - 1:
        j = halfdim[1]
        sI = i - halfdim[0]
        eI = i + halfdim[0] + halfoffset[0]
        h_sI = sI*2
        h_eI = h_sI + hWidth
        while j < imgshape[1] - halfdim[1] - 1:
            sJ = j - halfdim[1]
            eJ = j + halfdim[1] + halfoffset[1]
            h_sJ = sJ*2
            h_eJ = h_sJ + hHeight
            k = halfdim[2]
            while k < imgshape[2] - halfdim[2] - 1:
                sK = k - halfdim[2]
                eK = k + halfdim[2] + halfoffset[2]
                h_sK = sK*2
                h_eK = h_sK + hDepth
                sOutVol[h_sI:h_eI, h_sJ:h_eJ, h_sK:h_eK, :] = patches[pCount]
                pCount += 1
                k += step
            j+= step
        i += step
    print 'End Patch Copy to High Resolution Volume'
    return sOutVol

'''
Extract Patches from DW Volumes of lower and higher resolution for super resolution
'''
def ExtractPatchesForSuperResolution(img, highimg, width, height, depth, hWidth, hHeight, hDepth, step, skipThreshold, maxNumPatches):
    print 'Start Patch Extraction for SuperResolution'
    imgshape = img.shape
    # variables
    halfdim = [int(width/2), int(height/2), int(depth/2)]
    halfoffset = [width%2, height%2, depth%2]

    # Create arrays
    patchSet = np.zeros([maxNumPatches, width, height, depth, imgshape[-1]], dtype = np.float32)
    patchSet_High = np.zeros([maxNumPatches, hWidth, hHeight, hDepth, imgshape[-1]], dtype = np.float32)
    skipSet = np.zeros([maxNumPatches], dtype = np.int32)
    pCount = 0
    i = halfdim[0]
    threshold = skipThreshold*width*height*depth*imgshape[-1]
    while i < imgshape[0] - halfdim[0] - 1:
        j = halfdim[1]
        sI = i - halfdim[0]
        eI = i + halfdim[0] + halfoffset[0]
        h_sI = sI*2
        h_eI = h_sI + hWidth
        while j < imgshape[1] - halfdim[1] - 1:
            sJ = j - halfdim[1]
            eJ = j + halfdim[1] + halfoffset[1]
            h_sJ = sJ*2
            h_eJ = h_sJ + hHeight
            k = halfdim[2]
            while k < imgshape[2] - halfdim[2] - 1:
                sK = k - halfdim[2]
                eK = k + halfdim[2] + halfoffset[2]
                h_sK = sK*2
                h_eK = h_sK + hDepth

                patchSet[pCount, :, :, :, :] = img[sI:eI, sJ:eJ, sK:eK, :]
                patchSet_High[pCount, :, :, :, :] = highimg[h_sI:h_eI, h_sJ:h_eJ, h_sK:h_eK, :]

                if (np.count_nonzero(patchSet[pCount, :, :, :, :]) < threshold):
                    skipSet[pCount] = 1

                pCount += 1
                k += step
            j+= step
        i += step

    patchSet = patchSet[0:pCount, :, :, :]
    patchSet_High = patchSet_High[0:pCount, :, :, :]
    skipSet = skipSet[0:pCount]

    print 'End Patch Extraction for SuperResolution'
    return patchSet, patchSet_High, skipSet

'''
Extract Patches from DW Volumes
'''
def ExtractPatches(img, width, height, depth, step, skipThreshold, maxNumPatches):
    print 'Start ExtractPatches'
    imgshape = img.shape
    # variables
    halfdim = [int(width/2), int(height/2), int(depth/2)]
    halfoffset = [width%2, height%2, depth%2]

    # Create arrays
    patchSet = np.zeros([maxNumPatches, width, height, depth, imgshape[-1]], dtype = np.float32)
    skipSet = np.zeros([maxNumPatches], dtype = np.int32)
    pCount = 0
    i = halfdim[0]
    threshold = skipThreshold*width*height*depth*imgshape[-1]
    while i < imgshape[0] - halfdim[0] - 1:
        j = halfdim[1]
        sI = i - halfdim[0]
        eI = i + halfdim[0] + halfoffset[0]
        while j < imgshape[1] - halfdim[1] - 1:
            sJ = j - halfdim[1]
            eJ = j + halfdim[1] + halfoffset[1]
            k = halfdim[2]
            while k < imgshape[2] - halfdim[2] - 1:
                sK = k - halfdim[2]
                eK = k + halfdim[2] + halfoffset[2]
                patchSet[pCount, :, :, :, :] = img[sI:eI, sJ:eJ, sK:eK, :]
                if (np.count_nonzero(patchSet[pCount, :, :, :, :]) < threshold):
                    skipSet[pCount] = 1
                pCount += 1
                k += step
            j+= step
        i += step

    patchSet = patchSet[0:pCount, :, :, :]
    skipSet = skipSet[0:pCount]
    print 'End ExtractPatches'

    return patchSet, skipSet

'''
Extract Non-b0 volumes
'''
def ExtractNonB0Volume(data, bval):
    delFlag = []
    for b in range(len(bval)):
        if bval[b] == 0:
            delFlag.append(b)

    print 'Shape before extraction: ' + `data.shape`
    extData = np.delete(data, delFlag, axis=3)
    print delFlag
    print 'Shape after extraction: ' + `extData.shape`
    return extData
'''
Extract single shell
'''
def ExtractSingleShell(data, bval):
    #noB0Vol = np.count_nonzero(bval)
    
    delFlag = []
    for b in range(len(bval)):
        if bval[b] != 1:
            delFlag.append(b)
    print 'Shape before extraction: ' + `data.shape`
    extData = np.delete(data, delFlag, axis=3)
    print delFlag
    print 'Shape after extraction: ' + `extData.shape`
    return extData
'''
Group the non-b0 volumes using b-values
'''
def GroupNonB0Volumes(img, bvec, bval):
    numvols = len(bval)
    # check bvalues
    dNumVolumes = {}
    b0count = 0
    for b in range(numvols):
        if bval[b] == 0:
            b0count += 1
            continue
        if bval[b] in dNumVolumes:
            dNumVolumes[bval[b]] += 1
        else:
            dNumVolumes[bval[b]] = 1

    # Create matrices
    dVolumes = {}
    dBVec = {}
    dCount = {}
    for key in dNumVolumes:
        volCount = dNumVolumes[key]
        imgShape = [img.shape[0], img.shape[1], img.shape[2], volCount]
        imgMat = np.zeros(imgShape, dtype = np.float32)
        dVolumes[key] = imgMat
        bvecMat = np.zeros([3, volCount], dtype = np.float32)
        dBVec[key] = bvecMat
        dCount[key] = 0

    # Assign values
    for b in range(numvols):
        if bval[b] == 0:
            continue
        key = bval[b]
        count = dCount[key]
        dVolumes[key][:, :, :, count] = img[:, :, :, b]
        dBVec[key][:, count] = bvec[:, b]
        dCount[key] += 1

    return dVolumes, dBVec, dCount

'''
Compute the average b=0 volume
'''
def ComputeAverageB0(img, bval):
    imgShape = img.shape
    avgVol = np.zeros(imgShape[0:3], dtype=np.float32)

    bCount = 0
    imgData = img.get_data()
    for b in range(len(bval)):
        if bval[b] == 0:
            avgVol = avgVol + imgData[:, :, :, b]
            bCount += 1

    avgVol /= float(bCount)

    return avgVol
'''
Extract b0 volumes
'''
def ExtractB0Volume(data, bval):
    delFlag = []
    for b in range(len(bval)):
        if bval[b] != 0:
            delFlag.append(b)

    #print 'Shape before extraction: ' + `data.shape`
    extData = np.delete(data, delFlag, axis=3)
    #print delFlag
    #print 'Shape after extraction: ' + `extData.shape`
    return extData
'''
Normalize DW volumes using average b=0 volumes
'''
def NormalizeDW(img, bval):
    print 'Start: DW image normalization'
    avgB0 = ComputeAverageB0(img, bval)

    # Normalize
    outimg = np.zeros(img.shape, dtype=np.float32)
    onevol = np.ones(img.shape[0:3])
    imgData = img.get_data()
    for b in range(len(bval)):
        if (bval[b] == 0):
            outimg[:, :, :, b] = onevol
        else:
            outimg[:, :, :, b] = np.divide(imgData[:, :, :, b], avgB0)

    # Check Nan and inf values
    outimg[np.isinf(outimg)] = 0.0
    outimg[np.isnan(outimg)] = 0.0

    # remove outliers
    outimg[outimg > 1.0] = 0.0

    print 'End: DW image normalization'
    return outimg

'''
Read bval and bvec files
bval = 1xQ vector, Q: number of b-values
bvec = 3xQ matrix
'''
def ReadBValAndBVec(bvalfile, bvecfile):
    # Read bval file
    bvalf = open(bvalfile, 'r')
    bvalstr = bvalf.readline()
    bvalarr = np.fromstring(bvalstr, dtype=int, sep=' ')
    bvalf.close()
    # Read bvec file
    bvecf = open(bvecfile, 'r')
    bveclines = bvecf.readlines()
    if len(bveclines) < 3:
        print 'bvec file is not completed.'
        return 0,0

    linecount = 0
    bvecmat = np.zeros([3, len(bvalarr)])
    for l in range(len(bveclines)):
        line = bveclines[l]
        if (len(line) == 0):
            continue
        bvecarr = np.fromstring(line, dtype=np.float32, sep=' ')
        if len(bvecarr) != len(bvalarr):
            print 'Length of bvec array is not same as that of bval array'
            return 0,0
        bvecmat[linecount, :] = bvecarr
        linecount += 1
        if (linecount == 3):
            break
    return bvalarr, bvecmat

'''
Convert bval to group value
'''
def ConvertBVal(bvalarr, offset = 20, scale = 1, div = 1000):
    bvalarr = bvalarr + offset
    bvalarr = bvalarr * scale
    bvalarr = bvalarr/div
    return bvalarr

'''
Downsample data
'''
def DownsampleImage(img, ds_factor):
    imgshape = img.shape
    ds_factor1 = ds_factor*1.0
    imageshape_ds = [int(np.ceil(imgshape[0]/ds_factor1)), int(np.ceil(imgshape[1]/ds_factor1)), int(np.ceil(imgshape[2]/ds_factor1))]
    # imageshape_ds = [int((imgshape[0]/ds_factor)), int((imgshape[1]/ds_factor)), int((imgshape[2]/ds_factor))]

    # print imageshape_ds
    output = np.zeros(imageshape_ds, dtype=np.float32)
    for i in range(imageshape_ds[0]):
        ii = ds_factor*i
        for j in range(imageshape_ds[1]):
            jj = ds_factor*j
            for k in range(imageshape_ds[2]):
                kk = ds_factor*k 
                # endi = np.min(ii+ds_factor, imgshape[0])               
                output[i,j,k] = np.mean(img[ii:ii+ds_factor,jj:jj+ds_factor,kk:kk+ds_factor])
    return output
'''
Decompose a output patch to a set of sub-patches
for sub-pixel convolutional neural network
E.g. 8x8x8x90 --> 4x4x4x90x8 (2x2x2) --> 4x4x2x90x16 (2x2x4)
'''
def DecomposeFullPatch(outPatches, HRfactor, ds_factor):
    oshape = outPatches.shape
    r = HRfactor
    dshape = [oshape[0], oshape[1], oshape[2]/r, oshape[3]/r, oshape[4]/r, (r**3)]
    print dshape
    # create array
    dPatches = np.zeros(dshape, dtype=np.float32)
    # outPatches = np.transpose(outPatches, (0,4,1,2,3))
    for i in range(0,r):
        for j in range(0,r):
            for k in range(0,r):
                c = i + j*r + k*r*r # x->y->z
                # c = i*HRfactor*HRfactor + j*HRfactor + k # z->y->x
                dPatches[:,:,:,:,:,c] = outPatches[:,:, i::r, j::r, k::r]
    dPatches = np.reshape(dPatches, [oshape[0], oshape[1], oshape[2]/r, oshape[3]/r, oshape[4]/r/ds_factor, (r**3)*ds_factor])
    # dPatches = np.transpose(outPatches, (0, 4, 1, 2, 3))
    # dPatches = np.reshape(dPatches, dshape)
    # dPatches = np.transpose(dPatches, (0, 2, 3, 4, 1, 5))
    return dPatches
    '''
Merge multiple feature maps to compose output patches of higher resolution (x2)
for sub-pixel convolutional neural network
E.g. 5x5x5x90x8 --> 10x10x10x90
Ref. https://github.com/tetrachrome/subpixel
'''
def TF_ShifePhase2(outPatches, HRfactor, ds_factor):
    # bsize, a, b, c, g, f = outPatches.get_shape().as_list() # batch size, width, height, depth, gradient volumes, number of feature maps (should be 8)
    [bsize, a, b, c, g, f] = outPatches.shape
    r = HRfactor
    # bsize = tf.shape(outPatches)[0] # Handling Dimension(None) type for undefined batch dim
    X = np.reshape(outPatches, (bsize, a, b, c, g, r*ds_factor, r, r))
    # print X.shape
    X = np.transpose(X, (0, 1, 2, 3, 4, 7, 6, 5))  #!!important!! bsize, a, b, c, g, r, r, r*ds_factor
    # ## x -> y -> z
    X = np.split(X, a, axis=1)  # a, [bsize, b, c, g, r, r, r*ds_factor]
    X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, b, c, g, a*r, r, r*ds_factor
    X = np.split(X, b, axis=1)  # b, [bsize, c, g, a*r, r, r*ds_factor]
    X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, c, g, a*r, b*r, r*ds_factor
    X = np.split(X, c, axis=1)  # c, [bsize, g, a*r, b*r, r*ds_factor]
    X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, g, a*r, b*r, c*r*ds_factor
    # X = np.reshape(X, (bsize, c, g, a*r, b*r, r, ds_factor))
    # X = np.split(X, c, axis=1) # c, [bsize, g, a*r, b*r, r, ds_factor]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, g, a*r, b*r, c*r, ds_factor
    # X = np.split(X, c*r, axis=4) # c*r, [bsize, g, a*r, b*r, ds_factor]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, g, a*r, b*r, c*r*ds_factor

    # ## z -> y -> x
    # X = np.reshape(outPatches, (bsize, a, b, c, g, r, r, r, ds_factor))
    # X = np.split(X, c, axis=3) # c, [bsize, a, b, g, r, r, r, ds_factor]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=7)  # bsize, a, b, g, r, r, r, c*ds_factor
    # X = np.split(X, c*ds_factor, axis=7) # c*ds_factor, [bsize, a, b, g, r, r, r]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=6)  # bsize, a, b, g, r, r, c*ds_factor*r
    # X = np.split(X, b, axis=2) # b, [bsize, a, g, r, r, c*ds_factor*r]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=4)  # bsize, a, g, r, b*r, c*ds_factor*r
    # X = np.split(X, a, axis=1) # a, [bsize, g, r, b*r, c*ds_factor*r]
    # X = np.concatenate([np.squeeze(x) for x in X], axis=2)  # bsize, g, a*r, b*r, c*ds_factor*r
    # ##
    X = np.reshape(X, (bsize, g, a*r, b*r, c*r*ds_factor))
    X = np.transpose(X, (0, 2, 3, 4, 1))
    return X