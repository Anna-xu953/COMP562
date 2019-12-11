import h5py
import numpy as np
import imagepreprocessing as FuncImage
import random
import numpy.random
import math
import SimpleITK as sitk
import tensorflow as tf


def cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skip,  skipZero = False, dtype=np.uint16,):
    # print ("Extract slices ---")
    imgDim = srcArray.shape
    srcSliceDim = [tDim[0]-sDim[0], tDim[1]-sDim[1], srcNumSlice]
    srcHalfDim =  [int(srcSliceDim[0]/2.0), int(srcSliceDim[1]/2.0), int(srcSliceDim[2]/2.0)]
    srcHalfDimOffset = [srcSliceDim[0]%2, srcSliceDim[1]%2, srcNumSlice%2]

    # print("Image Dimension:", imgDim)
    estNum = _params["training_parameters"]["MaximumNumPatches"]["value"]
    # Extract patch from each diffusion-weighted volume
    eps = np.finfo(np.float32).eps

    cubicCnt = 0
    srcPatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], int(srcNumSlice)], dtype=dtype)

    # to reduce number of patches
    sidx_i = 12#int(imgDim[2]/2.0) - 50
    eidx_i = 118#int(imgDim[2]/2.0) + 50 - 1
    sidx_j = int(imgDim[0]/2.0) - 30
    eidx_j = int(imgDim[0]/2.0) + 30 - 1
    sidx_k = int(imgDim[1]/2.0) - 40
    eidx_k = int(imgDim[1]/2.0) + 40 - 1

    srcSliceMargin = int(srcNumSlice/2.0)
   
    if (skip == False):
        i = srcSliceMargin
        while i <= imgDim[2]-srcSliceMargin - 1:
            # if cubicCnt >= estNum:
            #     break;
            j = sDim[0]
            while j <= imgDim[0] - srcHalfDim[0] -1:
                # if cubicCnt >= estNum:
                #     break;
                sJ = j - srcHalfDim[0]
                tJ = j + srcHalfDim[0] + srcHalfDimOffset[0]
                k = sDim[1]
                while k <= imgDim[1] - srcHalfDim[1] -1:
                    # if cubicCnt >= estNum:
                    #     break;
                    sK = k - srcHalfDim[1]
                    tK = k + srcHalfDim[1] + srcHalfDimOffset[1]
                    srcPatch = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin:i+srcSliceMargin + srcHalfDimOffset[2]]
                    srcPatchSet[cubicCnt, :, :, :] = srcPatch
                    cubicCnt += 1
                    k += offset[1]
                # Index update
                j += offset[0]
            i += offset[2]
    else:
        i = sidx_i
        while i <= eidx_i:
            if cubicCnt >= estNum:
                break;
            j = sidx_j
            while j <= eidx_j:
                if cubicCnt >= estNum:
                    break;
                sJ = j - srcHalfDim[0]
                tJ = j + srcHalfDim[0] + srcHalfDimOffset[0]
                k = sidx_k
                while k <= eidx_k:
                    if cubicCnt >= estNum:
                        break;
                    sK = k - srcHalfDim[1]
                    tK = k + srcHalfDim[1] + srcHalfDimOffset[1]
                    srcPatch = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin:i+srcSliceMargin + srcHalfDimOffset[2]]
                    srcPatchSet[cubicCnt, :, :, :] = srcPatch
                    cubicCnt += 1
                    k += offset[1]
                # Index update
                j += offset[0]
            i += offset[2]

    srcPatchSet = srcPatchSet[0:cubicCnt, :, :, :]

    # print ("Extract slices: Done ---")
    return srcPatchSet

def load_trainingdata_fromOneImage_withRotation(srcFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, skip, offset = [], skipZero=False, dtype=np.uint16):
    # print ("Extract samples: Start ---")
    srcImageFile = str(srcFile)
    srcImg = sitk.ReadImage(str(srcFile))

    center = FuncImage.GetCenter(str(refFile))

    srcImg_Ori = srcImg

    allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    if len(offset) == 0:
        offset = [int(_params["training_parameters"]["input_width"]["value"]/2), int(_params["training_parameters"]["input_height"]["value"]/2), int(_params["training_parameters"]["input_channels"]["value"]/2)]

    stdIdx = 0
    for d in range(len(degreeSet)):
        # rotate Image
        # srcImg = FuncImage.RotateItkImg(srcImg_Ori, degreeSet[d], center, inverse=True)

        # Get array
        srcArray = sitk.GetArrayFromImage(srcImg)
        srcArray = srcArray.transpose((2, 1, 0))

        # Extract Patches
        srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skip, skipZero = skipZero, dtype=dtype)

        endIdx = stdIdx + srcPatchSet.shape[0]
        allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
        stdIdx = endIdx

        del srcPatchSet
        del srcArray
        del srcImg

    allSrcPatchSet = allSrcPatchSet[0:stdIdx, :, :, :]

    del srcImg_Ori

    # print ("Extract samples: Done ---")
    return allSrcPatchSet
