import h5py
import numpy as np
import imagepreprocessing as FuncImage
import random
import numpy.random
import math
import SimpleITK as sitk
from scipy.signal import resample_poly

MaximumNumPatches = 1000000

def cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = False, dtype=np.uint16):
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

    srcSliceMargin = int(srcNumSlice/2.0)
    i = srcSliceMargin
    while i <= imgDim[2]-srcSliceMargin - 1:
        j = sDim[0]
        while j <= imgDim[0] - srcHalfDim[0] -1:
            sJ = j - srcHalfDim[0]
            tJ = j + srcHalfDim[0] + srcHalfDimOffset[0]
            k = sDim[1]
            while k <= imgDim[1] - srcHalfDim[1] -1:
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

def cropOneSlice_Patch_withDownsampling(_params,srcArray, sDim, tDim, srcNumSlice, offset, offset_slice, skipZero = False, dtype=np.uint16):
    # print ("Extract slices ---")
    imgDim = srcArray.shape
    srcSliceDim = [tDim[0]-sDim[0], tDim[1]-sDim[1], srcNumSlice]
    srcHalfDim =  [int(srcSliceDim[0]/2.0), int(srcSliceDim[1]/2.0), int(srcSliceDim[2]/2.0)]
    srcHalfDimOffset = [srcSliceDim[0]%2, srcSliceDim[1]%2, srcNumSlice%2]

    # print("Image Dimension:", imgDim)
    estNum = _params["training_parameters"]["MaximumNumPatches"]["value"]
    # Extract patch from each diffusion-weighted volume
    eps = np.finfo(np.float32).eps

    ds_factor = _params["training_parameters"]["ds_factor"]["value"]

    cubicCnt = 0
    #srcPatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], int(srcNumSlice)], dtype=dtype)
    srcPatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], 1], dtype=dtype)

    srcSliceMargin = int(srcNumSlice/2.0)
    i = srcSliceMargin
    while i <= imgDim[2]-srcSliceMargin - 1:
        j = sDim[0]
        while j <= imgDim[0] - srcHalfDim[0] -1:
            sJ = j - srcHalfDim[0]
            tJ = j + srcHalfDim[0] + srcHalfDimOffset[0]
            k = sDim[1]
            while k <= imgDim[1] - srcHalfDim[1] -1:
                sK = k - srcHalfDim[1]
                tK = k + srcHalfDim[1] + srcHalfDimOffset[1]
                for ss in range(i-srcSliceMargin,i+srcSliceMargin + srcHalfDimOffset[2]):
                    if (ss % ds_factor == offset_slice):
                        slice_loc = ss
                        break; 
                srcPatch = np.expand_dims(srcArray[sJ:tJ, sK:tK, slice_loc], 2)
                #srcPatch = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin:i+srcSliceMargin + srcHalfDimOffset[2]]
                srcPatchSet[cubicCnt, :, :, :] = srcPatch
                cubicCnt += 1
                k += offset[1]
            # Index update
            j += offset[0]
        i += offset[2]

    srcPatchSet = srcPatchSet[0:cubicCnt, :, :, :]

    # print ("Extract slices: Done ---")
    return srcPatchSet

def load_trainingdata_fromOneImage_withRotation(srcFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset = [], img_aug = False, skipZero=False,  dtype=np.uint16):
    # print ("Extract samples: Start ---")
    srcImageFile = str(srcFile)
    srcImg = sitk.ReadImage(str(srcFile))

    center = FuncImage.GetCenter(str(refFile))

    srcImg_Ori = srcImg

    allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["output_channels"]["value"]], dtype = dtype)
    if len(offset) == 0:
        offset = [int(_params["training_parameters"]["input_width"]["value"]/2), int(_params["training_parameters"]["input_height"]["value"]/2), int(_params["training_parameters"]["input_channels"]["value"]/2)]

    stdIdx = 0
    for d in range(len(degreeSet)):
        # rotate Image
        # srcImg = FuncImage.RotateItkImg(srcImg_Ori, degreeSet[d], center, inverse=True)
        if(img_aug == True):
            # intensity augmentation by Histogram equalization
            srcImgList = FuncImage.augment_images_intensity(srcImg_Ori)
            for l in range(len(srcImgList)):
                srcImg = srcImgList[l]
                # Get array
                srcArray = sitk.GetArrayFromImage(srcImg)
                srcArray = srcArray.transpose((2, 1, 0))

                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del srcArray
                del srcImg
        else:
            # Get array
            srcArray = sitk.GetArrayFromImage(srcImg)
            srcArray = srcArray.transpose((2, 1, 0))

            # Extract Patches
            srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

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

def load_trainingdata_fromOneImage_withDownsampling_Interpl(srcFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset_slice, ds_factor,  offset = [], scan_dir = 'axial', triplanar = False, img_aug = False, skipZero=False,  dtype=np.uint16):
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
        if(img_aug == True):
            # intensity augmentation by Histogram equalization
            srcImgList = FuncImage.augment_images_intensity(srcImg_Ori)
            for l in range(len(srcImgList)):
                srcImg = srcImgList[l]
                # Get array
                srcArray = sitk.GetArrayFromImage(srcImg)
                srcArray = srcArray.transpose((2, 1, 0))

                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del srcArray
                del srcImg
        else:
            if (triplanar):
                # Get array
                srcArray = sitk.GetArrayFromImage(srcImg)
                srcArray = srcArray.transpose((2, 1, 0))
                # axial
                orgArray = srcArray
                # Downsampling in slices
                # zero-padding first
                if (srcArray.shape[2] % ds_factor != 0):
                    addslices = ds_factor - (srcArray.shape[2] % ds_factor)
                    zeroArray = np.zeros([srcArray.shape[0], srcArray.shape[1], addslices], dtype=dtype)
                    srcArray = np.concatenate([srcArray, zeroArray], axis = 2)
                #print srcArray.shape
                # srcArray_down = srcArray[:,:,offset_slice::ds_factor]
                # print srcArray_down.shape
                # Interpolation in slices
                #srcArray_up = resample_poly(srcArray_down, ds_factor, 1, axis = 2)
                # bi-linear interpolation
                noStep = int(srcArray.shape[2]/ds_factor) - 1
                srcArray_up = srcArray
                for k in range(0, noStep):
                    for z in range(1, ds_factor):
                        ind1 = k*ds_factor + offset_slice
                        ind2 = (k+1)*ds_factor + offset_slice
                        srcArray_up[:,:,ind1 + z] = (srcArray[:,:,ind1]*(ds_factor-z) + srcArray[:,:,ind2]*(z))/ds_factor
                #if (srcArray.shape[2] % ds_factor != 0):
                srcArray_axial = srcArray_up[:,:,0:orgArray.shape[2]]
                # sagittal
                srcArray = sitk.GetArrayFromImage(srcImg) # z, y, x
                orgArray = srcArray
                # Downsampling in slices
                # zero-padding first
                if (srcArray.shape[2] % ds_factor != 0):
                    addslices = ds_factor - (srcArray.shape[2] % ds_factor)
                    zeroArray = np.zeros([srcArray.shape[0], srcArray.shape[1], addslices], dtype=dtype)
                    srcArray = np.concatenate([srcArray, zeroArray], axis = 2)
                #print srcArray.shape
                # srcArray_down = srcArray[:,:,offset_slice::ds_factor]
                # print srcArray_down.shape
                # Interpolation in slices
                #srcArray_up = resample_poly(srcArray_down, ds_factor, 1, axis = 2)
                # bi-linear interpolation
                noStep = int(srcArray.shape[2]/ds_factor) - 1
                srcArray_up = srcArray
                for k in range(0, noStep):
                    for z in range(1, ds_factor):
                        ind1 = k*ds_factor + offset_slice
                        ind2 = (k+1)*ds_factor + offset_slice
                        srcArray_up[:,:,ind1 + z] = (srcArray[:,:,ind1]*(ds_factor-z) + srcArray[:,:,ind2]*(z))/ds_factor
                #if (srcArray.shape[2] % ds_factor != 0):
                srcArray_sag = srcArray_up[:,:,0:orgArray.shape[2]]
                srcArray_sag = srcArray_sag.transpose((2,1,0))
                # coronal
                srcArray = sitk.GetArrayFromImage(srcImg) # z, y, x
                srcArray = srcArray.transpose((2, 0, 1)) # x, z, y
                orgArray = srcArray
                # Downsampling in slices
                # zero-padding first
                if (srcArray.shape[2] % ds_factor != 0):
                    addslices = ds_factor - (srcArray.shape[2] % ds_factor)
                    zeroArray = np.zeros([srcArray.shape[0], srcArray.shape[1], addslices], dtype=dtype)
                    srcArray = np.concatenate([srcArray, zeroArray], axis = 2)
                #print srcArray.shape
                # srcArray_down = srcArray[:,:,offset_slice::ds_factor]
                # print srcArray_down.shape
                # Interpolation in slices
                #srcArray_up = resample_poly(srcArray_down, ds_factor, 1, axis = 2)
                # bi-linear interpolation
                noStep = int(srcArray.shape[2]/ds_factor) - 1
                srcArray_up = srcArray
                for k in range(0, noStep):
                    for z in range(1, ds_factor):
                        ind1 = k*ds_factor + offset_slice
                        ind2 = (k+1)*ds_factor + offset_slice
                        srcArray_up[:,:,ind1 + z] = (srcArray[:,:,ind1]*(ds_factor-z) + srcArray[:,:,ind2]*(z))/ds_factor
                #if (srcArray.shape[2] % ds_factor != 0):
                srcArray_cor = srcArray_up[:,:,0:orgArray.shape[2]]
                srcArray_cor = srcArray_cor.transpose((0, 2, 1))

                srcArray = np.mean(srcArray_axial, srcArray_sag, srcArray_cor)
                del srcArray_axial
                del srcArray_sag
                del srcArray_cor
                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del srcArray
                del srcImg
                del orgArray
                #del srcArray_down
                del srcArray_up
            else:
                # Get array
                srcArray = sitk.GetArrayFromImage(srcImg)
                srcArray = srcArray.transpose((2, 1, 0))
                orgArray = srcArray
                if (scan_dir == 'axial'):
                    # Downsampling in slices
                    # zero-padding first
                    if (srcArray.shape[2] % ds_factor != 0):
                        addslices = ds_factor - (srcArray.shape[2] % ds_factor)
                        zeroArray = np.zeros([srcArray.shape[0], srcArray.shape[1], addslices], dtype=dtype)
                        srcArray = np.concatenate([srcArray, zeroArray], axis = 2)
                    #print srcArray.shape
                    # bi-linear interpolation
                    noStep = int(srcArray.shape[2]/ds_factor) - 1
                    srcArray_up = srcArray
                    for k in range(0, noStep):
                        for z in range(1, ds_factor):
                            ind1 = k*ds_factor + offset_slice
                            ind2 = (k+1)*ds_factor + offset_slice
                            srcArray_up[:,:,ind1 + z] = (srcArray[:,:,ind1]*(ds_factor-z) + srcArray[:,:,ind2]*(z))/ds_factor

                    srcArray = srcArray_up[:,:,0:orgArray.shape[2]]
                elif (scan_dir == 'coronal'):
                    # Downsampling in slices
                    # zero-padding first
                    if (srcArray.shape[1] % ds_factor != 0):
                        addslices = ds_factor - (srcArray.shape[1] % ds_factor)
                        zeroArray = np.zeros([srcArray.shape[0], addslices, srcArray.shape[2]], dtype=dtype)
                        srcArray = np.concatenate([srcArray, zeroArray], axis = 1)
                    #print srcArray.shape
                    # bi-linear interpolation
                    noStep = int(srcArray.shape[1]/ds_factor) - 1
                    srcArray_up = srcArray
                    for k in range(0, noStep):
                        for y in range(1, ds_factor):
                            ind1 = k*ds_factor + offset_slice
                            ind2 = (k+1)*ds_factor + offset_slice
                            srcArray_up[:,ind1 + y,:] = (srcArray[:,ind1,:]*(ds_factor-y) + srcArray[:,ind2,:]*(y))/ds_factor

                    srcArray = srcArray_up[:,0:orgArray.shape[1],:]
                else: # scan_dir == 'sagittal'
                    # Downsampling in slices
                    # zero-padding first
                    if (srcArray.shape[0] % ds_factor != 0):
                        addslices = ds_factor - (srcArray.shape[0] % ds_factor)
                        zeroArray = np.zeros([addslices, srcArray.shape[1], srcArray.shape[2]], dtype=dtype)
                        srcArray = np.concatenate([srcArray, zeroArray], axis = 0)
                    #print srcArray.shape
                    # bi-linear interpolation
                    noStep = int(srcArray.shape[0]/ds_factor) - 1
                    srcArray_up = srcArray
                    for k in range(0, noStep):
                        for x in range(1, ds_factor):
                            ind1 = k*ds_factor + offset_slice
                            ind2 = (k+1)*ds_factor + offset_slice
                            srcArray_up[ind1 + x,:,:] = (srcArray[ind1,:,:]*(ds_factor-x) + srcArray[ind2,:,:]*(x))/ds_factor

                    srcArray = srcArray_up[0:orgArray.shape[0],:,:]

                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del srcArray
                del srcImg
                del orgArray
                #del srcArray_down
                del srcArray_up

    allSrcPatchSet = allSrcPatchSet[0:stdIdx, :, :, :]

    del srcImg_Ori

    # print ("Extract samples: Done ---")
    return allSrcPatchSet

def load_trainingdata_fromOneImage_withDownsampling_noInterpl(srcFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset_slice, offset = [], scan_dir = 'axial', triplanar = False, img_aug = False, skipZero=False,  dtype=np.uint16):
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
        if(img_aug == True):
            # intensity augmentation by Histogram equalization
            srcImgList = FuncImage.augment_images_intensity(srcImg_Ori)
            for l in range(len(srcImgList)):
                srcImg = srcImgList[l]
                # Get array
                srcArray = sitk.GetArrayFromImage(srcImg)
                srcArray = srcArray.transpose((2, 1, 0))

                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del srcArray
                del srcImg
        else:
            # Get array
            srcArray = sitk.GetArrayFromImage(srcImg)
            srcArray = srcArray.transpose((2, 1, 0))

            # Extract Patches
            srcPatchSet = cropOneSlice_Patch_withDownsampling(_params,srcArray, sDim, tDim, srcNumSlice, offset, offset_slice, skipZero = skipZero, dtype=dtype)

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