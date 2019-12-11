import h5py
import numpy as np
import imagepreprocessing as FuncImage
import random
import numpy.random
import math
import SimpleITK as sitk
from scipy.signal import resample_poly
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

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

    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    
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

def cropOneSlice_Patch_withDownsampling(_params,srcArray, B0Array, maskArray, sDim, tDim, srcNumSlice, offset, skipZero = False, dtype=np.uint16):
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

    srcPatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["input_channels"]["value"]], dtype=dtype)
    B0PatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["input_channels"]["value"]], dtype=dtype)
    maskPatchSet = np.zeros([estNum, srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["input_channels"]["value"]], dtype=dtype)

    tmpPatch = np.zeros([srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["output_channels"]["value"]], dtype=dtype)
    tmpB0Patch = np.zeros([srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["output_channels"]["value"]], dtype=dtype)
    tmpMaskPatch = np.zeros([srcSliceDim[0], srcSliceDim[1], _params["training_parameters"]["output_channels"]["value"]], dtype=dtype)
    srcSliceMargin = int(srcNumSlice/2.0)
    i = srcSliceMargin
    scanno = 0
    while i <= imgDim[2]-srcSliceMargin - 1:
        j = sDim[0]

        while j <= imgDim[0] - srcHalfDim[0] -1:
            sJ = j - srcHalfDim[0]
            tJ = j + srcHalfDim[0] + srcHalfDimOffset[0]
            k = sDim[1]
            while k <= imgDim[1] - srcHalfDim[1] -1:
                sK = k - srcHalfDim[1]
                tK = k + srcHalfDim[1] + srcHalfDimOffset[1]
                # for ss in range(i-srcSliceMargin,i+srcSliceMargin + srcHalfDimOffset[2]):
                #     if (ss % ds_factor == offset_slice):
                #         slice_loc = ss
                #         break; 
                # if (_params["training_parameters"]["input_channels"]["value"] == 1):
                #     srcPatch = np.expand_dims(srcArray[sJ:tJ, sK:tK, slice_loc], 2)
                # else:
                #     srcPatch = srcArray[sJ:tJ, sK:tK, slice_loc:i+srcSliceMargin + srcHalfDimOffset[2]:ds_factor]
                # #srcPatch = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin:i+srcSliceMargin + srcHalfDimOffset[2]]
                
                # Select first two nonzero slices

                ncount = 0
                for ss in range(i-srcSliceMargin,i+srcSliceMargin + srcHalfDimOffset[2]):
                    if (np.count_nonzero(srcArray[sJ:tJ, sK:tK, ss]) > 0):
                        tmpPatch[:,:,ncount] = srcArray[sJ:tJ, sK:tK, ss]
                        tmpB0Patch[:,:,ncount] = B0Array[sJ:tJ, sK:tK, ss]
                        tmpMaskPatch[:,:,ncount] = maskArray[sJ:tJ, sK:tK, ss]
                        ncount += 1
                        scanno += 1
                        # print ('scanned')
                    # if (ncount > _params["training_parameters"]["input_channels"]["value"]-1):
                    #     break

                # if ncount == 0 and _params["training_parameters"]["input_channels"]["value"] == 1: # select neighboring non-zero slice
                #     # if (np.count_nonzero(srcArray[sJ:tJ, sK:tK, i-srcSliceMargin-1]) > 0):
                #     if (i-srcSliceMargin > 0):
                #         if (i+srcSliceMargin + srcHalfDimOffset[2] <= imgDim[2]-srcSliceMargin - 1):
                #             tmpPatch[:,:,0] = (srcArray[sJ:tJ, sK:tK, i-srcSliceMargin-1]+srcArray[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]])/2.0
                #             tmpB0Patch[:,:,0] = (B0Array[sJ:tJ, sK:tK, i-srcSliceMargin-1]+B0Array[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]])/2.0
                #             tmpMaskPatch[:,:,0] = (maskArray[sJ:tJ, sK:tK, i-srcSliceMargin-1]+maskArray[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]])/2.0
                #         else:
                #             tmpPatch[:,:,0] = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin-1]
                #             tmpB0Patch[:,:,0] = B0Array[sJ:tJ, sK:tK, i-srcSliceMargin-1]
                #             tmpMaskPatch[:,:,0] = maskArray[sJ:tJ, sK:tK, i-srcSliceMargin-1]
                #     else:
                #         tmpPatch[:,:,0] = srcArray[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]]
                #         tmpB0Patch[:,:,0] = B0Array[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]]
                #         tmpMaskPatch[:,:,0] = maskArray[sJ:tJ, sK:tK, i+srcSliceMargin + srcHalfDimOffset[2]]


                # --------------------------------- 
                srcPatchSet[cubicCnt, :, :, :] = tmpPatch[:,:,0:_params["training_parameters"]["input_channels"]["value"]]
                B0PatchSet[cubicCnt, :, :, :] = tmpB0Patch[:,:,0:_params["training_parameters"]["input_channels"]["value"]]
                maskPatchSet[cubicCnt, :, :, :] = tmpMaskPatch[:,:,0:_params["training_parameters"]["input_channels"]["value"]]
                # srcPatchSet[cubicCnt, :, :, :] = srcArray[sJ:tJ, sK:tK, i-srcSliceMargin:i-srcSliceMargin+2]
                # B0PatchSet[cubicCnt, :, :, :] = B0Array[sJ:tJ, sK:tK, i-srcSliceMargin:i-srcSliceMargin+2]
                # maskPatchSet[cubicCnt, :, :, :] = maskArray[sJ:tJ, sK:tK, i-srcSliceMargin:i-srcSliceMargin+2]
                # print ('scanned slices: ' + `scanno`)
                # if (np.mean(srcPatchSet) > 0.0):
                #     print ('patch mean value: ' + `np.mean(srcPatchSet)`)
                cubicCnt += 1
                k += offset[1]
                # tmpPatch.fill(0)
                # tmpB0Patch.fill(0)
                # tmpMaskPatch.fill(0)
            # Index update
            j += offset[0]
        i += offset[2]

    srcPatchSet = srcPatchSet[0:cubicCnt, :, :, :]
    B0PatchSet = B0PatchSet[0:cubicCnt, :, :, :]
    maskPatchSet = maskPatchSet[0:cubicCnt, :, :, :]

    # print ('srcPatchSet mean: ' + `np.mean(srcPatchSet)`) 
    # print ("Extract slices: Done ---")
    return srcPatchSet, B0PatchSet, maskPatchSet

def load_trainingdata_fromOneImage_withRotation(srcFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset = [], img_aug = False, skipZero=False, SRCNN=False, dtype=np.uint16):
    # print ("Extract samples: Start ---")
    srcImageFile = str(srcFile)
    srcImg = sitk.ReadImage(str(srcFile))

    center = FuncImage.GetCenter(str(refFile))

    srcImg_Ori = srcImg

    if SRCNN == True:
        allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    else:
        allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["output_width"]["value"], _params["training_parameters"]["output_height"]["value"], _params["training_parameters"]["output_channels"]["value"]], dtype = dtype)
    if len(offset) == 0:
        offset = [int(_params["training_parameters"]["output_width"]["value"]/2), int(_params["training_parameters"]["output_height"]["value"]/2), int(_params["training_parameters"]["output_channels"]["value"]/2)]

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
                srcPatchSet, srcPatchMod = cropOneSlice_Patch(_params,srcArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)

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

def load_trainingdata_fromOneImage_withDownsampling_Interpl(srcFile, B0File, maskFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset = [], scan_dir = 'axial', triplanar = False, img_aug = False, skipZero=False, gaussiansmooth=False, dtype=np.uint16):
    # print ("Extract samples: Start ---")
    srcImageFile = str(srcFile)
    srcImg = sitk.ReadImage(str(srcFile))

    B0ImageFile = str(B0File)
    B0Img = sitk.ReadImage(str(B0ImageFile))

    maskImageFile = str(maskFile)
    maskImg = sitk.ReadImage(str(maskImageFile))

    center = FuncImage.GetCenter(str(refFile))

    srcImg_Ori = srcImg

    allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    allB0PatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    allMaskPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
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

                # find nonzero slices
                idx_scan = []
                ds_factor = _params["training_parameters"]["ds_factor"]["value"]
                for slc in range(orgArray.shape[2]):
                    if np.count_nonzero(orgArray[:,:,slc]) > 0:
                        idx_scan.append(slc)
                retArray_down = orgArray[:,:,idx_scan]                  
                ## using ndimage.zoom
                retArray_up = zoom(retArray_down, (1,1,ds_factor), order=1)

                retArray = retArray_up
                retArray[:,:,idx_scan] = orgArray[:,:,idx_scan]

                B0Array = sitk.GetArrayFromImage(B0Img)
                B0Array = B0Array.transpose((2, 1, 0))
                maskArray = sitk.GetArrayFromImage(maskImg)
                maskArray = maskArray.transpose((2, 1, 0))
                print ('srcArray mean: ' + `np.mean(retArray)`)

                # # # Downsampling in slices
                # # # zero-padding first
                # # if (srcArray.shape[2] % ds_factor != 0):
                # #     addslices = ds_factor - (srcArray.shape[2] % ds_factor)
                # #     zeroArray = np.zeros([srcArray.shape[0], srcArray.shape[1], addslices], dtype=dtype)
                # #     srcArray = np.concatenate([srcArray, zeroArray], axis = 2)
                # # #print srcArray.shape
                # # # bi-linear interpolation
                # noStep = int(srcArray.shape[2]/ds_factor) - 1
                # srcArray_up = srcArray
                # for k in range(0, noStep):
                #     for z in range(1, ds_factor):
                #         ind1 = k*ds_factor + offset_slice
                #         ind2 = (k+1)*ds_factor + offset_slice
                #         srcArray_up[:,:,ind1 + z] = (srcArray[:,:,ind1]*(ds_factor-z) + srcArray[:,:,ind2]*(z))/ds_factor

                # srcArray = srcArray_up[:,:,0:orgArray.shape[2]]


                # Extract Patches
                srcPatchSet = cropOneSlice_Patch(_params,retArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)
                B0PatchSet = cropOneSlice_Patch(_params,B0Array, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)
                maskPatchSet = cropOneSlice_Patch(_params,maskArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)


                endIdx = stdIdx + srcPatchSet.shape[0]
                allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
                allB0PatchSet[stdIdx:endIdx, :, : , :] = B0PatchSet
                allMaskPatchSet[stdIdx:endIdx, :, : , :] = maskPatchSet
                stdIdx = endIdx

                del srcPatchSet
                del B0PatchSet
                del maskPatchSet
                del srcArray
                del retArray
                del B0Array
                del maskArray
                del srcImg
                del B0Img
                del maskImg
                del orgArray
                #del srcArray_down
                # del srcArray_up

    allSrcPatchSet = allSrcPatchSet[0:stdIdx, :, :, :]
    allB0PatchSet = allB0PatchSet[0:stdIdx, :, :, :]
    allMaskPatchSet = allMaskPatchSet[0:stdIdx, :, :, :]

    del srcImg_Ori

    # print ("Extract samples: Done ---")
    return allSrcPatchSet, allB0PatchSet, allMaskPatchSet

def load_trainingdata_fromOneImage_withDownsampling_noInterpl(srcFile, B0File, maskFile, degreeSet, refFile, _params, sDim, tDim, srcNumSlice, offset = [], scan_dir = 'axial', triplanar = False, img_aug = False, skipZero=False, gaussiansmooth=False, dtype=np.uint16):
    # print ("Extract samples: Start ---")
    srcImageFile = str(srcFile)
    srcImg = sitk.ReadImage(str(srcFile))

    B0ImageFile = str(B0File)
    B0Img = sitk.ReadImage(str(B0ImageFile))

    maskImageFile = str(maskFile)
    maskImg = sitk.ReadImage(str(maskImageFile))

    center = FuncImage.GetCenter(str(refFile))

    srcImg_Ori = srcImg

    allSrcPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    allB0PatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
    allMaskPatchSet = np.zeros([_params["training_parameters"]["MaximumNumPatches"]["value"], _params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]], dtype = dtype)
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
            if (gaussiansmooth == True):
                srcArray = gaussian_filter(srcArray, sigma=1)
            B0Array = sitk.GetArrayFromImage(B0Img)
            B0Array = B0Array.transpose((2, 1, 0))
            maskArray = sitk.GetArrayFromImage(maskImg)
            maskArray = maskArray.transpose((2, 1, 0))
            print ('srcArray mean: ' + `np.mean(srcArray)`)
            # Extract Patches
            srcPatchSet, B0PatchSet, maskPatchSet = cropOneSlice_Patch_withDownsampling(_params,srcArray, B0Array, maskArray, sDim, tDim, srcNumSlice, offset, skipZero = skipZero, dtype=dtype)
            print ('srcPatchSet mean: ' + `np.mean(srcPatchSet)`)
            endIdx = stdIdx + srcPatchSet.shape[0]
            allSrcPatchSet[stdIdx:endIdx, :, : , :] = srcPatchSet
            allB0PatchSet[stdIdx:endIdx, :, : , :] = B0PatchSet
            allMaskPatchSet[stdIdx:endIdx, :, : , :] = maskPatchSet
            stdIdx = endIdx

            del srcPatchSet
            del B0PatchSet
            del maskPatchSet
            del srcArray
            del B0Array
            del maskArray
            del srcImg
            del B0Img
            del maskImg

    allSrcPatchSet = allSrcPatchSet[0:stdIdx, :, :, :]
    allB0PatchSet = allB0PatchSet[0:stdIdx, :, :, :]
    allMaskPatchSet = allMaskPatchSet[0:stdIdx, :, :, :]
    
    del srcImg_Ori

    # print ("Extract samples: Done ---")
    return allSrcPatchSet, allB0PatchSet, allMaskPatchSet