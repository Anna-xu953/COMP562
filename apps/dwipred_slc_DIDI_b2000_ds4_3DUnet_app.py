#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Wed May 24 09:59:10 2017

@author: jaeil , yoonmi

DCGAN-based Approach - Source: https://github.com/carpedm20/DCGAN-tensorflow
'''
import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random
import os.path
import math
import time
from sys import platform
import gc
import argparse
import sklearn
import scipy.sparse
import json
import cv2
from scipy.ndimage import zoom
import itertools

# import tensorflow & itk
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import SimpleITK as sitk
import h5py

# Caduceus

import caduceus.core.loadtrainingdata_slice_model15_DIDI as FuncLoad
import caduceus.core.imagepreprocessing as FuncImage
import caduceus.core.modelingrelated as Func
import caduceus.core.tools as Tools
import caduceus.models.EuclideanModels.Unet3D_resnet as Model

import caduceus.apps.learnable as Apps
import caduceus.core.dwimageprocessing as DWImage

# Training & Test Application Class 

class _LearnerImpl (Apps.Learnable) :
    _params={}
    _app_name=""
    AllIDs=[]
    TestingIDs=[]
    TrainingIDs=[]
    startW=0
    startH=0
    startD=0
    sIdx=0
    endW=0
    endH=0
    endD=0
    eIdx=0
    bval=[]
    bvec=[]
    def __init__(self,params):

        #define class variables
        self._params=params
        self._app_name=self._params["application"]["application_name"]

        self.AllIDs = self._params["etc"]["AllIDs"]["value"] #all data identifiers
        self.TestingIDs = self._params["etc"]["TestingIDs"]["value"]
        
        for i in range(len(self.AllIDs)):
            aID = self.AllIDs[i]
            if (aID in self.TestingIDs):
                continue
            else :
                self.TrainingIDs.append(aID)
        #self.log self.TrainingIDs
        
        self.NumSubjects = params["etc"]["NumSubjects"]["value"]

        self.startW = 3 +self._params["training_parameters"]["output_width"]["value"] /2
        self.startH = 3 + self._params["training_parameters"]["output_height"]["value"] /2
        self.startD = 0
        self.sIdx = [self.startW, self.startH, self.startD]
        self.endW = self.startW + self._params["training_parameters"]["input_width"]["value"] 
        self.endH = self.startH + self._params["training_parameters"]["input_height"]["value"] 
        self.endD = 70
        self.eIdx = [self.endW, self.endH, self.endD]

        self.bval = np.array(self._params["etc"]["bval"]["value"])
        # self.bval = DWImage.ConvertBVal(self.bval)
        self.bvec = np.array(self._params["etc"]["bvec"]["value"])

        self._initialize()

    def train(self,params=_params):
        self.log(self._app_name + " training starts")
        os.environ['CUDA_VISIBLE_DEVICES'] = self._params["training_parameters"]["GPU"]["value"]
        # TrainingAge('00', '03', TrainingIDs, TestingIDs, DoPretrain=self._params["training_parameters"]["DoPretrain"]["value"])
        self.TrainingAge(self._params["training_parameters"]["InAge"]["value"], self._params["training_parameters"]["OutAge"]["value"], self.TrainingIDs, self.TestingIDs, DoPretrain=self._params["training_parameters"]["DoPretrain"]["value"])
        self.log(self._app_name + " training finished")
    def test(self,params=_params):
        self.log(self._app_name + " test starts")
        os.environ['CUDA_VISIBLE_DEVICES'] = self._params["training_parameters"]["GPU"]["value"]
        self.TestAge(self._params["training_parameters"]["InAge"]["value"], self._params["training_parameters"]["OutAge"]["value"], self.TrainingIDs, self.TestingIDs, DoPretrain=self._params["training_parameters"]["DoPretrain"]["value"])
        self.log(self._app_name + " test finished")

    def MergeDWPatches(self, ResultPatches, ResultCheck, degree, refImg, refImgFile, imgDim):
        RotatedArray = []
        RotatedAddArray = []
        gCount = 0
        center = FuncImage.GetCenter(refImgFile)

        retPEnd = self._params["training_parameters"]["output_width"]["value"]

        oneArray = np.ones([self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]], dtype = np.float32)
        srcHalfDimOffset = [self._params["training_parameters"]["output_width"]["value"]%2, self._params["training_parameters"]["output_height"]["value"]%2, self._params["training_parameters"]["output_channels"]["value"]%2]

        srcHalfDim = [int(self._params["training_parameters"]["input_width"]["value"]/2), int(self._params["training_parameters"]["input_height"]["value"]/2), int(self._params["training_parameters"]["input_channels"]["value"]/2)]
        tarHalfDim = [int(self._params["training_parameters"]["output_width"]["value"]/2), int(self._params["training_parameters"]["output_height"]["value"]/2), int(self._params["training_parameters"]["output_channels"]["value"]/2)]
        for g in range(len(self.bval)):
            if (self.bval[g] == 2000):
                retAddArray = np.zeros(imgDim)
                retArray = np.zeros(imgDim, dtype = np.float32)
                pCount = 0
                i = tarHalfDim[2]#srcHalfDim[2]
                while i <= imgDim[2] - tarHalfDim[2] - 1:#imgDim[2] - srcHalfDim[2] - 1:
                    # if (pCount >= self._params["training_parameters"]["MaximumNumPatches"]["value"]):
                    #     break;
                    sI = i-tarHalfDim[2]
                    tI = i+tarHalfDim[2] + srcHalfDimOffset[2]
                    j = self.sIdx[0]
                    while j <= imgDim[0] - srcHalfDim[0] - 1:
                        # if (pCount >= self._params["training_parameters"]["MaximumNumPatches"]["value"]):
                        #     break;
                        sJ = j - tarHalfDim[0]
                        tJ = j + tarHalfDim[0] + srcHalfDimOffset[0]
                        k = self.sIdx[1]
                        while k <= imgDim[1] - srcHalfDim[1] - 1:
                            sK = k - tarHalfDim[1]
                            tK = k + tarHalfDim[1] + srcHalfDimOffset[1]
                            # if (pCount >= len(ResultCheck)):
                            #     break;
                            if (ResultCheck[pCount] == 1):
                                retArray[sJ:tJ, sK:tK, sI:tI] = retArray[sJ:tJ, sK:tK, sI:tI] + ResultPatches[pCount, :, :, gCount, :]
                                retAddArray[sJ:tJ, sK:tK, sI:tI] = retAddArray[sJ:tJ, sK:tK, sI:tI] + oneArray
                            pCount += 1
                            # Index update
                            k += self._params["training_parameters"]["patch_offset"]["value"]
                        # Index update
                        j += self._params["training_parameters"]["patch_offset"]["value"]
                    # index update
                    i += self._params["training_parameters"]["patch_offset"]["value"]

                # ItkImg
                retArray = retArray.transpose((2, 1, 0))
                retImg = sitk.GetImageFromArray(retArray)
                retImg.SetOrigin(refImg.GetOrigin())
                retImg.SetSpacing(refImg.GetSpacing())
                retImg.SetDirection(refImg.GetDirection())

                retAddArray = retAddArray.transpose((2, 1, 0))
                retAddImg = sitk.GetImageFromArray(retAddArray)
                retAddImg.SetOrigin(refImg.GetOrigin())
                retAddImg.SetSpacing(refImg.GetSpacing())
                retAddImg.SetDirection(refImg.GetDirection())

                # Rotate
                rotImg = FuncImage.RotateItkImg(retImg, degree, center, inverse=False)
                rotArray = sitk.GetArrayFromImage(rotImg)
                rotArray = rotArray.transpose((2, 1, 0))
                rotAddImg = FuncImage.RotateItkImg(retAddImg, degree, center, inverse=False)
                rotAddArray = sitk.GetArrayFromImage(rotAddImg)
                rotAddArray = rotAddArray.transpose((2, 1, 0))

                RotatedArray.append(rotArray)
                RotatedAddArray.append(rotAddArray)
                gCount += 1

                del retImg
                del rotImg
                del retAddImg
                del rotAddImg

                del rotArray
                del rotAddArray
                del retAddArray
                del retArray

        return RotatedArray, RotatedAddArray

    def TestDWImage(self,train_data, id, SourceTime, TargetTime, sIdx, eIdx, numSlice, ep, forward = True):
        td = train_data
        epoch = ep
        # Data Checking
        #b0File_Src = str(self._params["directories"]["image_path"]["value"]) + '/' + 'AffineB0' + '_' + id +'_' + SourceTime + '.nii.gz'
        b0File_Tar = str(self._params["directories"]["image_path"]["value"]) + '/' + id + '/' + 'target_0000.nii.gz'
        if not os.path.exists(b0File_Tar):
            self.log("File not exist: " + id)
            return

        #refFile = str(self._params["directories"]["image_path"]["value"]) + '/' + id + '/' + 'T2w_acpc_dc_restore_brain.nii.gz'
        refFile = b0File_Tar
        refImg = sitk.ReadImage(str(refFile))
        # srcSliceDim = [eIdx[0]-sIdx[0], eIdx[1]-sIdx[1], numSlice]
        # srcHalfDim = [srcSliceDim[0]/2, srcSliceDim[1]/2, numSlice]
        # srcSliceMargin = int(numSlice/2)

        refImgArray = sitk.GetArrayFromImage(refImg)
        refImgArray = refImgArray.transpose((2, 1, 0))
        imgDim = refImgArray.shape
        del refImgArray
        
        # mask generation from target file
        maskFile = str(self._params["directories"]["image_path"]["value"]) + '/' + id + '/' + 'mask.nii'
        maskImg = sitk.ReadImage(str(maskFile))
        tarImgArray = sitk.GetArrayFromImage(maskImg)
        tarImgArray = tarImgArray.transpose((2, 1, 0))
        tarDim = tarImgArray.shape
        maskArray = np.zeros(tarDim)
        maskArray[tarImgArray > 0.0] = 1.0
        maskArray = maskArray.transpose((2, 1, 0))
        del tarImgArray

        SaveDir = str(self._params["directories"]["test_dir"]["value"]) + '/' + SourceTime + '_' + TargetTime + '/'
        if not tf.gfile.Exists(SaveDir):
            tf.gfile.MakeDirs(SaveDir)

        ds_factor = self._params["training_parameters"]["ds_factor"]["value"]

        # Load Sampling Files
        # outFile_SrcFWD = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + id + '_SrcFWD' + SourceTime + '_' + TargetTime
        # outFile_TarFWD = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + id + '_TarFWD' + SourceTime + '_' + TargetTime

        angle1 = 45
        angle2 = 90
        degreeSet = [
            [0, 0, 0],
            # [math.radians(0), math.radians(angle2), math.radians(0)],
            # [math.radians(0), math.radians(0), math.radians(angle2)],
            # [math.radians(0), math.radians(-angle1), math.radians(0)],
        ]

        
        model = td.model
        #noB0Vol = np.count_nonzero(self.bval)
        nbval, nbvec = Tools.ExtractSingleShell(self.bval, self.bvec, 2000)
        noB0Vol = len(nbval)
        sumAddArray = np.zeros([imgDim[0], imgDim[1], imgDim[2], noB0Vol])
        sumArray = np.zeros([imgDim[0], imgDim[1], imgDim[2], noB0Vol], dtype = np.float32)

        for d in range(len(degreeSet)):
            
            if False: #debug
                DWSrcPatches_FWD, skipIdx = self.SamplingDWImage_Single(id, SourceTime, [degreeSet[d]], group_ind_a, group_ind_c, group_ind_s, 'source', skip = False)
                pLength = len(DWSrcPatches_FWD)
                ResultCheck = np.ones(pLength)
                ResultPatches = np.reshape(DWSrcPatches_FWD, [ noB0Vol, -1, 16, 16, 16])
                ResultPatches = np.transpose(ResultPatches, (1,2,3,0,4))
                RotatedArray, RotatedAddArray = self.MergeDWPatches(ResultPatches, ResultCheck, degreeSet[d], refImg, str(refFile), imgDim)
            else:
                DWSrcPatches_FWD, skipIdx = self.SamplingDWImage_Single(id, SourceTime, [degreeSet[d]], skip = False)
                # self.log (srcSliceMod.shape)
                DWSrcPatches_FWD = (DWSrcPatches_FWD - self._params["training_parameters"]["Mean"]["value"])
                DWSrcPatches_FWD = np.reshape(DWSrcPatches_FWD, [-1, self._params["training_parameters"]["input_width"]["value"], self._params["training_parameters"]["input_height"]["value"], self._params["training_parameters"]["input_channels"]["value"]])

                # Obtaining result samples
                
                BatchSetSrcFWD = np.zeros([self._params["training_parameters"]["batch_size"]["value"], self._params["training_parameters"]["input_width"]["value"], self._params["training_parameters"]["input_height"]["value"], self._params["training_parameters"]["input_channels"]["value"]])
                BatchIndex = np.zeros(self._params["training_parameters"]["batch_size"]["value"], dtype = np.int32)

                # DWSrcPatches_FWD = np.transpose(DWSrcPatches_FWD, [0, 1, 2, 4, 3])
                self.log (`np.count_nonzero(skipIdx)` +'/' + `len(skipIdx)`)

                pLength = len(DWSrcPatches_FWD)
                ResultPatchesArr = np.zeros([DWSrcPatches_FWD.shape[0], self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]])
                ResultCheck = np.zeros(DWSrcPatches_FWD.shape[0])

                totalValidPatch = len(skipIdx) - np.count_nonzero(skipIdx)
                tCount = 0
                bCount = 0
                for pIdx in range(pLength):
                    # if skipIdx[pIdx] == 1:
                    #     continue
                    if skipIdx[pIdx % (len(skipIdx))] == 1:
                        continue

                    BatchSetSrcFWD[bCount, :, :, :] = DWSrcPatches_FWD[pIdx, :, :, :]
                    BatchIndex[bCount] = pIdx
                    # BatchZpermIndex[bCount] = srcSliceMod[pIdx]
                    bCount += 1

                    if bCount < self._params["training_parameters"]["batch_size"]["value"] - 1:
                        continue

                    # Merge FA and DW volumes
                    oriM = BatchSetSrcFWD.size/self._params["training_parameters"]["batch_size"]["value"]
                    BatchSetSrcFWD_Arr = np.reshape(BatchSetSrcFWD, [self._params["training_parameters"]["batch_size"]["value"], -1])

                    feed_dict = {model.ph_data: BatchSetSrcFWD_Arr, model.ph_dropout: 1.0, model.ph_phase_train: False}
                    bResultArray_Arr = td.sess.run(model.op_results, feed_dict)

                    bResultArray = np.reshape(bResultArray_Arr, [self._params["training_parameters"]["batch_size"]["value"], self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]])
                
                    self.log (`tCount` + ' / ' + `totalValidPatch`)
                    tCount += 1

                    for b in range(bCount):
                        bIdx = BatchIndex[b]
                        ResultPatchesArr[bIdx, :, :, :] = bResultArray[b, :, :, :] + self._params["training_parameters"]["Mean"]["value"]
                        ResultCheck[bIdx] = 1
                    # self.log ('ResultMean: ' + `np.mean(bResultArray)`) 
                    del bResultArray
                    del bResultArray_Arr
                    bCount = 0
                    BatchSetSrcFWD.fill(0)
                    BatchIndex.fill(0)

                del DWSrcPatches_FWD
                del BatchSetSrcFWD
                del BatchIndex

                ResultPatchesArr[ResultPatchesArr > 1.0] = 1.0
                ResultPatchesArr[ResultPatchesArr < 0.0] = 0.0

                # # Rotate results
                # ResultPatches = np.transpose(ResultPatches, [0, 1, 2, 4, 3])
                # RotatedArray, RotatedAddArray = self.MergeDWPatches(ResultPatches, ResultCheck, degreeSet[d], refImg, str(refFile), imgDim)
                ResultPatches = np.reshape(ResultPatchesArr, [noB0Vol, -1, self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]])
                ResultPatches = np.transpose(ResultPatches, (1, 2, 3, 0, 4)) # p, x, y, q, z
                RotatedArray, RotatedAddArray = self.MergeDWPatches(ResultPatches, ResultCheck, degreeSet[d], refImg, str(refFile), imgDim)
            gCount = 0
            for g in range(len(self.bval)):
                if (self.bval[g] == 2000):
                    sumArray[:, :, :, gCount] = sumArray[:, :, :, gCount] + RotatedArray[gCount]
                    sumAddArray[:, :, :, gCount] = sumAddArray[:, :, :, gCount] + RotatedAddArray[gCount]
                    gCount += 1

            del RotatedArray
            del RotatedAddArray
            del ResultPatches
            del ResultCheck

        # Save results
        snrMean = 0
        mseMean = 0
        DWFiles = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '.txt'
        DWListFile = open(DWFiles, 'w')

        DWList = []
        # Initial B0        
        #retArray = np.ones(imgDim, dtype = np.float32)
        dwSrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + id +'/' + 'target_0000.nii.gz'
        retImg = sitk.ReadImage(str(dwSrcFile))
        B0Array = sitk.GetArrayFromImage(retImg)
        B0Array = B0Array.transpose((2, 1, 0))
        # # retArray = sitk.GetArrayFromImage(retImg)
        # # retArray = retArray.transpose((2, 1, 0))
        # # retImg = sitk.GetImageFromArray(retArray)
        # retImg.SetOrigin(refImg.GetOrigin())
        # retImg.SetSpacing(refImg.GetSpacing())
        # retImg.SetDirection(refImg.GetDirection())
        # #SaveName = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_B0_' + '.nii.gz'
        # # the following is ad-hoc. need to be updated for different gradients dataset
        # for g in range(0,7):
        #     gTag = str(g*7).zfill(2)
        #     SaveName = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_DW_'+ gTag + '.nii.gz'
        #     sitk.WriteImage(retImg, str(SaveName))        
        #     DWList.append(SaveName)
        #     DWListFile.write(SaveName + '\n')
        del retImg

        # DW image save
        # Compute average for overlapping patches
        sumArray = sumArray/sumAddArray
        sumArray = self.NanToZero(sumArray)
        #sumArray[sumArray < 0.02] = 0.0


        gCount = 0
        for g in range(len(self.bval)):
            if (self.bval[g] == 2000):
                gTag = str(g).zfill(4)
                # Accuracy validation
                retArray = sumArray[:, :, :, gCount]
                # # determine which group g belongs to 
                # for k in range(0, ds_factor):
                #     if (g in group_ind[k]):
                #         offset_slice = k
                #         break;
                dwSrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + id +'/'  + 'input_25perc_' + gTag + '.nii.gz'
                orgImg = sitk.ReadImage(str(dwSrcFile))
                orgArray = sitk.GetArrayFromImage(orgImg)
                orgArray = orgArray.transpose((2, 1, 0))
                # find nonzero slices
                idx_scan = []
                for slc in range(orgArray.shape[2]):
                    if np.count_nonzero(orgArray[:,:,slc]) > 0:
                        idx_scan.append(slc)
                
                
                # re-normalization
                #retArray = minvec[gCount] + retArray*(maxvec[gCount]-minvec[gCount])
                retArray = retArray*B0Array
            #    # replace result slices with source slices for given source location
            #     retArray[:,:,offset_slice::ds_factor] = orgArray[:,:,offset_slice::ds_factor]
                retArray[:,:,idx_scan] = orgArray[:,:,idx_scan]
                
                retArray = retArray.transpose((2, 1, 0))
                # masking
                retArray = retArray*maskArray
                #
                retImg = sitk.GetImageFromArray(retArray)
                retImg.SetOrigin(refImg.GetOrigin())
                retImg.SetSpacing(refImg.GetSpacing())
                retImg.SetDirection(refImg.GetDirection())
                gTag = str(g).zfill(4)
                SaveName = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_DW_' + gTag + '.nii.gz'
                DWList.append(SaveName)
                DWListFile.write(SaveName + '\n')
                sitk.WriteImage(retImg, str(SaveName))
                gCount += 1

                del retImg
                del retArray
                del orgImg
                del orgArray
            elif g == 0:
                #copy from source
                gTag = str(g).zfill(4)        
                dwSrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + id + '/' + 'target_'+ gTag + '.nii.gz'
                retImg = sitk.ReadImage(str(dwSrcFile))
                retImg = retImg * maskImg
                # retArray = sitk.GetArrayFromImage(retImg)
                # retArray = retArray.transpose((2, 1, 0))
                # retImg = sitk.GetImageFromArray(retArray)
                retImg.SetOrigin(refImg.GetOrigin())
                retImg.SetSpacing(refImg.GetSpacing())
                retImg.SetDirection(refImg.GetDirection())
                SaveName = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_DW_'+ gTag + '.nii.gz'
                sitk.WriteImage(retImg, str(SaveName))        
                DWList.append(SaveName)
                DWListFile.write(SaveName + '\n')
                del retImg

        del sumArray
        del sumAddArray
        del maskArray
        del B0Array
        DWListFile.close()

        with open(DWFiles, 'r') as f:
            DWList = f.readlines()
            DWList.sort()
        with open(DWFiles, 'w') as f:
            for l in range(len(DWList)):
                DWList[l] = DWList[l].replace('\n', '')
                f.write(DWList[l] + '\n') 


        # Merge volumes
        DWSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + '3DVolumeMerger ' + DWFiles + ' ' + DWSaveFile)

        # Remove 3D volumes
        for f in range(len(DWList)):
            os.remove(DWList[f])

        # Compute DTI and measures
        VecSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_Vec' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + '4DToVectorImageConverter ' + DWSaveFile + ' ' + VecSaveFile)

        #gradientTable = '/home/ymhong/data/gradients_nonB0.txt'
        #gradientTable = self._params["files"]["gradients_nonb0"]["value"]
        gradientTable = self._params["files"]["gradients_DIDI_2000"]["value"]
        DTSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_DTI' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIEstimator ' + VecSaveFile + ' ' + DTSaveFile + ' ' + gradientTable)

        # FA
        FASaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_FA' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIMeasures --anisotropy ' + DTSaveFile + ' ' + FASaveFile)
        MDSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_MD' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIMeasures --meandiffusivity ' + DTSaveFile + ' ' + MDSaveFile)
        ADSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_AD' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIMeasures --axialdiffusivity ' + DTSaveFile + ' ' + ADSaveFile)
        RDSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_RD' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIMeasures --radialdiffusivity ' + DTSaveFile + ' ' + RDSaveFile)
        FAWSaveFile = SaveDir + '/' + id + '_' + TargetTime + 'to' + SourceTime + '_' + `epoch` + '_Color' + '.nii.gz'
        os.system(self._params["directories"]["dwitk_path"]["value"] + 'DTIMeasures --pdrgb 1 --faweighting ' + DTSaveFile + ' ' + FAWSaveFile)

        os.remove(DTSaveFile)
        os.remove(VecSaveFile)

        del refImg
        return snrMean, mseMean


    def SamplingDWImage_Single(self,idSrc, SourceTime, degreeSet, mode = 'source', skip = True, dtype=np.float32):
        #start_time  = time.time()
        dwImages = []

        # Load B0 Image
        b0SrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc + '/' + 'target_0000.nii.gz'        
        #AtlasFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc + '/' + 'T2w_acpc_dc_restore_brain.nii.gz'
        AtlasFile = b0SrcFile
        maskFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc + '/' + 'mask.nii'

        ds_factor = self._params["training_parameters"]["ds_factor"]["value"]

        offset = [self._params["training_parameters"]["patch_offset"]["value"], self._params["training_parameters"]["patch_offset"]["value"], self._params["training_parameters"]["patch_offset"]["value"]]
        
        AtlasSlice = FuncLoad.load_trainingdata_fromOneImage_withRotation(AtlasFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=np.float32)
        # print ('AtlasSlice mean: ' + `np.mean(AtlasSlice)`)
        maskSlice = FuncLoad.load_trainingdata_fromOneImage_withRotation(maskFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=np.float32)
        AtlasSlice = AtlasSlice * maskSlice
        # Load FA Image
        # GFAFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc + '/' + idSrc + '_b2000_gFA.nii.gz'
        # FASlice,_ = FuncLoad.load_trainingdata_fromOneImage_withRotation(GFAFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"],  offset, False, skipZero = False, dtype=np.float32)

        if (mode == 'source'): #downsampling and interpolation
            start_time  = time.time()
            # offset = [self._params["training_parameters"]["patch_offset"]["value"], self._params["training_parameters"]["patch_offset"]["value"], self._params["training_parameters"]["patch_offset"]["value"]]
            # Load DW Image
            DWSrcImages = []
           
            gcount = 0
            for g in range(len(self.bval)):
                gTag = str(g).zfill(4)
                if self.bval[g] == 2000:
                    # # determine which group g belongs to 
                    # for k in range(0, ds_factor):
                    #     if (g in group_ind[k]):
                    #         offset_slice = k
                    #         break;

                    dwSrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc +'/' +  'input_25perc_' + gTag + '.nii.gz'

                    DWSrcSlice, B0SrcSlice, maskSlice = FuncLoad.load_trainingdata_fromOneImage_withDownsampling_Interpl(dwSrcFile, b0SrcFile, maskFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=dtype)
                    B0SrcSlice = B0SrcSlice * maskSlice
                    DWSrcSlice = DWSrcSlice.astype(np.float32)
                    # self.log('DW mean: ' + `np.mean(DWSrcSlice)`)
                    DWSrcSlice = DWSrcSlice/B0SrcSlice
                    DWSrcSlice = self.NanToZero(DWSrcSlice)
                    DWSrcSlice = DWSrcSlice * maskSlice
                    DWSrcSlice[DWSrcSlice > 1.0] = 1.0
                    DWSrcSlice[DWSrcSlice < 0.0] = 0.0
                    DWSrcImages.append(DWSrcSlice)
                    self.log('DW mean: ' + `np.mean(DWSrcSlice)`)
                    gcount += 1

            elapsed = time.time() - start_time
            self.log('DW Sampling: ' + `elapsed`)

        else: # no downsampling
            start_time  = time.time()            

            B0SrcSlice = FuncLoad.load_trainingdata_fromOneImage_withRotation(b0SrcFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=np.float32)
            maskSlice = FuncLoad.load_trainingdata_fromOneImage_withRotation(maskFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=np.float32)

            B0SrcSlice = B0SrcSlice * maskSlice
            # Load DW Image
            DWSrcImages = []
           
            gcount = 0
            for g in range(len(self.bval)):
                gTag = str(g).zfill(4)
                if self.bval[g] == 2000:
                    dwSrcFile = str(self._params["directories"]["image_path"]["value"]) + '/' + idSrc +'/' +  'target_' + gTag + '.nii.gz'

                    DWSrcSlice = FuncLoad.load_trainingdata_fromOneImage_withRotation(dwSrcFile, degreeSet, AtlasFile, self._params, self.sIdx, self.eIdx, self._params["training_parameters"]["output_channels"]["value"], offset, False, skipZero = False, dtype=dtype)
                    DWSrcSlice = DWSrcSlice.astype(np.float32)
                    
                    DWSrcSlice = DWSrcSlice/B0SrcSlice
                    DWSrcSlice = self.NanToZero(DWSrcSlice)
                    DWSrcSlice = DWSrcSlice * maskSlice
                    DWSrcSlice[DWSrcSlice > 1.0] = 1.0
                    DWSrcSlice[DWSrcSlice < 0.0] = 0.0
                    DWSrcImages.append(DWSrcSlice)
                    self.log('DW mean: ' + `np.mean(DWSrcSlice)`)
                    gcount += 1

            elapsed = time.time() - start_time
            self.log('DW Sampling: ' + `elapsed`)
            
        # DW reduction
        start_time  = time.time()

        DWSrcImages_Reduced = []

        gCount = 0
        skipIdx = np.ones(FuncLoad.MaximumNumPatches)
        for g in range(len(self.bval)):
            if self.bval[g] != 2000 :
                continue
            if (mode == 'source'): #downsampling
                DWSrcImage_Red = np.zeros([FuncLoad.MaximumNumPatches, self._params["training_parameters"]["input_width"]["value"], self._params["training_parameters"]["input_height"]["value"], self._params["training_parameters"]["input_channels"]["value"]])
            else:
                DWSrcImage_Red = np.zeros([FuncLoad.MaximumNumPatches, self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]])
            pCount = 0
            DWSrcImage = DWSrcImages[gCount]


            for p in range(DWSrcImages[gCount].shape[0]):
                aPatch = AtlasSlice[p, :, :, :]
                # FAPatch = FASlice[p,:,:,:]

                if skip == False:
                    DWSrcImage_Red[pCount, :, :, :] = DWSrcImage[p, :, :, :]
                    pCount += 1
                    if (np.count_nonzero(aPatch) > self._params["training_parameters"]["nonzeroratio"]["value"]*self._params["training_parameters"]["output_width"]["value"]*self._params["training_parameters"]["output_height"]["value"]*self._params["training_parameters"]["output_channels"]["value"] and gCount == 0):
                        skipIdx[p] = 0
                else:
                    if (np.count_nonzero(aPatch) > self._params["training_parameters"]["nonzeroratio"]["value"]*self._params["training_parameters"]["output_width"]["value"]*self._params["training_parameters"]["output_height"]["value"]*self._params["training_parameters"]["output_channels"]["value"]):
                        # if (np.mean(FAPatch) > 0.1):
                        DWSrcImage_Red[pCount, :, :, :] = DWSrcImage[p, :, :, :]
                        pCount += 1

            DWSrcImage_Red = DWSrcImage_Red[0:pCount, :, :, :] 

            if (gCount == 0):
                skipIdx = skipIdx[0:pCount]
            DWSrcImages_Reduced.append(DWSrcImage_Red)
            gCount += 1
            del DWSrcImage_Red

        del DWSrcImages
        elapsed = time.time() - start_time
        self.log('DW Reduction: ' + `elapsed`)

        start_time  = time.time()
        # DWSrcImages_Reduced = np.stack(DWSrcImages_Reduced, axis = 3)
        DWSrcImages_Reduced = np.concatenate(DWSrcImages_Reduced, axis = 0)
        elapsed = time.time() - start_time
        self.log('Stack: ' + `elapsed`)

        return DWSrcImages_Reduced, skipIdx

    def StackDWInput(self,AllDataSet, startIdx = 2):
        DWInputs = AllDataSet[startIdx:len(AllDataSet)]
        DWStack = tf.stack(DWInputs, axis = 1)
        return DWStack

    def NormalizeDW(self,DWList, B0):
        B0 = tf.cast(B0, tf.float32)
        for i in range(len(DWList)):
            DWList[i] = tf.cast(DWList[i], tf.float32)
            DWList[i] = tf.divide(DWList[i], B0)
            DWList[i] = tf.where(tf.is_nan(DWList[i]), tf.zeros_like(DWList[i]), DWList[i])
        return DWList

    class TrainData(object):
        def __init__(self, dictionary):
            self.__dict__.update(dictionary)

    def NanToZero(self,data):
        data[np.isinf(data)] = 0
        data[np.isnan(data)] = 0
        return data

    def Create_patches(self, idSrc, idTar, SourceTime, TargetTime, noB0Vol, degreeSet):
        outFile_SrcFWD = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + idSrc + '_FWD_DIDI'
        # outFile_SrcBWD = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + idTar + '_FWD' + TargetTime
        # outFile_SrcFWD_FA = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + idSrc + '_FWD_FA' + SourceTime
        # outFile_SrcBWD_FA = str(self._params["directories"]["checkpoint_dir"]["value"]) + '/' + idTar + '_FWD_FA' + TargetTime
  
        if (not os.path.exists(outFile_SrcFWD+'.h5')):
            self.log (outFile_SrcFWD+'.h5')
            DWSrcPatches_FWD, _ = self.SamplingDWImage_Single(idSrc, SourceTime, degreeSet, 'source')
            # DWSrcPatches_FWD = np.transpose(DWSrcPatches_FWD, (0, 1, 2, 4, 3))
            self.log('DWSrcPatches_FWD shape: ' + `DWSrcPatches_FWD.shape`)
            DWSrcPatches_FWD = np.reshape(DWSrcPatches_FWD, [-1, self._params["training_parameters"]["input_width"]["value"], self._params["training_parameters"]["input_height"]["value"], self._params["training_parameters"]["input_channels"]["value"]])
            self.log('DWSrcPatches_FWD shape: ' + `DWSrcPatches_FWD.shape`)
            DWSrcPatches_BWD,_ = self.SamplingDWImage_Single(idTar, TargetTime, degreeSet, 'target')
            # DWSrcPatches_BWD = np.transpose(DWSrcPatches_BWD, (0, 1, 2, 4, 3))
            DWSrcPatches_BWD = np.reshape(DWSrcPatches_BWD, [-1, self._params["training_parameters"]["output_width"]["value"], self._params["training_parameters"]["output_height"]["value"], self._params["training_parameters"]["output_channels"]["value"]])
            self.log('DWSrcPatches_BWD shape: ' + `DWSrcPatches_BWD.shape`)
            # # permutation of indices
            # numP = DWSrcPatches_FWD.shape[0]
            # DWSrcPatches_FWD_Arr = np.reshape(DWSrcPatches_FWD, [numP, -1])
            # DWSrcPatches_FWD_Arr = Coarsening.perm_data(DWSrcPatches_FWD_Arr, perm_indices)
            # numP = DWSrcPatches_BWD.shape[0]
            # # number of output channel = ds_factor
            # DWSrcPatches_BWD_Array = np.transpose(DWSrcPatches_BWD, (3, 0, 1, 2, 4)) 
            # ds_factor = self._params["training_parameters"]["ds_factor"]["value"]
            # DWSrcPatches_BWD_Array = np.reshape(DWSrcPatches_BWD_Array, [ds_factor, numP, -1])
            # DWSrcPatches_BWD_Arr = np.zeros([ds_factor, numP, len(perm_indices)], dtype=np.float32)
            # print(DWSrcPatches_BWD_Arr.shape)
            # for z in range(0, ds_factor):
            #     DWSrcPatches_BWD_Arr[z,:,:] = Coarsening.perm_data(DWSrcPatches_BWD_Array[z,:,:], perm_indices)
            with h5py.File(outFile_SrcFWD+'.h5','w') as hf:
                hf.create_dataset('DWSrcPatches_FWD', data = DWSrcPatches_FWD, dtype = np.float32)
                hf.create_dataset('DWSrcPatches_BWD', data = DWSrcPatches_BWD, dtype = np.float32)
            #     hf.create_dataset('DWSrcPatches_FWD_Arr', data = DWSrcPatches_FWD_Arr, dtype = np.float32)
            #     hf.create_dataset('DWSrcPatches_BWD_Arr', data = DWSrcPatches_BWD_Arr, dtype = np.float32)
            # del DWSrcPatches_BWD_Array
        else:
            with h5py.File(outFile_SrcFWD+'.h5', 'r') as hf:
                DWSrcPatches_FWD = hf['DWSrcPatches_FWD'][:]
                DWSrcPatches_BWD = hf['DWSrcPatches_BWD'][:]
                # DWSrcPatches_FWD_Arr = hf['DWSrcPatches_FWD_Arr'][:]
                # DWSrcPatches_BWD_Arr = hf['DWSrcPatches_BWD_Arr'][:]

        return DWSrcPatches_FWD, DWSrcPatches_BWD

    def TrainingAge(self,SourceTime, TargetTime, TrainingIDs, TestingIDs, DoPretrain = True, numFold = 1):
        # # Load IDs and Indices
        # trainIDs_Src, trainIDs_Tar = Func.check_IDs_Separate(self._params,self.TrainingIDs, SourceTime, TargetTime)
        # testIDs = Func.check_IDs(self._params,self.TestingIDs, SourceTime, TargetTime)
        # trainIDs_Src = self.TrainingIDs
        trainIDs_Src = self.AllIDs
        trainIDs_Tar = trainIDs_Src
        testIDs = self.TestingIDs
        self.log ('trainIDs_Src: ' + `trainIDs_Src`)
        self.log ('trainIDs_Tar: ' + `trainIDs_Tar`)
        self.log ('testIDs: ' + `testIDs`)

        # Compute Laplacian Matrix
        if not os.path.exists(self._params["directories"]["checkpoint_dir"]["value"]):
            os.makedirs(self._params["directories"]["checkpoint_dir"]["value"])

        #cbval = DWImage.ConvertBVal(self.bval)
        nbval, nbvec = Tools.ExtractSingleShell(self.bval, self.bvec, 2000)
        #self.log ('nbval: ' + `nbval`)
        #nbval, nbvec = Tools.RemoveB0Element_addB0(self.bval, self.bvec)
        ds_factor = self._params["training_parameters"]["ds_factor"]["value"]
        #self.log ('bvals: ' + `self.bval`)


        for i in range(0, numFold):
            # Setup global tf state
            tf.reset_default_graph()
            #tf.set_random_seed(int(time.time()))
            tf.set_random_seed(i+1)
            sess, summary_writer = Func.setup_tensorflow(self._params)

            # Get Non-b=0 volumes
            noB0Vol = len(nbval)

            # Model Generation
            model = Model.Unet3D(self._params, sess, 'Generator_FWD')

            # Load checkpoints, if they exist
            pretrain_epoch = self._params["training_parameters"]["pretrain_epoch"]["value"]
            saveParams = 'checkpoint_' + SourceTime + '_' + TargetTime + '_new'+`i`+'.txt-'+`pretrain_epoch`
            saveParams = os.path.join(self._params["directories"]["checkpoint_dir2"]["value"], saveParams)

            if not os.path.exists(self._params["directories"]["checkpoint_dir2"]["value"]):
                os.makedirs(self._params["directories"]["checkpoint_dir2"]["value"])
            if not os.path.exists(self._params["directories"]["pretrain_checkpoint_dir"]["value"]):
                os.makedirs(self._params["directories"]["pretrain_checkpoint_dir"]["value"])

            if not os.path.exists(saveParams + '.meta') or self._params["training_parameters"]["initialize"]["value"] == True:
                sess.run(tf.global_variables_initializer())
            else:
                self.log('Load saved variables')
                model.op_saver.restore(sess, saveParams)

            summary_writer.add_graph(sess.graph)
            sess.graph.finalize()
            self.log("Initialize Done")


            self.log('Start Training')

            # Training
            angle1 = 45
            angle2 = 90
            degreeSet = [
                [0, 0, 0],
            ]

            # index for trainIDs
            trSrcIdx = 0
            trTarIdx = 0

            bestLoss = 100.0
            earlyStopStep = 0
            checkEP = 0

            #np.random.seed(seed=int(time.time()))
            np.random.seed(seed = i+1)
            lenTrain_All = len(trainIDs_Src)
            if (lenTrain_All > len(trainIDs_Tar)):
                lenTrain_All = len(trainIDs_Tar)

            # while True:
            #     # valSub = np.random.randint(0, lenTrain_All)
            #     valSub = 0
            #     valSub_SrcID = trainIDs_Src[valSub]
            #     if (valSub_SrcID in trainIDs_Tar):
            #         valSub_TarID = valSub_SrcID
            #         break;

            # Select validation subject
            # trainIDs_Src.remove(valSub_SrcID)
            # trainIDs_Tar.remove(valSub_TarID)
            #valSub_SrcID = 'M001'
            valSub_SrcID = testIDs[0]
            valSub_TarID = valSub_SrcID
            
            self.log('Validation Subjects: ' + valSub_SrcID + ' ' + valSub_TarID)

            # determine minimum size of trainIDs (src, tar)
            lenTrain = len(trainIDs_Src)
            if (lenTrain > len(trainIDs_Tar) and DoPretrain == False):
                lenTrain = len(trainIDs_Tar)

            # for ep in range(self._params["training_parameters"]["epoch"]["value"]):
            ep = 0
            if (pretrain_epoch != 0):
                ep = pretrain_epoch
            ValidationLossAvg = 0.0
            ValidationBatchCount = 0.0

            
            while True:
                if ep > self._params["training_parameters"]["epoch"]["value"]:
                    break
                self.log('Epoch: ' + `ep` +'---------------------')
                batchCount = 0
                LossAvg = 0.0
                if (ep % self._params["training_parameters"]["eval_frequency"]["value"] == 0):
                    ValidationLossAvg = 0.0
                    ValidationBatchCount = 0

                for sub in range(0, lenTrain):
                    # while (1):
                        # np.random.seed(seed=int(time.time()))
                        # trSrcIdx_temp = np.random.randint(0, len(trainIDs_Src))
                        # if (trSrcIdx != trSrcIdx_temp):
                        #     trSrcIdx = trSrcIdx_temp
                        #     break;
                    # idSrc = trainIDs_temp[trSrcIdx]
                    idSrc = trainIDs_Src[trSrcIdx]
                    trSrcIdx += 1
                    if (trSrcIdx >= lenTrain):
                        trSrcIdx = 0
                    # paired
                    idTar = idSrc
                    if idTar not in trainIDs_Tar:
                        continue

                    # skip the following sampling and training if not eval_frequency for validation subject
                    if (idSrc == valSub_SrcID and ep % self._params["training_parameters"]["eval_frequency"]["value"] != 0):
                        continue

                    ##------------ step 1: Create patches ------------##
                    start_time  = time.time()
                    DWSrcPatches_FWD, DWSrcPatches_BWD = self.Create_patches(idSrc, idTar, SourceTime, TargetTime, noB0Vol, degreeSet)
                    numP = DWSrcPatches_FWD.shape[0]
                    DWSrcPatches_FWD_Arr = np.reshape(DWSrcPatches_FWD, [numP, -1])
                    DWSrcPatches_BWD_Arr = np.reshape(DWSrcPatches_BWD, [numP, -1])
                    elapsed = time.time() - start_time


                    self.log('Samping: ' + `elapsed`)
                    perM = DWSrcPatches_FWD_Arr.shape[-1]
                    BatchSetSrcFWD = np.zeros([self._params["training_parameters"]["batch_size"]["value"], perM])
                    perM = DWSrcPatches_BWD_Arr.shape[-1]
                    BatchSetSrcFWD_Target = np.zeros([self._params["training_parameters"]["batch_size"]["value"], perM])

                    ##------------ step 2: Shuffle the patches ------------##
                    pLength = len(DWSrcPatches_FWD)
                    if (pLength > len(DWSrcPatches_BWD)):
                        pLength = len(DWSrcPatches_BWD)
                    #np.random.seed(seed=int(time.time()))
                    #np.random.seed(seed=ep+sub+1)
                    np.random.seed(seed=ep)
                    pIndices = numpy.random.permutation(pLength)
                    # skipValue = np.random.randint(50, 200)
                    if idSrc == valSub_SrcID:
                        skipValue = 30
                    else:
                        skipValue = 50

                    tarS = int(self._params["training_parameters"]["input_width"]["value"]/2) - int(self._params["training_parameters"]["output_width"]["value"]/2)
                    tarE = tarS + self._params["training_parameters"]["output_width"]["value"]

                    ##------------ step 3: Training ------------##
                    self.log ('Start Training for ' + idSrc + ' and ' + idTar + ' skipValue: ' + `skipValue`)
                    bCount = 0

                    for pIdx in range(pLength):
                        #start_time  = time.time()
                        p = pIndices[pIdx]
                        SrcFWD = DWSrcPatches_FWD[p, :, :, :]
                        SrcBWD = DWSrcPatches_BWD[p, :, :, :]
                        threshold = self._params["training_parameters"]["nonzeroratio"]["value"]*self._params["training_parameters"]["output_height"]["value"]*self._params["training_parameters"]["output_width"]["value"]*self._params["training_parameters"]["output_channels"]["value"]
                        if (np.count_nonzero(SrcBWD) < threshold):
                            continue

                        if (pIdx % skipValue != 0):
                            continue

                        BatchSetSrcFWD[bCount, :] = DWSrcPatches_FWD_Arr[p, :]
                        BatchSetSrcFWD_Target[bCount, :] = DWSrcPatches_BWD_Arr[p, :]
                        bCount += 1

                        if bCount < self._params["training_parameters"]["batch_size"]["value"] - 1:
                            continue

                        oriM = BatchSetSrcFWD.size / self._params["training_parameters"]["batch_size"]["value"]

                        # validation
                        if idSrc == valSub_SrcID:
                            if ep % self._params["training_parameters"]["eval_frequency"]["value"] == 0:
                                feed_dict = {model.ph_data: BatchSetSrcFWD, model.ph_target: BatchSetSrcFWD_Target, model.ph_dropout: 1.0, model.ph_phase_train: False}
                                loss = sess.run(model.op_loss, feed_dict)
                                ValidationLossAvg += loss
                                ValidationBatchCount += 1
                        # Training
                        else:
                            feed_dict = {model.ph_data: BatchSetSrcFWD, model.ph_target: BatchSetSrcFWD_Target, model.ph_dropout: self._params["training_parameters"]["gene_drop"]["value"], model.ph_phase_train: True}
                            learning_rate, loss_average = sess.run([model.op_train, model.op_loss_average], feed_dict)                        
                            
                        # Init arrays
                        bCount = 0
                        BatchSetSrcFWD.fill(0)
                        BatchSetSrcFWD_Target.fill(0)

                    del BatchSetSrcFWD
                    del BatchSetSrcFWD_Target
                    del DWSrcPatches_FWD
                    del DWSrcPatches_BWD
                    del DWSrcPatches_FWD_Arr
                    del DWSrcPatches_BWD_Arr
                    self.log ('End Training for ' + idSrc + ' and ' + idTar + ' skipValue: ' + `skipValue`)

                self.log('learning_rate = {:.10}, loss_average = {:.10}'.format(learning_rate, loss_average))
                # self.log('loss_average = {:.10}'.format(loss_average))

                ##------------ Start validation and save models ------------##
                if ep % self._params["training_parameters"]["save_frequency"]["value"] == 0: 

                    path = os.path.join(self._params["directories"]["checkpoint_dir2"]["value"], 'checkpoint_' + SourceTime + '_' + TargetTime + '_new'+`i`+'.txt')
                    model.op_saver.save(sess, path, global_step = ep)

                    if ep % self._params["training_parameters"]["eval_frequency"]["value"] == 0:
                        if ValidationBatchCount != 0:
                            ValidationLossAvg /= float(ValidationBatchCount)
                            self.log ('Validation Loss: ep ' + `ep` + ' loss: ' + `ValidationLossAvg`)

                            summary = tf.Summary()
                            summary.ParseFromString(sess.run(model.op_summary, feed_dict))
                            summary.value.add(tag='validation/loss', simple_value=ValidationLossAvg)
                            summary_writer.add_summary(summary, ep)
                            summary_writer.flush()
                            del summary
                            # if bestLoss > ValidationLossAvg:
                            #     # Save
                            #     bestLoss = ValidationLossAvg
                            #     # train_data = TrainData(locals())
                            #     path = os.path.join(self._params["directories"]["checkpoint_dir2"]["value"], 'checkpoint_' + SourceTime + '_' + TargetTime + '_new'+`i`+'.txt')
                            #     model.op_saver.save(sess, path)

                            #     if ep != 0:
                            #         self.log ('Start testing at epoch: ' + `ep`)
                            #         for t in range(0, len(testIDs)):
                            #             id = testIDs[t]
                            #             train_data = self.TrainData(locals())
                            #             # Forward
                            #             #TestDWImage(train_data, id, SourceTime, TargetTime, sIdx, eIdx, self._params["training_parameters"]["input_channels"]["value"], perm_indices, forward = True, epoch = 0)
                            #             self.TestDWImage(train_data, id, SourceTime, TargetTime, self.sIdx, self.eIdx, self._params["training_parameters"]["input_channels"]["value"], perm_indices, ep, forward = True)
                            #             gc.collect()
                            #         self.log ('End testing at epoch: ' + `ep`)
                            #     # Func._save_checkpoint(train_data, i, ep, prefix = 'checkpoint_' + SourceTime + '_' + TargetTime + '_', preTrain = False)
                            #     # del train_data
                else:
                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(model.op_summary, feed_dict))
                    summary_writer.add_summary(summary, ep)
                    summary_writer.flush()
                    del summary

                # End of training

                ep += 1

            summary_writer.close()
            sess.close()


    def TestAge(self,SourceTime, TargetTime, TrainingIDs, TestingIDs, DoPretrain = True, numFold = 1):
        # Load IDs and Indices
        #testIDs = Func.check_IDs(self._params,self.TestingIDs, SourceTime, TargetTime)
        testIDs = self.TestingIDs
        self.log ('testIDs: ' + `testIDs`)

        # Compute Laplacian Matrix
        if not os.path.exists(self._params["directories"]["checkpoint_dir"]["value"]):
            os.makedirs(self._params["directories"]["checkpoint_dir"]["value"])

        nbval, nbvec = Tools.ExtractSingleShell(self.bval, self.bvec, 2000)
        ds_factor = self._params["training_parameters"]["ds_factor"]["value"]


        for i in range(0, numFold):
            # Setup global tf state
            tf.reset_default_graph()
            #tf.set_random_seed(int(time.time()))
            tf.set_random_seed(i+1)
            sess, summary_writer = Func.setup_tensorflow(self._params)

            # Get Non-b=0 volumes
            noB0Vol = len(nbval)

            # # Model Generation
            model = Model.Unet3D(self._params, sess, 'Generator_FWD')

            # Load checkpoints, if they exist
            eval_step = self._params["training_parameters"]["test_eval_step"]["value"]
            saveParams = 'checkpoint_' + SourceTime + '_' + TargetTime + '_new'+`i`+'.txt-'+`eval_step`
            saveParams = os.path.join(self._params["directories"]["checkpoint_dir2"]["value"], saveParams)

            if not os.path.exists(self._params["directories"]["checkpoint_dir2"]["value"]):
                os.makedirs(self._params["directories"]["checkpoint_dir2"]["value"])
            if not os.path.exists(self._params["directories"]["pretrain_checkpoint_dir"]["value"]):
                os.makedirs(self._params["directories"]["pretrain_checkpoint_dir"]["value"])


            self.log('Load saved variables')
            model.op_saver.restore(sess, saveParams)

            summary_writer.add_graph(sess.graph)
            sess.graph.finalize()
            self.log("Initialize Done")

            if not os.path.exists(self._params["directories"]["test_dir"]["value"]):
                os.makedirs(self._params["directories"]["test_dir"]["value"])

            for t in range(0, len(testIDs)):
                id = testIDs[t]
                train_data = self.TrainData(locals())
                # Forward
                #TestDWImage(train_data, id, SourceTime, TargetTime, sIdx, eIdx, self._params["training_parameters"]["input_channels"]["value"], perm_indices, forward = True, epoch = 0)
                self.TestDWImage(train_data, id, SourceTime, TargetTime, self.sIdx, self.eIdx, self._params["training_parameters"]["input_channels"]["value"], eval_step, forward = True)
            gc.collect()
            sess.close()
            self.log('Finish testing!!!')
            return