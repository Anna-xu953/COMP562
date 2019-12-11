import os
import scipy.misc
import numpy as np

import random
import numpy.random
import os.path
import math
import time
from sys import platform

# import tensorflow
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import SimpleITK as sitk
import h5py

def GetCenter(srcImageFile):
    srcImg = sitk.ReadImage(srcImageFile)

    thresholdFilter = sitk.BinaryThresholdImageFilter()
    thresholdFilter.SetLowerThreshold(0.0)
    thresholdFilter.SetUpperThreshold(10000000)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(1)

    thImg = thresholdFilter.Execute(srcImg)

    cal = sitk.LabelShapeStatisticsImageFilter()
    cal.Execute(thImg)
    center = cal.GetCentroid(1)

    del thresholdFilter
    del srcImg
    del thImg
    del cal

    return center

def RotateItkImg(srcImg, degree, center, inverse = False):
    # Transformation
    trans = sitk.Euler3DTransform()
    trans.SetCenter(center)
    trans.SetRotation(degree[0], degree[1], degree[2])

    if (inverse == True):
        trans = trans.GetInverse()

    # Resample Image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(srcImg)
    resample.SetTransform(trans)
    resample.SetInterpolator(sitk.sitkLinear)

    retImg = resample.Execute(srcImg)

    del resample
    del trans

    return retImg

def LoadImages(srcImageFile, srcImageFile2, tarImageFile, tarImageFile2):
    # Print filename
    print(srcImageFile)
    print(srcImageFile2)

    # Intensity scale
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)

    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetLower(0.0)
    thresholdFilter.SetUpper(10000000)
    thresholdFilter.SetOutsideValue(0.0)

    FAthresholdFilter = sitk.ThresholdImageFilter()
    FAthresholdFilter.SetLower(0.0)
    FAthresholdFilter.SetUpper(1.0)
    FAthresholdFilter.SetOutsideValue(0.0)

    # Src 1, 2
    srcImg = sitk.ReadImage(srcImageFile)
    srcImg = castFilter.Execute(srcImg)
    srcImg = thresholdFilter.Execute(srcImg)

    srcImg2 = sitk.ReadImage(srcImageFile2)
    srcImg2 = castFilter.Execute(srcImg2)
    srcImg2 = FAthresholdFilter.Execute(srcImg2)

    # Load target file
    tarImg = sitk.ReadImage(tarImageFile)
    tarImg = castFilter.Execute(tarImg)
    tarImg = thresholdFilter.Execute(tarImg)

    tarImg2 = sitk.ReadImage(tarImageFile2)
    tarImg2 = castFilter.Execute(tarImg2)
    tarImg2 = FAthresholdFilter.Execute(tarImg2)

    del castFilter
    del thresholdFilter
    del FAthresholdFilter

    return srcImg, srcImg2, tarImg, tarImg2

def ImagePreprocessing_ImageOnly(srcImageFile, srcImageFile2, tarImageFile, tarImageFile2):
    # Print filename
    print(srcImageFile)
    print(srcImageFile2)

    # Intensity scale
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)
    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetLower(0.0)
    thresholdFilter.SetUpper(10000000)
    thresholdFilter.SetOutsideValue(0.0)

    thresholdFilter2 = sitk.ThresholdImageFilter()
    thresholdFilter2.SetUpper(1500)
    thresholdFilter2.SetOutsideValue(1500.0)

    FAthresholdFilter = sitk.ThresholdImageFilter()
    FAthresholdFilter.SetLower(0.0)
    FAthresholdFilter.SetUpper(1.0)
    FAthresholdFilter.SetOutsideValue(0.0)

    # Src 1, 2
    srcImg = sitk.ReadImage(srcImageFile)
    srcImg = castFilter.Execute(srcImg)
    srcImg = thresholdFilter.Execute(srcImg)
    srcImg = thresholdFilter2.Execute(srcImg)

    srcImg2 = sitk.ReadImage(srcImageFile2)
    srcImg2 = castFilter.Execute(srcImg2)
    srcImg2 = FAthresholdFilter.Execute(srcImg2)

    # Load target file
    tarImg = sitk.ReadImage(tarImageFile)
    tarImg = castFilter.Execute(tarImg)
    tarImg = thresholdFilter.Execute(tarImg)
    tarImg = thresholdFilter2.Execute(tarImg)

    tarImg2 = sitk.ReadImage(tarImageFile2)
    tarImg2 = castFilter.Execute(tarImg2)
    tarImg2 = FAthresholdFilter.Execute(tarImg2)

    del castFilter
    del thresholdFilter
    del thresholdFilter2
    del FAthresholdFilter

    return srcImg, srcImg2, tarImg, tarImg2

def ImagePreprocessing_Slice_Ori(srcImageFile, srcImageFile2, tarImageFile, tarImageFile2):
    # Print filename
    print(srcImageFile)
    print(srcImageFile2)

    # Intensity scale
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)
    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetLower(0.0)
    thresholdFilter.SetUpper(10000000)
    thresholdFilter.SetOutsideValue(0.0)

    thresholdFilter2 = sitk.ThresholdImageFilter()
    thresholdFilter2.SetUpper(1500)
    thresholdFilter2.SetOutsideValue(1500.0)

    FAthresholdFilter = sitk.ThresholdImageFilter()
    FAthresholdFilter.SetLower(0.0)
    FAthresholdFilter.SetUpper(1.0)
    FAthresholdFilter.SetOutsideValue(0.0)

    # Src 1, 2
    srcImg = sitk.ReadImage(srcImageFile)
    srcImg = castFilter.Execute(srcImg)
    srcImg = thresholdFilter.Execute(srcImg)
    srcImg = thresholdFilter2.Execute(srcImg)

    srcImg2 = sitk.ReadImage(srcImageFile2)
    srcImg2 = castFilter.Execute(srcImg2)
    srcImg2 = FAthresholdFilter.Execute(srcImg2)

    # Perform testing patch by patch
    srcArray = sitk.GetArrayFromImage(srcImg)
    srcArray = srcArray.transpose((2, 1, 0))
    srcArray2 = sitk.GetArrayFromImage(srcImg2)
    srcArray2 = srcArray2.transpose((2, 1, 0))

    # Load target file
    tarImg = sitk.ReadImage(tarImageFile)
    tarImg = castFilter.Execute(tarImg)
    tarImg = thresholdFilter.Execute(tarImg)
    tarImg = thresholdFilter2.Execute(tarImg)
    tarArray = sitk.GetArrayFromImage(tarImg)

    tarImg2 = sitk.ReadImage(tarImageFile2)
    tarImg2 = castFilter.Execute(tarImg2)
    tarImg2 = FAthresholdFilter.Execute(tarImg2)
    tarArray2 = sitk.GetArrayFromImage(tarImg2)

    del tarImg
    del tarImg2
    del srcImg2
    del castFilter
    del thresholdFilter
    del thresholdFilter2
    del FAthresholdFilter

    return srcImg, srcArray, srcArray2, tarArray, tarArray2

def ImagePreprocessing_Slice(srcImageFile, srcImageFile2, tarImageFile, tarImageFile2):
    # Print filename
    print(srcImageFile)
    print(srcImageFile2)

    # Intensity scale
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)
    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetLower(0.0)
    thresholdFilter.SetUpper(10000000)
    thresholdFilter.SetOutsideValue(0.0)

    FAthresholdFilter = sitk.ThresholdImageFilter()
    FAthresholdFilter.SetLower(0.0)
    FAthresholdFilter.SetUpper(1.0)
    FAthresholdFilter.SetOutsideValue(0.0)

    scaleFilter = sitk.RescaleIntensityImageFilter()
    scaleFilter.SetOutputMinimum(0.0)
    scaleFilter.SetOutputMaximum(1.0)

    # Src 1, 2
    srcImg = sitk.ReadImage(srcImageFile)
    srcImg = castFilter.Execute(srcImg)
    srcImg = thresholdFilter.Execute(srcImg)
    srcImg = scaleFilter.Execute(srcImg)

    srcImg2 = sitk.ReadImage(srcImageFile2)
    srcImg2 = castFilter.Execute(srcImg2)
    srcImg2 = FAthresholdFilter.Execute(srcImg2)
    srcImg2 = scaleFilter.Execute(srcImg2)

    # Perform testing patch by patch
    srcArray = sitk.GetArrayFromImage(srcImg)
    srcArray = srcArray.transpose((2, 1, 0))
    srcArray2 = sitk.GetArrayFromImage(srcImg2)
    srcArray2 = srcArray2.transpose((2, 1, 0))

    # Load target file
    tarImg = sitk.ReadImage(tarImageFile)
    tarImg = castFilter.Execute(tarImg)
    tarImg = thresholdFilter.Execute(tarImg)
    tarArray = sitk.GetArrayFromImage(tarImg)

    tarImg2 = sitk.ReadImage(tarImageFile2)
    tarImg2 = castFilter.Execute(tarImg2)
    tarImg2 = FAthresholdFilter.Execute(tarImg2)
    tarArray2 = sitk.GetArrayFromImage(tarImg2)

    del tarImg
    del tarImg2
    del srcImg2
    del castFilter
    del thresholdFilter
    del scaleFilter
    del FAthresholdFilter

    return srcImg, srcArray, srcArray2, tarArray, tarArray2

def ImagePreprocessing(srcImageFile, srcImageFile2, tarImageFile):
    # Print filename
    print(srcImageFile)
    print(srcImageFile2)
    # Read src image
    srcImg = sitk.ReadImage(srcImageFile)
    # Intensity scale
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)
    thresholdFilter = sitk.ThresholdImageFilter()
    thresholdFilter.SetLower(0.0)
    thresholdFilter.SetUpper(10000000)
    thresholdFilter.SetOutsideValue(0.0)

    scaleFilter = sitk.RescaleIntensityImageFilter()
    scaleFilter.SetOutputMinimum(0.0)
    scaleFilter.SetOutputMaximum(1.0)

    srcImg = castFilter.Execute(srcImg)
    srcImg = thresholdFilter.Execute(srcImg)
    srcImg_Nor = scaleFilter.Execute(srcImg)

    srcImg2 = sitk.ReadImage(srcImageFile2)
    srcImg2 = castFilter.Execute(srcImg2)
    srcImg2 = thresholdFilter.Execute(srcImg2)
    srcImg2 = scaleFilter.Execute(srcImg2)

    # Perform testing patch by patch
    srcArray = sitk.GetArrayFromImage(srcImg)
    srcArray_Nor = sitk.GetArrayFromImage(srcImg_Nor)
    srcArray2 = sitk.GetArrayFromImage(srcImg2)

    # Load target file
    tarImg = sitk.ReadImage(tarImageFile)
    tarImg = castFilter.Execute(tarImg)
    tarImg = thresholdFilter.Execute(tarImg)

    tarArray = sitk.GetArrayFromImage(tarImg)

    del tarImg
    del srcImg2
    del castFilter
    del thresholdFilter
    del scaleFilter

    return srcImg, srcArray, srcArray_Nor, srcArray2, tarArray

def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                        output_prefix, output_suffix,
                       interpolator = sitk.sitkLinear, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
                given transformation_parameters. The reason we use this transformation as a parameter
                is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    all_images = [] # Used only for display purposes in this notebook.
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)        
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                    interpolator, default_intensity_value)
        # sitk.WriteImage(aug_image, output_prefix + '_' + 
        #                 '_'.join(str(param) for param in current_parameters) +'_.' + output_suffix)
            
        all_images.append(aug_image) # Used only for display purposes in this notebook.
    return all_images # Used only for display purposes in this notebook."

def augment_images_intensity(image_list, output_prefix = [], output_suffix = []):
    '''
    Generate intensity modified images from the originals.
    Args:
        image_list (iterable containing SimpleITK images): The images which we whose intensities we modify.
        output_prefix (string): output file name prefix (file name: output_prefixi_FilterName.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefixi_FilterName.output_suffix).
    '''

    # Create a list of intensity modifying filters, which we apply to the given images
    filter_list = []
    aug_image_lists = []
    
    # Smoothing filters
    
    # filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
    # filter_list[-1].SetSigma(2.0)
    
    # filter_list.append(sitk.DiscreteGaussianImageFilter())
    # filter_list[-1].SetVariance(4.0)
    
    bilateralFilter = sitk.BilateralImageFilter()
    bilateralFilter.SetDomainSigma(4.0)
    bilateralFilter.SetRangeSigma(8.0)
    aug_image_lists.append(bilateralFilter.Execute(image_list))
    # filter_list.append(bilateralFilter)
    # filter_list[-1].SetDomainSigma(4.0)
    # filter_list[-1].SetRangeSigma(8.0)

    medFilter = sitk.MedianImageFilter()
    medFilter.SetRadius(8)
    aug_image_lists.append(medFilter.Execute(image_list))
    # filter_list.append(sitk.MedianImageFilter())
    # filter_list[-1].SetRadius(8)
    
    # # Noise filters using default settings
    
    # # Filter control via SetMean, SetStandardDeviation.
    # filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

    # # Filter control via SetProbability
    # filter_list.append(sitk.SaltAndPepperNoiseImageFilter())
    
    # # Filter control via SetScale
    # filter_list.append(sitk.ShotNoiseImageFilter())
    
    # # Filter control via SetStandardDeviation
    # filter_list.append(sitk.SpeckleNoiseImageFilter())

    # filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    # filter_list[-1].SetAlpha(1.0)
    # filter_list[-1].SetBeta(0.0)

    # filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    # filter_list[-1].SetAlpha(0.0)
    # filter_list[-1].SetBeta(1.0)
    
    #aug_image_lists = [] # Used only for display purposes in this notebook.

    # for i,img in enumerate(image_list):
    #     aug_image_lists.append([f.Execute(img) for f in filter_list])            
        # for aug_image,f in zip(aug_image_lists[-1], filter_list):
        #     sitk.WriteImage(aug_image, output_prefix + str(i) + '_' +
        #                     f.GetName() + '.' + output_suffix)
    return aug_image_lists