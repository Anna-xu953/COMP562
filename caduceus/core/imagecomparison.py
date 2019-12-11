import numpy as np
import os
import scipy.misc
import SimpleITK as sitk
import tensorflow as t


def PSNR(srcArray, tarArray):
    # print(srcArray.shape)
    # print(tarArray.shape)
    mse=np.sqrt(np.mean((srcArray-tarArray)**2))
    print 'mse ',mse
    max_I=np.max([np.max(srcArray),np.max(tarArray)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse), mse
