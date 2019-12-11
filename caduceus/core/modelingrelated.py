import os
import scipy.misc
import numpy as np

import random
import numpy.random
import os.path
import math
import time
from sys import platform
import gc

# import tensorflow
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import SimpleITK as sitk
import h5py
#import skimage.io as io

def setup_tensorflow(_params):
    #Create dirs
    if not os.path.exists(_params["directories"]["checkpoint_dir"]["value"]):
        os.makedirs(_params["directories"]["checkpoint_dir"]["value"])

    if not os.path.exists(_params["directories"]["test_dir"]["value"]):
        os.makedirs(_params["directories"]["test_dir"]["value"])

    # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ['TF_AUTOTUNE_THRESHOLD'] = '0'
    # os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '0'
    # Create session
    config = tf.ConfigProto(log_device_placement=_params["training_parameters"]["log_device_placement"]["value"], allow_soft_placement=True)
    # config = tf.ConfigProto(log_device_placement=_params["training_parameters"]["log_device_placement"]["value"], device_count={'GPU':0}, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)

    # config for linux (GPU server)
    if platform != "darwin":
        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(_params["directories"]["checkpoint_dir2"]["value"], sess.graph)

    return sess, summary_writer

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def WriteTFRecord(allSrcData, allSrcData2, allTarData, allTarData2, writer):
    for i in range(len(allSrcData)):
        srcData_raw = allSrcData[i].tostring()
        srcData2_raw = allSrcData2[i].tostring()
        tarData_raw = allTarData[i].tostring()
        tarData2_raw = allTarData2[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'srcData': _bytes_feature(srcData_raw),
            'srcData2': _bytes_feature(srcData2_raw),
            'tarData': _bytes_feature(tarData_raw),
            'tarData2': _bytes_feature(tarData2_raw)
            }))
        writer.write(example.SerializeToString())

# Write TF Record of two sets
def WriteTwoTFRecord(allSrcData, allTarData, refSrcData, refTarData, writer):
    eps = np.finfo(np.float32).eps
    numSlices = 0
    for i in range(len(allSrcData)):
        if (np.sum(refTarData[i]) < eps or np.sum(refSrcData[i]) < eps):
            continue
        srcData_raw = allSrcData[i].tostring()
        tarData_raw = allTarData[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'srcData': _bytes_feature(srcData_raw),
            'tarData': _bytes_feature(tarData_raw),
            }))
        writer.write(example.SerializeToString())
        numSlices = numSlices + 1
    return numSlices

def ReadTFRecord(_params,file, numEnque = 2, dtype = np.int16):
    filename_queue = tf.train.string_input_producer(file, shuffle = False, capacity = 50)
    options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=options)
    srcDataSet = []
    tarDataSet = []
    # for i in range(numEnque):
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'srcData': tf.FixedLenFeature([], tf.string),
        'tarData': tf.FixedLenFeature([], tf.string)
        }
    )
    srcData = tf.decode_raw(features['srcData'], dtype)
    tarData = tf.decode_raw(features['tarData'], dtype)
    srcDataShape = tf.stack([_params["training_parameters"]["input_width"]["value"], _params["training_parameters"]["input_height"]["value"], _params["training_parameters"]["input_channels"]["value"]])
    tarDataShape = tf.stack([_params["training_parameters"]["output_width"]["value"], _params["training_parameters"]["output_height"]["value"], _params["training_parameters"]["output_channels"]["value"]])
    srcData = tf.reshape(srcData, srcDataShape)
    tarData = tf.reshape(tarData, tarDataShape)
        #
        # srcDataSet.append(srcData)
        # tarDataSet.append(tarData)

    return srcData, tarData

# Write data
def WriteData(data, fPath, startIdx):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint16)
        op = tf.image.encode_png(data_t, compression=3)
        init = tf.initialize_all_variables()

    dShape = data.shape
    with tf.Session(graph=g) as sess:
        sess.run(init)
        for i in range(dShape[0]):
            data_np = sess.run(op, feed_dict={ data_t: data[i, :, :, :] })
            fName = fPath + str(i + startIdx).zfill(5) + '.png'
            with open(fName, 'w') as fd:
                fd.write(data_np)

def check_traindata(_params,trainIDs, SavePath, SourceTime, TargetTime):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(_params["directories"]["checkpoint_dir"]["value"]):
        tf.gfile.MakeDirs(_params["directories"]["checkpoint_dir"]["value"])

    # Return names of training files
    if not tf.gfile.Exists(_params["directories"]["sample_dir"]["value"]) or \
       not tf.gfile.IsDirectory(_params["directories"]["sample_dir"]["value"]):
        raise FileNotFoundError("Could not find folder `%s'" % (_params["directories"]["sample_dir"]["value"],))

    trainfiles = []
    retIDs = []
    for i in range(0, len(trainIDs)):
        id = trainIDs[i]
        filename = SavePath + 'Patches_' + id + '_' + SourceTime + '_' + TargetTime + '_.h5'
        if os.path.exists(filename):
            trainfiles.append(filename)
            retIDs.append(id)

    return trainfiles, retIDs

def check_testdata(_params,testIDs, SourceTime, TargetTime):
    # Return names of testing files
    if not tf.gfile.Exists(_params["directories"]["image_path"]["value"]) or \
       not tf.gfile.IsDirectory(_params["directories"]["image_path"]["value"]):
        raise FileNotFoundError("Could not find folder `%s'" % (_params["directories"]["image_path"]["value"],))

    testFiles = []
    testFiles2 = []
    targetFiles = []
    targetFiles2 = []

    for i in range(0, len(testIDs)):
        id = testIDs[i]
        filename = _params["directories"]["image_path"]["value"] + '/AffineB0_' + id + '_' + SourceTime + '.nii.gz'
        filename2 = _params["directories"]["image_path"]["value"] + '/AffineFA_' + id + '_' + SourceTime + '.nii.gz'
        tarFilename= _params["directories"]["image_path"]["value"] + '/NonLinearB0_' + id + '_' + TargetTime + 'to' +SourceTime + '.nii.gz'
        tarFilename2 = _params["directories"]["image_path"]["value"] + '/NonLinearFA_' + id + '_' + TargetTime + 'to' +SourceTime + '.nii.gz'
        if os.path.exists(filename):
            testFiles.append(filename)
            testFiles2.append(filename2)
            targetFiles.append(tarFilename)
            targetFiles2.append(tarFilename2)

    return testFiles, testFiles2, targetFiles, targetFiles2

def check_IDs_Separate(_params,ids, SourceTime, TargetTime):
    # Return names of testing files
    if not tf.gfile.Exists(_params["directories"]["image_path"]["value"]) or \
       not tf.gfile.IsDirectory(_params["directories"]["image_path"]["value"]):
        raise FileNotFoundError("Could not find folder `%s'" % (_params["directories"]["image_path"]["value"],))
        #raise IOError("Could not find folder `%s'" % (_params["directories"]["image_path"]["value"],))

    retIDs_Src = []
    retIDs_Tar = []
    for i in range(0, len(ids)):
        id = ids[i]
        tarFilename= _params["directories"]["image_path"]["value"] + '/AffineB0_' + id + '_' + TargetTime + '.nii.gz'
        srcFilename= _params["directories"]["image_path"]["value"] + '/AffineB0_' + id + '_' + SourceTime + '.nii.gz'
        if os.path.exists(tarFilename):
            retIDs_Tar.append(id)

        if os.path.exists(srcFilename):
            retIDs_Src.append(id)

    return retIDs_Src, retIDs_Tar

def check_IDs(_params,ids, SourceTime, TargetTime):
    # Return names of testing files
    if not tf.gfile.Exists(_params["directories"]["image_path"]["value"]) or \
       not tf.gfile.IsDirectory(_params["directories"]["image_path"]["value"]):
        raise FileNotFoundError("Could not find folder `%s'" % (_params["directories"]["image_path"]["value"],))

    retIDs = []
    for i in range(0, len(ids)):
        id = ids[i]
        tarFilename= _params["directories"]["image_path"]["value"] + '/AffineB0_' + id + '_' + TargetTime + '.nii.gz'
        tarFilename2= _params["directories"]["image_path"]["value"] + '/AffineB0_' + id + '_' + TargetTime + '.nii.gz'
        if os.path.exists(tarFilename) and os.path.exists(tarFilename2):
            retIDs.append(id)

    return retIDs

def _save_checkpoint(_params,train_data, fold, epoch, prefix = 'checkpoint_', preTrain=False):
    td = train_data

    oldname = prefix + 'old' +`fold`+'.txt'
    newname = prefix + 'new' +`fold`+'.txt'

    if (preTrain == False):
        oldname = os.path.join(_params["directories"]["checkpoint_dir2"]["value"], oldname)
        newname = os.path.join(_params["directories"]["checkpoint_dir2"]["value"], newname)
    else:
        oldname = os.path.join(_params["directories"]["pretrain_checkpoint_dir"]["value"], oldname)
        newname = os.path.join(_params["directories"]["pretrain_checkpoint_dir"]["value"], newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)
    del saver
    saver = 0
    print("Checkpoint saved")
