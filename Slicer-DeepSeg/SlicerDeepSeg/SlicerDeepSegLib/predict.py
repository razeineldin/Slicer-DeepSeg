#from models import *
# Model imports
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv3D, UpSampling3D, Activation, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add, SpatialDropout3D, Conv3DTranspose #old: Deconvolution3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow_addons.layers import InstanceNormalization
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import numpy as np

def get_whole_tumor_mask(data):
    return data > 0

def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

def get_enhancing_tumor_mask(data):
    return data == 4

def postprocess_tumor(seg_data, tumor_type = "all", output_shape = (240, 240, 155)):
    # post-process the enhancing tumor region
    seg_enhancing = (seg_data == 3)
    if np.sum(seg_enhancing) < 200:
        seg_data[seg_enhancing] = 1
    else:
        seg_data[seg_enhancing] = 4

    if tumor_type == "whole":
        seg_data = get_whole_tumor_mask(seg_data)
    elif tumor_type == "core":
        pred_seg_datadata = get_tumor_core_mask(seg_data)
    elif tumor_type == "enhancing":
        seg_data = get_enhancing_tumor_mask(seg_data)

    input_shape = np.array(seg_data.shape)
    output_shape = np.array(output_shape)
    offset = np.array((output_shape - input_shape)/2).astype(np.int)
    offset[offset<0] = 0
    x, y, z = offset

    # pad the preprocessed image
    padded_seg = np.zeros(output_shape).astype(np.ubyte)
    #padded_seg[x:x+seg_data.shape[0],y:y+seg_data.shape[1],z:z+seg_data.shape[2]] = seg_data[:,:,:padded_seg.shape[2]]
    padded_seg[x:x+seg_data.shape[0],y:y+seg_data.shape[1],z:z+seg_data.shape[2]] = seg_data[:,:,2:padded_seg.shape[2]+2]

    return padded_seg

def predict_segmentations(model, images, tumor_type = "all", output_shape = (240, 240, 155)):
    # load the MRI imaging modalities (flair, t1, t1ce, t2)
    img_arrays = np.array(images).astype(np.float)

    # predict 3-channel segmentation using the pre-trained model
    pred_data_3ch = np.squeeze(model.predict(img_arrays[np.newaxis, ...], steps=1))

    # convert into 1-channel segmentation
    pred_data = pred_data_3ch.argmax(axis=-1)

    # post-process the output segmentation
    pred_data = postprocess_tumor(pred_data, tumor_type, output_shape)

    return pred_data

