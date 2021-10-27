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

def norm_image(img, norm_type = "norm"):
    if norm_type == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif norm_type == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif norm_type == "norm_slow": # different datasets
#         img = (img - np.min(img))/(np.max(img) - np.min(img))
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img) 
        img = (img - np.min(img))/img_ptp

    return img

def crop_image(img, output_shape=np.array((192, 224, 160))):
    # manual cropping to (160, 224, 192)
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - output_shape)/2).astype(np.int)
    offset[offset<0] = 0
    x, y, z = offset
    crop_img = img[x:x+output_shape[0], y:y+output_shape[1], z:z+output_shape[2]]

    # pad the preprocessed image
    padded_img = np.zeros(output_shape)
    x, y, z = np.array((output_shape - np.array(crop_img.shape))/2).astype(np.int)
    padded_img[x:x+crop_img.shape[0],y:y+crop_img.shape[1],z:z+crop_img.shape[2]] = crop_img

    return padded_img

def preprocess_images(imgs, dim):
    # TODO: automatic cropping using img[~np.all(img == 0, axis=1)]
    img_preprocess = np.zeros(dim)
    print("Shape img_preprocess", img_preprocess.shape)
    for i in range(dim[-1]):
      img_preprocess[:,:,:,i] = crop_image(imgs[:,:,:,i])
      img_preprocess[:,:,:,i] = norm_image(img_preprocess[:,:,:,i])

    return img_preprocess

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

