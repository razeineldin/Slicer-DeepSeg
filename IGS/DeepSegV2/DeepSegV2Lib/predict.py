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

# define input
#config["image_shape"] = (192, 224, 160) # the input to the pre-trained model
image_shape = (160, 224, 192) # the input to the pre-trained model
tumor_type = "all" # "all", "whole", "core", "enhancing"
predict_name = 'BraTS20_sample_case_pred.nii.gz' # name of the predicted segmentation

def get_whole_tumor_mask(data):
    return data > 0

def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

def get_enhancing_tumor_mask(data):
    return data == 4

def predict_segmentations(model, images):
    # load the MRI imaging modalities (flair, t1, t1ce, t2)
    img_arrays = np.array(images).astype(np.float)

    # predict 3-channel segmentation using the pre-trained model
    pred_data_3ch = np.squeeze(model.predict(img_arrays[np.newaxis, ...]))

    # convert into 1-channel segmentation
    pred_data = pred_data_3ch.argmax(axis=-1)
    pred_data[pred_data == 3] = 4

    if tumor_type == "whole":
        pred_data = get_whole_tumor_mask(pred_data)
    elif tumor_type == "core":
        pred_data = get_tumor_core_mask(pred_data)
    elif tumor_type == "enhancing":
        pred_data = get_enhancing_tumor_mask(pred_data)

    return pred_data

