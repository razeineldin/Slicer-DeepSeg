# preprocessing
import os
import numpy as np
import nibabel as nib
import tensorflow as tf

# utlity functions imports
import matplotlib.pyplot as plt

config = dict()

# define input
#config["image_shape"] = (192, 224, 160) # the input to the pre-trained model
config["image_shape"] = (160, 224, 192) # the input to the pre-trained model

config["input_dir"] = 'BraTS20_sample_case' # directory of the input image(s)
config["preprocess_dir"] = 'BraTS20_sample_case_preprocess' # directory of the pre-processed image(s)
config["predict_dir"] = 'BraTS20_sample_case_predict' # directory of the predicted segmentation
config["predict_name"] = 'BraTS20_sample_case_pred.nii.gz' # name of the predicted segmentation

#config["image_path"] = 'BraTS20_sample_case'
#config["image_path_preprocess"] = 'BraTS20_sample_case_preprocess'
#config['image_path_predict'] = os.path.join('BraTS20_sample_case_predict','BraTS20_sample_case_pred.nii.gz')

# define used MRI modalities 
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["image_modalities"] = config["all_modalities"]
# one variable for each MRI modality
config["image_1"] = 'BraTS20_sample_case_flair.nii.gz'
config["image_2"] = 'BraTS20_sample_case_t1.nii.gz'
config["image_3"] = 'BraTS20_sample_case_t1ce.nii.gz'
config["image_4"] = 'BraTS20_sample_case_t2.nii.gz'

# OR one variable for all MRI modalities
config["images"] = ['BraTS20_sample_case_flair.nii.gz', 'BraTS20_sample_case_t1.nii.gz', 
                     'BraTS20_sample_case_t1ce.nii.gz', 'BraTS20_sample_case_t2.nii.gz']

# model parameters
config["input_shape"] = (config["image_shape"][0], config["image_shape"][1], 
                         config["image_shape"][2], len(config["images"]))

config['model_path'] = os.path.join('weights', 'model-238.h5')
config['tumor_type'] = "all" # "all", "whole", "core", "enhancing"


# Tensorflow 2.XX\n",
if float(tf.__version__[:3]) >= 2.0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
#     print("Num GPUs Available: {len(gpus)}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
