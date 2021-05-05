from config import *
from models import *

def get_whole_tumor_mask(data):
    return data > 0

def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)

def get_enhancing_tumor_mask(data):
    return data == 4

def predict_segmentation(model, preprocess_dir, predict_dir, images):
    # load the MRI imaging modalities (flair, t1, t1ce, t2)
    img_arrays = np.zeros(config["input_shape"])
    for i, img in enumerate(images):
        input_img = nib.load(os.path.join(preprocess_dir, img))
        img_arrays[:,:,:,i] = np.array(input_img.get_fdata()) # dtype='float32'))


    # predict 3-channel segmentation using the pre-trained model
    pred_data_3ch = np.squeeze(model.predict(img_arrays[np.newaxis, ...]))

    # convert into 1-channel segmentation
    pred_data = pred_data_3ch.argmax(axis=-1)
    pred_data[pred_data == 3] = 4

    if config['tumor_type'] == "whole":
        pred_data = get_whole_tumor_mask(pred_data)
    elif config['tumor_type'] == "core":
        pred_data = get_tumor_core_mask(pred_data)
    elif config['tumor_type'] == "enhancing":
        pred_data = get_enhancing_tumor_mask(pred_data)

    # save nifti images
    pred_image = nib.Nifti1Image(pred_data.astype(int), input_img.affine, input_img.header)
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    nib.save(pred_image, os.path.join(predict_dir, config["predict_name"]))

"""
def main(preprocess_dir, predict_dir, images):
    # create the residual U-Net model
    trained_model = get_model(input_shape=config["input_shape"])
    #trained_model.summary(line_length=120)

    # load the weights of the pre-trained model
    trained_model.load_weights(config['model_path'])#, by_name=True)

    # predict and save the tumor boundries
    predict_segmentation(trained_model, preprocess_dir, predict_dir, images)

if __name__ == "__main__":
    main(preprocess_dir=config['preprocess_dir'], predict_dir=config['predict_dir'], images=config["images"])
"""
