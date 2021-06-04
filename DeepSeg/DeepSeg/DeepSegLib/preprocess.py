from config import *
from nilearn.image import crop_img as crop_image

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

def preprocess_images(input_dir, preprocess_dir, images, dim=config["image_shape"]):
    for img in images:
        print("Preprocessing: ", img)

        # load the MRI imaging modalities (flair, t1, t1ce, t2)
        img_nifti = nib.load(os.path.join(input_dir, img)) #.get_fdata(dtype='float32')

        # crop the input image
        img_preprocess = crop_image(img_nifti)
    
        # convert into numpy array
        img_array = np.array(img_preprocess.get_fdata(dtype='float32'))

        # pad the preprocessed image
        padded_image = np.zeros((dim[0],dim[1],dim[2]))
        padded_image[:img_array.shape[0],:img_array.shape[1],:img_array.shape[2]] = img_array
        
        # save nifti images
        img_preprocess_nifti = nib.Nifti1Image(norm_image(padded_image), img_nifti.affine, img_nifti.header) 
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)
        nib.save(img_preprocess_nifti, os.path.join(preprocess_dir, img))

def main(input_dir, preprocess_dir, images):
    # preprocess the input MRI image(s) and save them as nifti data 
    preprocess_images(input_dir, preprocess_dir, images)

if __name__ == "__main__":
    main(input_dir=config['input_dir'], preprocess_dir=config['preprocess_dir'], images=config["images"])
