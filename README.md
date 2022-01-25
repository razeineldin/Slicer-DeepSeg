# Slicer-DeepSeg: Open-Source Deep Learning Toolkit for Brain Tumour Segmentation

## Purpose
Computerized medical imaging processing assists neurosurgeons to localize tumours precisely. It plays a key role in recent image-guided neurosurgery. Hence, we developed a new open-source toolkit, namely Slicer-DeepSeg, for efficient and automatic brain tumour segmentation based on deep learning methodologies for aiding clinical brain research. 

## Methods
Our developed toolkit consists of three main components. First, Slicer-DeepSeg extends the 3D Slicer application and thus provides support for multiple data input/ output data formats and 3D visualization libraries. Second, Slicer core modules offer powerful image processing and analysis utilities. Third, the Slicer-DeepSeg extension provides a customized GUI for brain tumour segmentation using deep learning-based methods. 

## Results
The developed Slicer-DeepSeg was validated using a public dataset of high-grade glioma patients. The results showed that our proposed platform considerably outperforms other [3D Slicer](https://www.slicer.org/) cloud-based approaches.

## Conclusions
Developed Slicer-DeepSeg allows the development of novel AI-assisted medical applications in neurosurgery. Moreover, it can enhance the outcomes of computer-aided diagnosis of brain tumours.

# Installation
First, download the 3D Slicer from [here](https://download.slicer.org/). Select the version the corresponds to your operating system.

Second, click on the "Install Slicer Extensions" button, and choosing Slicer-DeepSeg to be downloaded. 


**[2022_01_01] Update:** The Slicer-DeepSeg has not been added to the official 3D Slicer extension repository yet. However, it can be manually installed in the developper mode according to [the following instructions](https://www.slicer.org/wiki/Documentation/4.10/Training#Developing_and_contributing_extensions_for_3D_Slicer). Please follow the first 10 slides, but choose **Select Extension** instead of **Create Extension** as ilustrated in the tutorial. 

# Use Case Example
In order to demonstrate the capabilities of using Slicer-DeepSeg extension with 3D Slicer for addressing brain cancer research problems, a sample high-grade glioma (HGG) case from [the BraTS 2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html) was employed.

Slicer-DeepSeg can be selected from the machine learning category in the modules list in 3D Slicer. The default parameter settings include two different pre-trained deep learning models based on the input MRI image modalities. The first model is our previous work, [DeepSeg](https://link.springer.com/article/10.1007/s11548-020-02186-z) which requires only the T2-FLAIR MRI data as an input and automatically predicts the tumour region. The second model is the winning approach in the segmentation task of MICCAI BraTS 2020 challenge, [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z), which requires the four MRI modalities like the BraTS challenge: FLAIR, T1, T1ce, and T2. 

After the Slicer-DeepSeg installation, the user can choose one model, specifies its input data, creates a new segmentation volume, and presses the “apply” button. Then, an automatic pre-processing stage, including resampling, cropping and registration, is applied before the resultant tumour region is predicted using the specified pre-trained deep neural networks. Finally, the segmented tumour is displayed in both Slicer 2- and 3D scenes as presented in the following Fig:

![GUI](https://github.com/razeineldin/Slicer-DeepSeg/raw/main/Slicer-DeepSeg_Module_UI.png)
> Visualization of the brain tumour boundaries in MRI using Slicer-DeepSeg extension.

# License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

# Citation
The work has been published in the Current Directions in Biomedical Engineering, after the presentation in the German Society for Computer and Robot-Assisted Surgery (CURAC 2021). If you find this extension usefull, feel free to use it (or part of it) in your project and please cite the following paper:

    @article{ZeineldinWeimannKararMathisUllrichBurgert+2021+30+34,
    author = {Ramy A. Zeineldin and Pauline Weimann and Mohamed E. Karar and Franziska Mathis-Ullrich and Oliver Burgert},
    doi = {doi:10.1515/cdbme-2021-1007},
    url = {https://doi.org/10.1515/cdbme-2021-1007},
    title = {Slicer-DeepSeg: Open-Source Deep Learning Toolkit for Brain Tumour Segmentation},
    journal = {Current Directions in Biomedical Engineering},
    number = {1},
    volume = {7},
    year = {2021},
    pages = {30--34}
    }
    
# Disclaimer
*Slicer-DeepSeg*, like 3D Slicer, is for research purposes and not intended for clinical use. Therefore, The user assumes full responsibility to comply with the appropriate regulations.

