# Mask-Face-Inpainting-Using-Residual-Attention-Unet
Masked Face Inpainting Through Residual Attention UNet

## How to run this project
Results can be replicated by following those steps:

### How to setup ENV
- If your **system does not have an Nvidia CUDA device available**, please comment `tensorflow-gpu==2.2.0` in the _environment.yml_ file.
- If you are running MacOS, change `tensorflow==2.2.0` to `tensorflow==2.0.0` in the _environment.yml_ file.
- Use [Conda ENV Manager](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to create new ENV: `conda env create -f environment.yml`
- Activate the ENV: 

### Get Data
- Download [Labeled Faces in the Wild data](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) and unzip its content into _data_ folder or use the "RAU_trian_test" that will download it automatically.
- You can get better results using larger dataset or dataset with higher quality images. For example [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) contains over 200 000 high quality images. 

### Part of CelebA process data can be download from here


### Configuration
You can configure the project using the `configuration.json`. Some of the items are set up and should not be changed. However, changing some of the following items can be useful. 
- `input_images_path`: Path to dataset with images that are input for the DataGenerator. If you want to use a different dataset than the default one, set the path to it here.
- `train_data_path`: Path where are training images generated and where training algorithm is looking for training data.
- `test_data_path`: Path where are testing images generated.
- `train_image_count`: Number of training image pairs generated by DataGenerator.
- `test_image_count`: Number of testing image pairs generated by DataGenerator.
- `train_data_limit`: Number of training image pairs used for model training.
- `test_data_limit`: Number of testing image pairs used for model testing.

### Train and test the model
- Run Jupyter server in the ENV: `RAU_trian_test`


###############################
###Acknowledgment

#### This work based on the following blog, please visit the original website ###

**Check out [the article](https://www.strv.com/blog/mask2face-how-we-built-ai-that-shows-face-beneath-mask-engineering) for a more in-depth explanation**

Also I would like to thanks Digital sreeni : https://github.com/bnsreenu

###########

Please cite our paper: 

@inproceedings{hosen2022masked,
  title={Masked Face Inpainting Through Residual Attention UNet},
  author={Hosen, Md Imran and Islam, Md Baharul},
  booktitle={2022 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
