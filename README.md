# Image Captioning
Image captioning is describing an image fed to the model. The task of object detection has been studied for a long time but recently the task of image captioning is coming into light. This repository contains the "Neural Image Caption" model proposed by Vinyals et. al.[1]

## Dataset
The dataset used is flickr8k. You can request the data [here](https://illinois.edu/fb/sec/1713398). An email for the links 
of the data to be downloaded will be mailed to your id. Extract the images in Flickr8K_Data and the text data in Flickr8K_Text.

## Requirements
1. Tensorflow
2. Keras
3. Numpy
4. h5py
5. Pandas
6. Pillow
7. Pyttsx

## Steps to execute
1. After extracting the data, execute the preprocess_data.py file by locating the file directory and execute "python preprocess_data.py". This file adds "start " and " end" token to the training and testing text data. On execution the file creates new txt files in Flickr8K_Text folder.

2. Execute the encode_image.py file by typing "python encode_image.py" in the terminal window of the file directory. This creates image_encodings.p which generates image encodings by feeding the image to VGG16 model. In case the weights are not directly available in your temp directory, the weights will be downloaded first.

3. Execute the train.py file in terminal window as "python train.py (int)". Replace "(int)" by any integer value. The variable will denote the number of epochs for which the model will be trained. The models will be saved in the Output folder in this directory. The weights and model after training for 70 epochs can be found [here](https://drive.google.com/drive/folders/1aukgi_3xtuRkcQGoyAaya5pP4aoDzl7r).

4. After training execute "python test.py image" for generating a caption of an image. Pass the extension of the image along with the name of the image file for example, "python test.py beach.jpg". The image file must be present in the test folder.

NOTE - You can skip the training part by directly downloading the weights and model file and placing them in the Output folder since the training part wil take a lot of time if working on a non-GPU system. A GTX 1050 Ti with 4 gigs of RAM takes around 10-15 minutes for one epoch.

## Output
The output of the model is a caption to the image and a python library called pyttsx which converts the generated text to audio

## Results
Following are a few results obtained after training the model for 70 epochs.

Image | Caption 
--- | --- 
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/beach.jpg" width="400"> | **Generated Caption:** A brown dog is running in the water.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/image.jpg" width="400"> | **Generated Caption:** A tennis player hitting the ball.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/child.png" width="400"> | **Generated Caption:** A child in a helmet is riding a bike.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/street.png" width="400"> | **Generated Caption:** A group of people are walking on a busy street.


On providing an ambiguous image for example a hamsters face morphed on a lion the model got confused but since the data is a bit biased towards dogs hence it captions it as a dog and the reddish pink nose of the hamster is identified as red ball

Image | Caption
--- | ---
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/9.jpg" width="400"> | **Generated Caption:** A black dog is running through the snow with a red ball.


In some cases the classifier got confused and on blurring an image it produced bizzare results

Image | Caption
--- | ---
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/img1.jpg" width="400"> | **Generated Caption:** A brown dog and a brown dog are playing with a ball in the snow.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/img1_blur.jpg" width="400"> | **Generated Caption:** A little girl in a white shirt is running on the grass.

## References
#### NIC Model
[1] Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
#### Data
https://illinois.edu/fb/sec/1713398
#### VGG16 Model
https://github.com/fchollet/deep-learning-models
#### Saved Model
https://drive.google.com/drive/folders/1aukgi_3xtuRkcQGoyAaya5pP4aoDzl7r
#### Code reference
https://github.com/anuragmishracse/caption_generator

You can find a detailed report in the Report folder.
