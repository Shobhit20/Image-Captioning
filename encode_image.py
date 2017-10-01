from vgg16 import VGG16
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input	
import six.moves.cPickle as pickle


def encodings(model, path):
	processed_img = image.load_img(path, target_size=(224,224))
	x = image.img_to_array(processed_img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	image_final = np.asarray(x)
	prediction = model.predict(image_final)
	prediction = np.reshape(prediction, prediction.shape[1])
	print prediction
	return prediction


def encode_image():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	image_encodings = {}

	train_imgs_id = open("Flickr8K_Text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]
	print len(train_imgs_id)
	test_imgs_id = open("Flickr8K_Text/Flickr_8k.testImages.txt").read().split('\n')[:-1]
	images = []
	images.extend(train_imgs_id)
	images.extend(test_imgs_id)
	print len(images)
	counter=1

	for img in images:
		path = "Flickr8K_Data/"+str(img)
		image_encodings[img] = encodings(model, path)
		print counter
		counter += 1

	with open( "image_encodings.p", "wb" ) as pickle_f:
		pickle.dump( image_encodings, pickle_f )





if __name__=="__main__":
	encode_image()
