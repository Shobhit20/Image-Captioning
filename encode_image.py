from vgg16 import VGG16
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input	
import six.moves.cPickle as pickle
import progressbar

def model_gen():
	model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
	return model


def encodings(model, path):
	processed_img = image.load_img(path, target_size=(224,224))
	x = image.img_to_array(processed_img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	image_final = np.asarray(x)
	prediction = model.predict(image_final)
	prediction = np.reshape(prediction, prediction.shape[1])

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
	bar = progressbar.ProgressBar(maxval=len(images), \
    		widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	counter=1
	print "Encoding images"

	for img in images:
		path = "Flickr8K_Data/"+str(img)
		image_encodings[img] = encodings(model, path)
		bar.update(counter)
		counter += 1

	bar.finish()
	with open( "image_encodings.p", "wb" ) as pickle_f:
		pickle.dump( image_encodings, pickle_f )
	print "Encodings dumped into image_encodings.p"




if __name__=="__main__":
	encode_image()
