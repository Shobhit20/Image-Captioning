import encode_image as ei
import SceneDesc
import test_mod as tm
import time as time
import pyttsx
import sys
def text(img):
	t1= time.time()
	encode = ei.model_gen()
	weight = 'Output/Weights.h5'
	sd = SceneDesc.scenedesc()
	model = sd.create_model(ret_model = True)
	model.load_weights(weight)
	image_path = img
	encoded_images = ei.encodings(encode, image_path)

	image_captions = tm.generate_captions(sd, model, encoded_images, beam_size=3)
	engine = pyttsx.init()
	print image_captions
	engine.say(	str(image_captions))
	engine.runAndWait()


if __name__ == '__main__':
	image = str(sys.argv[1])
	image = "test/"+image
	text(image)
