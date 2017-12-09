

def preprocessing():
	image_captions = open("Flickr8K_Text/Flickr8k.token.txt").read().split('\n')
	caption = {}
	for i in range(len(image_captions)-1):
		id_capt = image_captions[i].split("\t")
		id_capt[0] = id_capt[0][:len(id_capt[0])-2] 	# to rip off the #0,#1,#2,#3,#4 from the tokens file
		try:
			caption[id_capt[0]].append(id_capt[1])
		except:
			caption[id_capt[0]] = [id_capt[1]]
	train_imgs_id = open("Flickr8K_Text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]

	
	train_imgs_captions = open("Flickr8K_Text/trainimgs.txt",'wb')
	for img_id in train_imgs_id:
		for captions in caption[img_id]:
			desc = "<start> "+captions+" <end>"
			train_imgs_captions.write(img_id+"\t"+desc+"\n")
			train_imgs_captions.flush()
	train_imgs_captions.close()

	test_imgs_id = open("Flickr8K_Text/Flickr_8k.testImages.txt").read().split('\n')[:-1]

	test_imgs_captions = open("Flickr8K_Text/testimgs.txt",'wb')
	for img_id in test_imgs_id:
		for captions in caption[img_id]:
			desc = "<start> "+captions+" <end>"
			test_imgs_captions.write(img_id+"\t"+desc+"\n")
			test_imgs_captions.flush()
	test_imgs_captions.close()

 

if __name__=="__main__":
	preprocessing()
