import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation
from keras.preprocessing import image, sequence
import cPickle as pickle

EMBEDDING_DIM = 128

class scenedesc():
	def __init__(self):
		self.vocab_size = None
		self.no_samples = None
		self.max_length = None
		self.index_word = None
		self.word_index = None
		self.image_encodings = pickle.load( open( "image_encodings.p", "rb" ) )
		self.captions = None
		self.img_id = None
		self.values()

	def values(self):
		dataframe = pd.read_csv('Flickr8K_Text/trainimgs.txt', delimiter='\t')
		self.captions = []
		self.img_id = []
		iter = dataframe.iterrows()

		for i in range(len(dataframe)):
		    nextiter = iter.next()
		    self.captions.append(nextiter[1][1])
		    self.img_id.append(nextiter[1][0])

		self.no_samples=0
		tokens = []
		for caption in self.captions:
		    self.no_samples+=len(caption.split())-1
		    tokens.append(caption.split())
		vocab = []
		for token in tokens:
		    vocab.extend(token)
		print len(vocab)
		vocab = list(set(vocab))
		self.vocab_size = len(vocab)

		caption_length = [len(caption.split()) for caption in self.captions]
		self.max_length = max(caption_length)
		self.word_index = {}
		self.index_word = {}
		for i, word in enumerate(vocab):
		    self.word_index[word]=i
		    self.index_word[i]=word


	def data_process(self, batch_size):
	    partial_captions = []
	    next_words = []
	    images = []
	    total_count = 0
	    while 1:
		    image_counter = -1
		    for caption in self.captions:
		        image_counter+=1
		        current_image = self.image_encodings[self.img_id[image_counter]]
		        for i in range(len(caption.split())-1):
		            total_count+=1
		            partial = [self.word_index[txt] for txt in caption.split()[:i+1]]
		            partial_captions.append(partial)
		            next = np.zeros(self.vocab_size)
		            next[self.word_index[caption.split()[i+1]]] = 1
		            next_words.append(next)
		            images.append(current_image)

		            if total_count>=batch_size:
		                next_words = np.asarray(next_words)
		                images = np.asarray(images)
		                partial_captions = sequence.pad_sequences(partial_captions, maxlen=self.max_length, padding='post')
		                total_count = 0
		                
		                yield [[images, partial_captions], next_words]
		                partial_captions = []
		                next_words = []
		                images = []


	def load_image(self, path):
		img = image.load_img(path, target_size=(224,224))
		x = image.img_to_array(img)
		return np.asarray(x)


	def create_model(self, ret_model = False):
	       
		image_model = Sequential()
		image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))
		image_model.add(RepeatVector(self.max_length))

		lang_model = Sequential()
		lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_length))
		lang_model.add(LSTM(256,return_sequences=True))
		lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

		model = Sequential()
		model.add(Merge([image_model, lang_model], mode='concat'))
		model.add(LSTM(1000,return_sequences=False))
		model.add(Dense(self.vocab_size))
		model.add(Activation('softmax'))

		print ("Model created!")

		if(ret_model==True):
		    return model

		model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
		return model

	def get_word(self,index):
		return self.index_word[index]

		
