import cPickle as pickle
import SceneDesc as desc
from keras.preprocessing import sequence
import nltk
import numpy as np

def process_caption(sd, caption):
    tokens = caption.split()
    tokens = tokens[1:]
    terminate_id = tokens.index('<end>')
    tokens = tokens[:terminate_id]
    return " ".join([word for word in tokens])


def generate_captions(sd, model, encoded_images, beam_size):
    first_word = [sd.word_index['<start>']]
    prob_level = 0.0
    capt_seq = [[first_word, prob_level]]
    max_cap_length = sd.max_length
    while len(capt_seq[0][0]) < max_cap_length:
        temp_capt_seq = []
        for caption_id in capt_seq:
            iter_capt = sequence.pad_sequences([caption_id[0]], max_cap_length, padding = 'post')
            next_word_prob = model.predict([np.asarray([encoded_images]), np.asarray(iter_capt)])[0]
            next_word_ids = np.argsort(next_word_prob)[-beam_size:]
            for word_id in next_word_ids:
                new_iter_capt, new_iter_prob = caption_id[0][:], caption_id[1]
                new_iter_capt.append(word_id)
                new_iter_prob+=next_word_prob[word_id]
                temp_capt_seq.append([new_iter_capt,new_iter_prob])
        capt_seq = temp_capt_seq
        capt_seq.sort(key = lambda l:l[1])
        capt_seq = capt_seq[-beam_size:]
    best_caption = capt_seq[len(capt_seq)-1][0]
    best_caption = " ".join([sd.index_word[index] for index in best_caption])
    image_desc = process_caption(sd, best_caption)
    return image_desc
