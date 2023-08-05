import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from pickle import load
# from gtts.tts import gTTS
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

def extract_features(filename):
	vgg_model = VGG16()
	vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = vgg_model.predict(image, verbose=0)
	return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break      
    return in_text

tokenizer = load(open('pkl/tokenizer.pkl', 'rb'))
model = load_model('pkl/best_model.h5', compile=False)

from PIL import Image
import io

def gen(image_path):   
    max_length = 35
    features = extract_features(image_path)
    cap = predict_caption(model, features, tokenizer, max_length)
    final = cap.split()
    final = final[1:-1]
    final = ' '.join(final)
    final =  final.capitalize() + '.'
    return final