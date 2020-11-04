# IMPORTS
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json
import tensorflow as tf
nltk.download('punkt')

# loading the adta from json file
datafile = open('data.json')
data = json.load(datafile)

class chatbot:
	def __init__(self):
		self.model = None
		self.words = []
		self.classes = []
		self.documents = []
		self.train_x = None
		self.train_y = None

	def createdata(self):
		for intent in data['intents']:
		    for pattern in intent['patterns']:
		        w = nltk.word_tokenize(pattern)
		        self.words.extend(w)
		        self.documents.append((w, intent['tag']))
		        if intent['tag'] not in self.classes:
		            self.classes.append(intent['tag'])
		self.words = [stemmer.stem(w.lower()) for w in self.words if w not in '?']
		self.words = sorted(list(set(self.words)))
		self.classes = sorted(list(set(self.classes)))
		training = []
		output_empty = [0] * len(self.classes)
		for doc in self.documents:
		    bag = []
		    pattern_words = doc[0]
		    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
		    for w in self.words:
		        bag.append(1) if w in pattern_words else bag.append(0)
		    output_row = list(output_empty)
		    output_row[self.classes.index(doc[1])] = 1
		    
		    training.append([bag, output_row])
		random.shuffle(training)
		training = np.array(training)
		self.train_x = list(training[:,0])
		self.train_y = list(training[:,1])

	def create_model(self):
		self.model = Sequential()
		self.model.add(Dense(128, input_shape=(len(self.train_x[0]),), activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(len(self.train_y[0]), activation='softmax'))

	def compile_model(self):
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	def train_model(self):
		self.model.fit(np.array(self.train_x), np.array(self.train_y), epochs=500, batch_size=5, verbose=1)
		self.model.save('chatbot.h5')

	def clean_up_sentence(self, sentence):
	    sentence_words = nltk.word_tokenize(sentence)
	    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	    return sentence_words

	def bow(self, sentence, words, show_details=True):
	    sentence_words = self.clean_up_sentence(sentence)
	    bag = [0]*len(words)  
	    for s in sentence_words:
	        for i,w in enumerate(words):
	            if w == s: 
	                bag[i] = 1
	    return np.array(bag)

	def classify_local(self, sentence):
	    ERROR_THRESHOLD = 0.25
	    input_data = pd.DataFrame([self.bow(sentence, self.words)], dtype=float, index=['input'])
	    results = self.model.predict([input_data])[0]
	    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
	    results.sort(key=lambda x: x[1], reverse=True)
	    return_list = []
	    for r in results:
	        return_list.append([self.classes[r[0]], str(r[1])])
	    
	    return return_list

	def getresponse(self, inputString):
	  	ans = self.classify_local(inputString)
	  	prediction_list = [ans[i][1] for i in range(len(ans))]
	  	prediction = max(prediction_list)
	  	predicred_index = prediction_list.index(prediction)
	  	for tag in data['intents']:
		    if tag['tag'] == ans[predicred_index][0]:
		      return random.choice(tag['responses'])

	def model_reboot(self):
		self.createdata()
		self.create_model()
		self.compile_model()
		self.train_model()

if __name__ == '__main__':
	chatbot = chatbot()
	chatbot.model_reboot()
