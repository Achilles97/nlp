#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:01:07 2017

@author: Pranjal
"""

import pandas as pd
import numpy as np
 
data = pd.read_csv('Tweets.csv')
x = data.iloc[:, 10].values
y = data.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_y = LabelEncoder()
y = encoder_y.fit_transform(y)
y = y.reshape(-1, 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

import re
import nltk
from gensim import corpora


corpus = []
for i in range(len(x)):
    text = re.sub('[^a-zA-Z0-9]',' ', x[i])
    text = text.lower()
    text = nltk.word_tokenize(text)
    corpus.append(text)
    
#dictionary = corpora.Dictionary(corpus)
#word2index = dict(dictionary.token2id)

import itertools
freq_dist = nltk.FreqDist(itertools.chain(*corpus))

vocab = freq_dist.most_common(15079)

index2word = ['_'] + ['unk'] + [m[0] for m in vocab]

word2index = dict([(w,i) for i,w in enumerate(index2word)])

train_text = []
for i in range(len(corpus)):
    line = []
    for word in corpus[i]:
        if word in word2index:
            index = word2index[word]
        else:
            index = word2index['unk']
        line.append(index)
    train_text.append(line)    

from keras.preprocessing.sequence import pad_sequences
train_text = pad_sequences(train_text, maxlen=30, dtype='int32',
    padding='post', truncating='post', value=0)

from sklearn.cross_validation import ShuffleSplit
bs = ShuffleSplit(len(train_text), test_size = 0.05, random_state = 0)

for train_index, test_index in bs:
    print ("TRAIN:", train_index, "TEST:", test_index)
    
x_train = train_text[train_index]
y_train = y[train_index]
x_test = train_text[test_index]
y_test = y[test_index]

from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, SpatialDropout1D
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(len(word2index), 32, input_length=30))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.2,kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', recurrent_dropout=0.2))
model.add(Dense(32, kernel_initializer='random_uniform', activation='tanh'))
model.add(Dense(3, kernel_initializer='random_uniform', activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=5,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=32)

print('Test score:', score)
print('Test accuracy:', acc)


