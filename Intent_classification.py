# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:48:39 2019

@author: Dell
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

#%%
def load_dataset(filename):
#   df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
    df = pd.read_excel(filename, sheet_name="dataset_XY", encoding="utf8")
    Label_tag = pd.read_excel(filename, sheet_name="Label_tag", encoding="utf8")  
    print(df.head())
    intent = df['Label']
    for idx, i in enumerate(intent):
#         print(i)
        intent[idx] = Label_tag.loc[i,"Tag"]
#     print("intent: ", intent)
    unique_intent = list(set(intent))
    sentences = list(df["Questions"])
  
    return (intent, unique_intent, sentences)

#%%
intent, unique_intent, sentences = load_dataset("dataset_XY_XLS_updatedbytho.xls")
print("Unique intents: ", unique_intent)
print("Number of intents: ", len(unique_intent))
#%%
import pyvi.ViTokenizer as viToken
def viTokenList(word):
    return viToken.tokenize(word).split()
viTokenList("Anh thọ quá đẹp trai!")

#%%
def cleaning(sentences):
    words = []
    for s in sentences:
#         clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s) # Find and replace duplicate 
        w = viTokenList(s)
        #stemming
        words.append([i.lower() for i in w])

    return words  

#%%
cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  

#%%
def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

#%%
def max_length(words):
    return(len(max(words, key = len)))
    
#%%
word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

#%%
def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)

def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
    
padded_doc = padding_doc(encoded_doc, max_length)

#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

output_tokenizer.word_index

encoded_output = encoding_doc(output_tokenizer, intent)

encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

def one_hot(encode):
    o = OneHotEncoder(sparse = False)
    return(o.fit_transform(encode))

output_one_hot = one_hot(encoded_output)

from sklearn.model_selection import train_test_split

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)

def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
    model.add(Bidirectional(LSTM(128)))
    #   model.add(LSTM(128))
    model.add(Dense(32, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation = "softmax"))

    return model

model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()
#%%
model = load_model("model.h5")
 
 #%%
 
def predictions(text):
#     clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = viTokenList(text)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
#     print("test_ls: ", test_ls)
#     print("test_word: ", test_word)
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))

    test_ls = np.array(test_ls).reshape(1, len(test_ls))

    x = padding_doc(test_ls, max_length)

    pred = model.predict_proba(x)


    return pred

#%%
def load_response_dataset(filename):
    df_res = pd.read_excel(filename, sheet_name="Tag_Response", encoding="utf8") 
    return df_res

#%%
import random
def response(classes):
    df_res = load_response_dataset("dataset_XY_XLS_updatedbytho.xls")
    s = df_res[classes].dropna()
    s_res = s[random.randint(0,len(s)-1)]
    print(s_res)
    
#%%
def get_final_output(pred, classes):
    predictions = pred[0]

    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)
 
    for i in range(3):
        print("%s has confidence = %s" % (classes[i], (predictions[i])))
#    response(classes[0])

#%%
a = False
while a == False:
    text = input("Please ask: ")
    pred = predictions(text)
    get_final_output(pred, unique_intent)
    
    if text == 'q':
        a = True

                                   









