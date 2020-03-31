# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zL1tvO_AKuqThFtl1KN7QsC5Zz6c6gj6
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x magic
#!pip install music21
#!pip install keras-gcn
#!pip install hyperas==0.4.1
from __future__ import print_function,division, absolute_import
import os
import glob
import csv
import math
import time
import h5py
import numpy as np
import tensorflow as tf
import pretty_midi
from music21 import converter, instrument, note, chord, stream
import keras
from keras import Sequential , layers, activations, initializers, constraints, regularizers
from keras.engine import Layer
from keras.layers.core import Reshape, Dense, Dropout,Activation,Flatten
from keras.layers import Input,Dropout,RepeatVector, Dense, TimeDistributed,Embedding,LSTM, CuDNNLSTM,Flatten,concatenate,Lambda,Conv2D
from keras.optimizers import Adam,RMSprop
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.layers.normalization import *
from keras.activations import *
from keras.optimizers import *
from keras.models import Model, load_model
from keras.regularizers import l2
import keras.backend as K
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from keras_gcn import GraphConv
import random
import hyperas
import torch as th

import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform
from PIL import Image
import matplotlib.pyplot as plt
midi_dir = 'classicinput'
out_dir = 'output'
X_train=np.load('output\Chinese_X_train.npy')
A_train=np.load('output\Chinese_A_train.npy')
def preparedata(X_train,A_train):
    return X_train, A_train
def create_model():
    K.clear_session()
    midi_dir ='input'
    out_dir = 'output'
    pitch = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","C"]
    def pitch_to_note(pitchname):
        for i in range(128):
            if(pitchname == pitch[i%12] + str(int(i/12)-1)):
                return i
            elif(pitchname == pitch[i%12+1] + str(-1*(int(i/12)-1))):
                return i
    # def note_to_pitch(notee):
    #     notee = np.array(notee)
    #     print(len(notee))
    #     for k in range(64):
    #         pm = pretty_midi.PrettyMIDI(resolution=220,initial_tempo=120.0)
    #         instrument = pretty_midi.Instrument(0)
    #         for i in range(32):
    #             for j in range(128):
    #                 element = notee[k,j,i]
    #                 if(element > 1e-3):
    #                     note = pretty_midi.Note(velocity=100,pitch=j,start=i*0.5,end=(i+1)*0.5)
    #                     instrument.notes.append(note)
    #                     pm.instruments.append(instrument)

    #     count = 0
    #     for note in instrument.notes:
    #         count += 1
    #     return pm
    def note_to_pitch(notee):
        notee = np.array(notee)
        k = random.randint(0,len(notee))
        pm = pretty_midi.PrettyMIDI(resolution=220,initial_tempo=120.0)
        instrument = pretty_midi.Instrument(0)
        for i in range(64):
            for j in range(128):
                element = notee[k,j,i]
                if(element > 1e-3):
                    note = pretty_midi.Note(velocity=100,pitch=j,start=i*0.25,end=(i+1)*0.25)
                    instrument.notes.append(note)
                    pm.instruments.append(instrument)
                # else:
                    # print("element<1e-3")
        count = 0
        for note in instrument.notes:
            count += 1
        return pm                    
    X_train=np.load('output\Chinese_X_train.npy')
    A_train=np.load('output\Chinese_A_train.npy')
    graph = []
    #
    X_rows = 128 
    X_cols = 64
    X_channels = 1
    X_shape = (X_rows, X_cols)

    A_rows = 128 
    A_cols = 128
    A_channels = 1
    A_shape = (A_rows, A_cols) 
    z_dim = 32
    steps=150
    optimizer = Adam(0.0002, 0.5)
    G = [Input(shape=(128, 64),batch_shape=None,sparse=False)]
    # discriminator

    # # Generator
    # self.Xgenerator = self.create_Xgenerator()
    # self.Agenerator = self.create_Agenerator()
    # # 
    # self.Xgenerator.compile(loss='binary_crossentropy', optimizer=optimizer)
    # self.Agenerator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

    # self.combined = self.build_combined2()
    # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    Z_in = Input(shape=(z_dim,))
    H = Dense({{choice([16,32,64,128,256,512])}})(Z_in)
    H = LeakyReLU(alpha = {{uniform(0,1)}})(H)
    H = BatchNormalization(momentum={{uniform(0,1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0,1)}})(H)
    H = Dropout(0.5)(H)
    H = BatchNormalization(momentum={{uniform(0,1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0,1)}})(H)
    H = Dropout(0.5)(H)
    H = BatchNormalization(momentum={{uniform(0,1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0,1)}})(H)
    H = Dense(np.prod(X_shape),activation='tanh')(H)
    H = Reshape(X_shape)(H)
    Xgenerator = Model(Z_in,H,name='Xgenerator')
    Xgenerator.compile(loss='binary_crossentropy', optimizer=Adam(lr = {{uniform(0, 1)}},beta_1={{uniform(0,1)}},beta_2={{uniform(0,1)}},decay={{uniform(0,1)}}))
    Xgenerator.summary()
    




    Z_in = Input(shape=(z_dim,))
    H = Dense({{choice([16,32,64,128,256,512])}})(Z_in)
    H = LeakyReLU({{uniform(0, 1)}})(H)
    H = BatchNormalization(momentum={{uniform(0, 1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0,1)}})(H)
    H = Dropout(0.5)(H)
    H = BatchNormalization(momentum={{uniform(0,1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0,1)}})(H)
    H = Dropout(0.5)(H)
    H = BatchNormalization(momentum={{uniform(0,1)}})(H)
    H = Dense({{choice([16,32,64,128,256,512])}})(H)
    H = LeakyReLU({{uniform(0, 1)}})(H)
    H = Dense(np.prod(A_shape), activation='tanh')(H)
    H = Reshape(A_shape)(H)
    Agenerator = Model(Z_in,H,name='Agenerator')
    Agenerator.compile(loss='binary_crossentropy', optimizer=Adam(lr = {{uniform(0, 1)}},beta_1={{uniform(0,1)}},beta_2={{uniform(0,1)}},decay={{uniform(0,1)}}))
    Agenerator.summary()
    




    # 
    X_in = Input(shape=(128,64))
    A = Input(shape=(128,128))
    H = GraphConv(64)([X_in,A])
    H = GraphConv(64)([H,A])
    H = GraphConv(64)([H,A])
    H = Flatten()(H)
    H = Dropout({{uniform(0, 1)}})(H)
    Y = Dense(units={{choice([16,32,64,128,256,512])}},activation='tanh')(H)
    Y = LeakyReLU({{uniform(0,1)}})(Y)
    Y = Dense(1,activation='sigmoid')(Y)
    #
    discriminator = Model(inputs=[X_in,A], outputs=Y, name='Discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer=SGD(lr={{uniform(0,1)}},momentum={{uniform(0,1)}},decay={{uniform(0,1)}},nesterov={{choice([True,False])}}))
    discriminator.summary()
    



    z = Input(shape=(z_dim,))
    X = Xgenerator(z)
    A = Agenerator(z)
    support = 1
    gan_V = discriminator([X,A])
    model = Model(inputs=z,outputs=gan_V,name='Model')
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr = {{uniform(0, 1)}},beta_1={{uniform(0,1)}},beta_2={{uniform(0,1)}},decay={{uniform(0,1)}}))
    model.summary()
    



        # discriminator
    

    
    
    batch_size = 64 
    half_batch = int(batch_size / 2)
    noise_half = np.random.normal(0, 1, (half_batch, z_dim))
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    #starttrain = time.time()
    for step in range(steps):
        graphs = []
        # ---------------------
        #  Discriminator
        # ---------------------

        # noise = np.random.normal(0, 1, (half_batch, z_dim))
        # noise = np.random.uniform(0, 1, (half_batch, z_dim))
        gen_A = Agenerator.predict(noise_half)
        gen_X = Xgenerator.predict(noise_half)


            # print(gen_X)
        for t in range(0,len(X_train)//batch_size):
        #
            idx = np.random.randint(0,len(X_train), half_batch)
            #print(idx)

            graphsX = []
            graphsA = []
            for i in idx:
                graphsX.append(X_train[i])
                graphsA.append(A_train[i])
            graphs = [graphsX,graphsA]
            # discriminator
            # 
            valid_y = np.array([1] * batch_size)
            # noise = np.random.normal(0, 1, (batch_size, z_dim))
                # noise = np.random.uniform(0,1,(batch_size,self.z_dim))
            g_loss = model.train_on_batch(noise, valid_y)
            # noise = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = model.train_on_batch(noise, valid_y,)
            d_loss_real = discriminator.train_on_batch(graphs, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch([gen_X,gen_A], np.zeros((half_batch, 1)))
            #
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Generator
            # ---------------------
            valid_y = np.array([1] * batch_size)
            # noise = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = model.train_on_batch(noise, valid_y,)
            # noise = np.random.normal(0, 1, (batch_size, z_dim))
            # noise = np.random.uniform(0,1,(batch_size,self.z_dim))
            g_loss = model.train_on_batch(noise, valid_y)
            

        # Train the generator
        # g_loss = self.combined.train_on_batch(noise, valid_y)
        # g_loss = model.train_on_batch(noise, np.ones((half_batch,1)))
        

    
    #elapsed_time = time.time() - starttrain
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")    
    return {'loss':g_loss,'status':STATUS_OK}
            #,'model':model}


if __name__ == '__main__':
    #     gan = GAN()
#     gan.train(steps=3000, batch_size=5376, save_interval=150)
    
    best_run, best_model = optim.minimize(model=create_model,data = preparedata,algo=tpe.suggest,max_evals=75,trials=Trials(),keep_temp=True)
    
    print(best_run)

