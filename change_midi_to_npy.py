from __future__ import print_function,division, absolute_import
import os
import glob
import csv
import math
import time
import h5py
import random
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from music21 import converter, instrument, note, chord, stream
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import keras
from keras import Sequential , layers, activations, initializers, constraints
import keras.backend as K
from keras.engine import Layer
from keras.optimizers import *
from keras.activations import *
from keras.models import Model, load_model
from keras.optimizers import Adam,RMSprop
from keras.layers.normalization import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Reshape, Dense, Dropout,Activation,Flatten
from keras.layers import Input,Dropout,RepeatVector, Dense, TimeDistributed,Embedding,LSTM, CuDNNLSTM,Flatten,concatenate,Lambda,Conv2D
from keras_gcn import GraphConv
from tensorflow.nn import sigmoid_cross_entropy_with_logits
import hyperas
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform


np.set_printoptions(threshold=np.inf)
midi_dir = 'parse'
out_dir = 'parse'


pitch = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B","C"]
def pitch_to_note(pitchname):
    for i in range(128):
        if(pitchname == pitch[i%12] + str(int(i/12)-1)):
            return i
        elif(pitchname == pitch[i%12+1] + str(-1*(int(i/12)-1))):
            return i
def note_to_pitch(notee):
    notesize = 64
    notee = np.array(notee)
    k = random.randint(0,len(notee)-1)
    pm = pretty_midi.PrettyMIDI(resolution=220,initial_tempo=120.0)
    instrument = pretty_midi.Instrument(0)
    for j in range(128):
        for i in range(notesize):
            element = notee[k,j,i]
            if(element > 0.09):
                note = pretty_midi.Note(velocity=100,pitch=j,start=i*0.125,end=(i+1)*0.125)
                instrument.notes.append(note)
                pm.instruments.append(instrument)
    count = 0
    for note in instrument.notes:
        count += 1
    return pm
   
def parse_midi_files(dir):
    notesize = 64 
    files = glob.glob(os.path.join(dir, '*.mid'))
    notes = []
    songs = []
    # file_list = np.empty(shape=[0,128,128])
    X_train = []
    A_train = []
    start = time.time()
    song = np.zeros((128,notesize))
    adjacency = np.zeros((128,128))
    for file in files:
    
        num = 0
        t = 0
        k = 0
        pre_node = 0
        now_node = 0
        song = np.zeros((128,notesize))
        adjacency = np.zeros((128,128))
        digree = np.zeros((128,128))
        laplacian = np.zeros((128,128))
        rest_adjacency = np.zeros((128,128))
        rest_digree = np.zeros((128,128))
        rest_laplacian = np.zeros((128,128))

        # file_list = file_list.tolist()
        # file_list.append(os.path.basename(file))
        # file_list = np.array(file_list)
        midi = converter.parse(file)
        print("Parsing %s" % file)
        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
            
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        
        for element in range (len(notes_to_parse)//notesize):
            num = 0
            for j in range((element-1)*notesize,element*notesize):
                print(type(j))
                if isinstance(j, note.Note):
                    print('AAA')
                    song[pitch_to_note((j%notesize).nameWithOctave)][num]=1
                    now_node = pitch_to_note((element%notesize).nameWithOctave)
                    adjacency[now_node][pre_node] = 1
                    t = pitch_to_note((j%notesize).nameWithOctave)
                    if(k != num ):
                        rest_adjacency[pre_node][now_node]=1
                        rest_digree[pre_node][pre_node] += 1
                        rest_digree[now_node][now_node] += 1
                    num += 1
                    k=num  
                    pre_node = pitch_to_note(j.nameWithOctave)
                    
                elif isinstance(j,note.Rest):       
                    adjacency[num][t]=1     
                    num+=1
                else:  
                    print("AA")
                    num+=1  
            A = np.matrix(adjacency)
            X = np.matrix(song)
            X_train.append(X)
            A_train.append(A)
            # elif(num >128):
            #     for j in range (100):
            #         for i in range (128*(j-1),128*j):
            #             if isinstance(element, note.Note):
            #                 song[num][pitch_to_note(element.nameWithOctave)]=1
            #                 now_node = pitch_to_note(element.nameWithOctave)
            #                 adjacency[pre_node][now_node] = 1
            #                 t = pitch_to_note(element.nameWithOctave)
            #                 if(k != num ):
            #                     rest_adjacency[pre_node][now_node]=1
            #                     rest_digree[pre_node][pre_node] += 1
            #                     rest_digree[now_node][now_node] += 1
            #                 num += 1
            #                 k=num  
            #                 pre_node = pitch_to_note(element.nameWithOctave)
            #                 A = np.matrix(adjacency)
            #                 X = np.matrix(song)
            #                 X_train.append(X)
            #                 A_train.append(A)
            #             elif isinstance(element,note.Rest):       
            #                 adjacency[num][t]=1     
            #                 num+=1
            #             else:  
            #                 num+=1
        songs.append(song.tolist())
        notes += song.tolist()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return notes, songs, A_train,song,X_train

notes, songs,A_train,song,X_train = parse_midi_files(midi_dir)
print("parse finished")
print(len(X_train))

note_to_pitch(X_train).write('AAAAAA_.mid')