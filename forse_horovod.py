from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, Dense, Input
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy as np
import os
from tensorflow.keras import backend as K
import re

from slack import WebClient
from flask import Flask
from slackeventsapi import SlackEventAdapter

import threading
from datetime import datetime
import time

import horovod.tensorflow.keras as hvd

class DCGAN:
    def __init__(self, output_directory, img_size, slack=False, slack_talk=False, generator = None, discriminator = None):
        self.img_size = img_size
        self.channels = 1
        self.kernel_size = 5
        self.output_directory = output_directory
        self.slack = slack; self.slack_talk = slack_talk; 
        
        if generator is not None:
            self.checkpoint = re.findall(r'\d+',generator)[-2] # find the numbers in a string, -1 is 5
            self.generator = load_model(generator) # the filename of the trained models 
            self.discriminator = load_model(discriminator)
        else:
            self.generator = None
            self.discriminator = None
            self.checkpoint = None
           
        if slack:
            with open('/global/homes/j/jianyao/Small_Scale_Foreground/slack_key.txt', 'r') as f:
                slack_api_token = f.read()   
            self.client = WebClient(token=slack_api_token)
        if slack_talk:
            self.app = Flask(__name__)
            signing_secrete = '8399c5f959906a3e5c97d3e50e5dfc21'
            slack_event_adapter = SlackEventAdapter(signing_secrete, '/slack/events',self.app)
            self.BOT_ID = self.client.api_call("auth.test")['user_id']
               
            @slack_event_adapter.on('message')
            def message(payload):
                event = payload.get('event', {})
                channel_id = event.get('channel')
                user_id = event.get('user')
                if self.BOT_ID!=user_id:
                    self.client.chat_postMessage(channel=channel_id, text='Now is at epoch {}/{}, time cost is {:0.2f} mins, ETA:{:0.2f} hours.'.format(self.epoch, self.epochs, (self.time-self.start)/60, (self.epochs - self.epoch)*(self.time-self.start)/60/60/self.epoch))

    def smooth_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))   
        

    def build_generator(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same")) # 64x64x64
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same", strides=2)) #32x32x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, padding="same", strides=2)) #16x16x256
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=self.kernel_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(self.channels, kernel_size=self.kernel_size, padding="same"))
        model.add(Activation("tanh"))
        img_in = Input(shape=img_shape)
        img_out = model(img_in)
        return Model(img_in, img_out)

    def build_discriminator(self):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=1, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=img_shape)
        validity = model(img)
        return Model(img, validity)

    def build_gan(self, optimizer):
        img_shape = (self.img_size[0], self.img_size[1], self.channels)
        # optimizer = Adam(0.000005, 0.2)
        
        if self.generator is None:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], experimental_run_tf_function=False)
            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer, experimental_run_tf_function=False)
            
        z = Input(shape=img_shape)
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, experimental_run_tf_function=False)
        
        return self.generator, self.discriminator, self.combined
