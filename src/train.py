# Generating Faces model training
# The goal is to improve the learning speed of anime character face generation using a pretrained VGG16 convolutional neural network model**
# Here's the original approach without the use of transfer learning:
# https://www.kaggle.com/code/nassimyagoub/gan-anime-faces

# VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes.

# Importing libraries and loading data

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input, Conv2D, MaxPooling2D, Activation, Dropout, Conv2DTranspose, BatchNormalization
from tensorflow.keras.applications import VGG16

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath
                
def load_images(directory='', size=(64,64)):
    images = []
    labels = []  # Integers corresponding to the categories in alphabetical order
    label = 0
    
    imagePaths = list(list_images(directory))
    
    for path in tqdm.tqdm(imagePaths):
        
        if not('OSX' in path):
        
            path = path.replace('\\','/')
            print(path)
            image = cv2.imread(path) #Reading the image with OpenCV
            image = cv2.resize(image,size) #Resizing the image, in case some are not of the same size

            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    return images

# images=load_images('/content/drive/MyDrive/DL_models/anime_ds_full/images')

import pickle

# File path to stored pickle data
file_path = './data/processed/images.pkl'

# Load the images list from the pickle file
with open(file_path, 'rb') as file:
    images = pickle.load(file)

print("Images loaded from the pickle file.")

## Looking at some images"""

_,ax = plt.subplots(5,5, figsize = (8,8)) 
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(images[5*i+j])
        ax[i,j].axis('off')

## Generative Adversarial Networks

# The objective of a GAN is to train a data generator in order to imitate a given dataset.
# A GAN is similar to a zero sum game between two neural networks, the generator of data and a discriminator, trained to recognize original data from fakes created by the generator.

## Creating the GAN

## We incorporated a pretrained VGG16 model into the discriminator model and added the capability to continue training from the epoch at which the model was saved. This allows us to leverage the knowledge learned by the VGG16 model and further enhance the discriminator's performance in our GAN-based anime character face generation. The pretrained VGG16 model provides a strong backbone for feature extraction, improving the discriminator's ability to distinguish between real and generated images. Additionally, the ability to continue training from a saved checkpoint allows us to fine-tune the model and potentially achieve even better results over time.

class GAN():
    def __init__(self, load_model=False, start_epoch=0):
        self.img_shape = (64, 64, 3)
        self.start_epoch = start_epoch
        self.noise_size = 100

        optimizer = Adam(0.0002,0.5)

        # Check if models need to be loaded
        if load_model:
            # load last saved models
            self.generator = tf.keras.models.load_model("/models/generator_%d" % start_epoch)
            self.discriminator = tf.keras.models.load_model("/models/discriminator_%d" % start_epoch)
        else:
            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()

        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined = Sequential()
        self.combined.add(self.generator)
        self.combined.add(self.discriminator)
        
        self.discriminator.trainable = False
        
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.combined.summary()
        
    # Creating the generator, the large kernels in the convolutional layers allow the network to create complex structures.
    def build_generator(self):
        epsilon = 0.00001 # Small float added to variance to avoid dividing by zero in the BatchNorm layers.
        noise_shape = (self.noise_size,)
        
        model = Sequential()
        
        model.add(Dense(4*4*512, activation='linear', input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 512)))
        
        model.add(Conv2DTranspose(512, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(256, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(128, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(64, kernel_size=[4,4], strides=[2,2], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2DTranspose(3, kernel_size=[4,4], strides=[1,1], padding="same",
                                  kernel_initializer= keras.initializers.TruncatedNormal(stddev=0.02)))

        # Standard activation for the generator of a GAN
        model.add(Activation("tanh"))
        
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    from keras.applications import VGG16

    def build_discriminator(self):

        vgg = VGG16(weights='imagenet', include_top=False, input_shape=self.img_shape)

        # Freeze the layers
        for layer in vgg.layers:
            layer.trainable = False

        model = Sequential()
        model.add(vgg)

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=128, metrics_update=50, save_images=100, save_model=2000):

        X_train = np.array(images)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)
        
        mean_d_loss=[0,0]
        mean_g_loss=0

        for epoch in tqdm(range(self.start_epoch, epochs)):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)

            # Training the discriminator
            self.discriminator.trainable = True
            self.discriminator.compile(loss='binary_crossentropy', 
                                      optimizer=Adam(0.0002,0.5),
                                      metrics=['accuracy'])
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Training the generator
            self.discriminator.trainable = False
            self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002,0.5))
            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            valid_y = np.array([1] * batch_size)
            g_loss = self.combined.train_on_batch(noise, valid_y)
                
            # We print the losses and accuracy of the networks every 200 batches mainly to make sure the accuracy of the discriminator
            # is not stable at around 50% or 100% (which would mean the discriminator performs not well enough or too well)
            if epoch % metrics_update == 0:
                    print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, mean_d_loss[0]/metrics_update, 100*mean_d_loss[1]/metrics_update, mean_g_loss/metrics_update))
                    mean_d_loss=[0,0]
                    mean_g_loss=0
                
            # Saving 25 images
            if epoch % save_images == 0:
                self.save_images(epoch)
                
            # We save the architecture of the model, the weights and the state of the optimizer
            # This way we can restart the training exactly where we stopped
            if epoch % save_model == 0:
                self.generator.save("/models/generator_%d" % epoch)
                self.discriminator.save("/models/discriminator_%d" % epoch)

          # Saving 25 generated images to have a representation of the spectrum of images created by the generator
    def save_images(self, epoch):
        noise = np.random.normal(0, 1, (25, self.noise_size))
        gen_imgs = self.generator.predict(noise)
        
        # Rescale from [-1,1] into [0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(5,5, figsize = (8,8))

        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(gen_imgs[5*i+j])
                axs[i,j].axis('off')

        plt.show()
        
        fig.savefig("/content/drive/MyDrive/DL_models/animeGenerated/Faces_%d.png" % epoch)
        plt.close()

### Training session

gan = GAN(load_model=True, start_epoch=300)
gan.train(epochs=15001, batch_size=256, metrics_update=250, save_images=50, save_model=50)
