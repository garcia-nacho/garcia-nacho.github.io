---
layout: post
title:  "VAAAAAE!"
---

*Do Androids Dream of Electric Sheep?* Let's talk about that...

## Introduction
My son's favourite animal is the sheep, he loves to bleat when he sees one. It doesn't matter if it is in a book, a video, a plastic figure or in real life.
He knows how a sheep looks like and although I have disscussed about this kind of topics [before](https://garcia-nacho.github.io/FaceRecognition/), I would like to bring this topic up again because I think that the human brain is just amazing. With just one year a human brain has all the neuronal circuits that allows it to learn to reconize patterns no matter how different they are, think how different it is to see a sheep in the fields when compared with a more or less accurated drawn of a sheep in a book. 
But not only that, a human brain has stored an idea of how a sheep looks like and it can extract that information to draw a sheep anytime. 
In this post we will see how computers can do exactly the same, recognize, store and draw a picture of a sheep. 
![Sheep](/images/sheeps.jpg)
<!--more-->
Image taken from [here](https://www.how-to-draw-funny-cartoons.com/cartoon-sheep.html)

## VAE's again, a bit more of the theory.
Although, in a previous post I already talked about [VAE's](https://garcia-nacho.github.io/VAEs/), I would like to discuss that type of models a bit more.
VAEs models are based on the idea that it is possible to create a representation of the *inputs* called latent space and that by sampling that latent space it is possible to create new items simmilar to the ones present in the data used for training , but different from all of them. Translating this to the sheeps example it would be like when you are asked to draw a sheep, you could draw 1000 sheeps all of them different and all of them different from other pictures of sheeps just by using the idea of a sheep that it is stored in your brain. 

To do something similar in a computer, we need an algorithm that can encode a set of instructions to recreate the drawing of a sheep. These sets of instructions are vectors, each indeed sets of vectors of N-dimensions, although it is very common for these vectors to have only two dimensions. The sets of all possible vectors lay in an N-dimesion space called latent space, so we can distinguish regions in the latent space, those encoding the instructions to draw a sheep and those which don't so the model needs to find these regions to make more likely that when we sample a vector from a sheep-region in the latent space the final product is a sheep. This can be seen as a probabilistic problem. 

To increase the probability of drawing a sheep P(X) we neeed to increase two factors: The probability of selecting an area of the latent space that belongs to a sheep-region P(z) and the probability of actually drawing a sheep if we select any point in that sheep-region P(X|z):

$\[\[P(X)=\sum_i P(X|z_i)(z_i)\]]$

And this is exactly what a VAE does: It increases the P(z) by costraining it and it increases P(X|z) by reducing the differences between inputs and predictions during the training. All this is achieved by using a custom loss function with two terms, one to maximize P(z) and another to maximize P(X|z). You can read more of the statistical details [here](https://papers.nips.cc/paper/6528-variational-autoencoder-for-deep-learning-of-images-labels-and-captions.pdf).

## The sheep dataset.

First we need to download a process the files. We are going to use a dataset of thousands of hand drawn sheep from Google's Quickdraw game dataset from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap). There are several formats but we are going to use .npy one. Once you have it downloaded you need to import the .npy files into R. 

The .npy is a Python-specific format, so in order to load the files into R you need to import the Python numpy library first. We do that with the R library *reticulate*. Then, you have to reshape the array so it has the shape c(number-of-samples, 28, 28) -the pictures are 28x28 pixels-. The I normalize the values dividing them by 255 and I save a copy of the files in a R-friendly format. Finally it is important to unload the reticulate package otherwise Keras doesn't work.

<pre><code>
 library(reticulate)
 np <- import("numpy")
  
 df<-np$load("/home/nacho/VAE_Faces/full_numpy_bitmap_sheep.npy")
 df<-array(df, dim = c(dim(df)[1], 28,28)) #Reshape 
 df<-df/255 #Normalization
 write.csv(df, "/home/nacho/VAE_Faces/datasheep.csv") #Checkpoint
 
 detach("package:reticulate", unload=TRUE)</code></pre>

If you have saved the files in a .csv format you don't need to run this part each time you play around with the dataset. You just load it, from the checkpoint. 

<pre><code>
library(keras)
library(imager)
library(ggplot2)

df<-read.csv("/home/nacho/VAE_Faces/datasheep.csv")
df<-as.matrix(df)
df<-array(as.numeric(df[,2:785]), dim = c(nrow(df),28,28,1))</pre></code>

You also have to reshape the array so it has the shape c(samples,28,28,1) -1 because there is only one channel-

## The encoder

Now we can start creating the encoder part of the model which is in deed very similar to the one that I described [here](https://garcia-nacho.github.io/VAEs/) and that is indeed and adapted version of this one [here]()

<pre><code>

#Model
reset_states(vae)
filters <- 10
intermediate_dim<-100
latent_dim<-2
epsilon_std <- 1
batch_size <- 8
epoch<-15
activation<-"relu"

dimensions<-dim(df)
dimensions<-dimensions[-1]

Input <- layer_input(shape = dimensions)

faces<- Input %>%
  layer_conv_2d(filters=filters, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')%>% 
  layer_conv_2d(filters=filters*2, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')%>%
  layer_conv_2d(filters=filters*4, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')%>%
  layer_conv_2d(filters=filters*8, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')%>%
  layer_flatten()

hidden <- faces %>% layer_dense( units = intermediate_dim, activation = activation) %>% 
  layer_dropout(0.1) %>% 
  layer_batch_normalization() %>% 
  layer_dense( units = round(intermediate_dim/2), activation = activation) %>% 
  layer_dropout(0.1) %>% 
  layer_batch_normalization() %>% 
  layer_dense( units = round(intermediate_dim/4), activation = activation)</code></pre>

## The decoder

## The loss function

## Training the model 

## Setting up the GPU to be used by Keras in R

## Exploring the latent space

## KL-Anneling

## Going further

## Conclusions



http://anotherdatum.com/vae.html
