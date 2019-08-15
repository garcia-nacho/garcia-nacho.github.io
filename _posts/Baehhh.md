---
layout: post
title:  "VAAAAAE!"
---

*Do Androids Dream of Electric Sheep?* 
Let's talk about VAAAAAAAAAAAAAAAE!!! 

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

Let's explore some of the "sheeps" of the training dataset

You have to agree with me that they remotely resemble a sheep. It is also fair to admit that they are cotained in a square of 28x28 pixels so it is not very easy to draw something semi-decent in such reduced area. 

## The encoder

Now we can start creating the encoder part of the model which is in deed very similar to the one that I described [here](https://garcia-nacho.github.io/VAEs/) and that is indeed and adapted version of this one [here](https://tensorflow.rstudio.com/keras/articles/examples/variational_autoencoder.html). The encoder consists in a mixture of convolutional layers connected to a neural net to capture the features of the drawings.

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

## The latent space

The latent space is created using a lambda layer. Lambda layers in Keras are custom layers that are used to wrap a function. In our case we create two additional layers z_mean and z_log_var that we concatenate and transformed using the sampling function.

<pre><code>
z_mean <- hidden %>% layer_dense( units = latent_dim)
z_log_var <- hidden %>% layer_dense( units = latent_dim)


sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling, name="LatentSpace")
</code></pre>

## The decoder

Now we need to write the decoder part of the VAE and the decoder model itself so we can sample the latent space. This part is a bit unusual if you compare with the encoder, since it doesn't contain the pipe operator <code>%>%</code>, the reason for that is that we need to initialize all layers independently to share them with their weights between the different models (Decoder, Encoder-Decoder). Another option would be to create independent models with the same arquitecture and transfer the weights from the trained Encoder-Decoder to the other model.

We initialize the layers one by one and we bind them consecutiverly:

<pre><code>
#Initialization of layers

Output1<- layer_dense( units = round(intermediate_dim/4), activation = activation)
Output2<- layer_dropout(rate=0.1)
Output3<- layer_batch_normalization()
Output4<- layer_dense( units = round(intermediate_dim/2), activation = activation)
Output5<- layer_dense(units = intermediate_dim, activation = activation)
Output6<- layer_dense(units = prod(28,28,filters*8), activation = activation)
Output7<- layer_reshape(target_shape = c(28,28,filters*8))
Output8<- layer_conv_2d_transpose(filters=filters*8, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')
Output9<- layer_conv_2d_transpose(filters=filters*4, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')
Output10<-layer_conv_2d_transpose(filters=filters*2, kernel_size=c(4,4), activation=activation, padding='same',strides=c(1,1),data_format='channels_last')
Output11<-layer_conv_2d_transpose(filters=filters, kernel_size=c(4,4), activation=activation, padding='same',strides=c(2,2),data_format='channels_last') 
Output12<-layer_conv_2d(filters=1, kernel_size=c(4,4), activation="sigmoid", padding='same',strides=c(2,2),data_format='channels_last') 

#Concatenation of layers for the VAE

O1<-Output1(z)
O2<-Output2(O1)
O3<-Output3(O2)
O4<-Output4(O3)
O5<-Output5(O4)
O6<-Output6(O5)
O7<-Output7(O6)
O8<-Output8(O7)
O9<-Output9(O8)
O10<-Output10(O9)
O11<-Output11(O10)
O12<-Output12(O11)

#Concatenation of layers for the Decoder

decoder_input <- layer_input(shape = latent_dim)
O1D<-Output1(decoder_input)
O2D<-Output2(O1D)
O3D<-Output3(O2D)
O4D<-Output4(O3D)
O5D<-Output5(O4D)
O6D<-Output6(O5D)
O7D<-Output7(O6D)
O8D<-Output8(O7D)
O9D<-Output9(O8D)
O10D<-Output10(O9D)
O11D<-Output11(O10D)
O12D<-Output12(O11D)</code></pre>

Now we can create the two models and explore them:

<pre><code>
## variational autoencoder
vae <- keras_model(Input, O12)
summary(vae)

## Decoder
decoder<- keras_model(decoder_input, O12D)
summary(decoder)</code></pre>


<pre><code>
> summary(vae)
_________________________________________________________________________________________
Layer (type)                 Output Shape       Param #    Connected to                  
=========================================================================================
input_1 (InputLayer)         (None, 28, 28, 1)  0                                        
_________________________________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 10) 170        input_1[0][0]                 
_________________________________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 20) 3220       conv2d[0][0]                  
_________________________________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 40) 12840      conv2d_1[0][0]                
_________________________________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 80) 51280      conv2d_2[0][0]                
_________________________________________________________________________________________
flatten (Flatten)            (None, 62720)      0          conv2d_3[0][0]                
_________________________________________________________________________________________
dense (Dense)                (None, 100)        6272100    flatten[0][0]                 
_________________________________________________________________________________________
dropout (Dropout)            (None, 100)        0          dense[0][0]                   
_________________________________________________________________________________________
batch_normalization_v1 (Batc (None, 100)        400        dropout[0][0]                 
_________________________________________________________________________________________
dense_1 (Dense)              (None, 50)         5050       batch_normalization_v1[0][0]  
_________________________________________________________________________________________
dropout_1 (Dropout)          (None, 50)         0          dense_1[0][0]                 
_________________________________________________________________________________________
batch_normalization_v1_1 (Ba (None, 50)         200        dropout_1[0][0]               
_________________________________________________________________________________________
dense_2 (Dense)              (None, 25)         1275       batch_normalization_v1_1[0][0]
_________________________________________________________________________________________
dense_3 (Dense)              (None, 2)          52         dense_2[0][0]                 
_________________________________________________________________________________________
dense_4 (Dense)              (None, 2)          52         dense_2[0][0]                 
_________________________________________________________________________________________
concatenate (Concatenate)    (None, 4)          0          dense_3[0][0]                 
                                                           dense_4[0][0]                 
_________________________________________________________________________________________
LatentSpace (Lambda)         (None, 2)          0          concatenate[0][0]             
_________________________________________________________________________________________
dense_5 (Dense)              (None, 25)         75         LatentSpace[0][0]             
_________________________________________________________________________________________
dropout_2 (Dropout)          (None, 25)         0          dense_5[2][0]                 
_________________________________________________________________________________________
batch_normalization_v1_2 (Ba (None, 25)         100        dropout_2[2][0]               
_________________________________________________________________________________________
dense_6 (Dense)              (None, 50)         1300       batch_normalization_v1_2[2][0]
_________________________________________________________________________________________
dense_7 (Dense)              (None, 100)        5100       dense_6[2][0]                 
_________________________________________________________________________________________
dense_8 (Dense)              (None, 62720)      6334720    dense_7[2][0]                 
_________________________________________________________________________________________
reshape (Reshape)            (None, 28, 28, 80) 0          dense_8[2][0]                 
_________________________________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 80) 102480     reshape[2][0]                 
_________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 40) 51240      conv2d_transpose[2][0]        
_________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 20) 12820      conv2d_transpose_1[2][0]      
_________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 56, 56, 10) 3210       conv2d_transpose_2[2][0]      
_________________________________________________________________________________________
conv2d_4 (Conv2D)            (None, 28, 28, 1)  161        conv2d_transpose_3[2][0]      
=========================================================================================
Total params: 12,857,845
Trainable params: 12,857,495
Non-trainable params: 350
_________________________________________________________________________________________
</code></pre>


<pre><code>
> summary(decoder)
_________________________________________________________________________________________
Layer (type)                            Output Shape                       Param #       
=========================================================================================
input_5 (InputLayer)                    (None, 2)                          0             
_________________________________________________________________________________________
dense_5 (Dense)                         (None, 25)                         75            
_________________________________________________________________________________________
dropout_2 (Dropout)                     (None, 25)                         0             
_________________________________________________________________________________________
batch_normalization_v1_2 (BatchNormaliz (None, 25)                         100           
_________________________________________________________________________________________
dense_6 (Dense)                         (None, 50)                         1300          
_________________________________________________________________________________________
dense_7 (Dense)                         (None, 100)                        5100          
_________________________________________________________________________________________
dense_8 (Dense)                         (None, 62720)                      6334720       
_________________________________________________________________________________________
reshape (Reshape)                       (None, 28, 28, 80)                 0             
_________________________________________________________________________________________
conv2d_transpose (Conv2DTranspose)      (None, 28, 28, 80)                 102480        
_________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTranspose)    (None, 28, 28, 40)                 51240         
_________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTranspose)    (None, 28, 28, 20)                 12820         
_________________________________________________________________________________________
conv2d_transpose_3 (Conv2DTranspose)    (None, 56, 56, 10)                 3210          
_________________________________________________________________________________________
conv2d_4 (Conv2D)                       (None, 28, 28, 1)                  161           
=========================================================================================
Total params: 6,511,206
Trainable params: 6,511,156
Non-trainable params: 50
_________________________________________________________________________________________
</code></pre>


## KL-Anneling

Now that we have the models we are ready to create a custom loss function but before we do it I would like to introduce an additional concept, the Kullback-Leibler anneling (KL-Annealing). As I mentioned previously, the loss function is defined by assembling two sub-loss functions: The mean squared error that accounts for the fidelity of the model and the Kullback-Leibler divergence that forces the distribution of samples in the latent space; however these two terms are opposing each other and if under some circumstances (e.g. noisy input, high input variance) the trained model falls into a local minima called *posterior collapse*, when this happens the model *learns* to avoid the latent space so all samples have the same values for the latent  variables. I have personally experience it when training models for generation of faces. 
There are several solutions to prevent the posterior collapse and of them is the KL-Annealing. 

The KL-Annealing consist in the introduction of a weight to control the KL divergence during the training in a way that we train the model for some epochs just using the MSE and after certain epoch we start increasing the weight of the KL divergence to reach 1 after few epochs. 

Implementing KL-Annealing means that we need to modify the loss function dynamically during the training. This can be done in Keras/Tensoflow by using callbacks. The callbacks allow us to modify some parameters of the training process while it is happening. The documentation to do that in Python is pretty sparse but in R ins just negligible. So if you are interested in how to write callbacks for Keras in R keep reading.

The first thing that we need to do is to create a class-like element in R using the <code>R6</code> package

<pre><code>
#Custom callback KL-Annealing

KL.Ann <- R6::R6Class("KL.Ann",
#We create a KerasCallback-like class by inheriting the shape
                      inherit = KerasCallback,
                     
                      public = list(
#Here we define and assingn the variables that we are going to use in the class (weight)

                        losses = NULL,
                        params = NULL,
                        model = NULL,
                        weight = NULL,
                        
                        set_context = function(params = NULL, model = NULL) {
                          self$params <- params
                          self$model <- model
                          self$weight<-weight
                        },                        
#Here we define a function that runs at the end of each epoch
#We modify the weight value according to three parameters
#Epoch: The current epoch
#kl.start: The last epoch in which the KL weight is 0
#kl.steep: The length in epochs of the KL increase

                        on_epoch_end = function(epoch, logs = NULL) {
                          
                          if(epoch>kl.start){
                            new_weight<- min((epoch-kl.start)/kl.steep,1)
                            k_set_value(self$weight, new_weight)
                            print(paste("     ANNEALING KLD:", k_get_value(self$weight), sep = " "))
                            
                          }
                        }
                        
                      ))</code></pre>

Now we define the missing variables for the KL-Annealing.
<pre><code>
#Starting weight = 0
weight <- k_variable(0)
epochs <-30
kl.start <-10
kl.steep <- 10</code></pre>

## The loss function

Now we are really ready to create the custom loss function. The loss function consist of a function wrapping a proper loss function (<code>l.f</code> in the code). I say proper because it compares Y and Ŷ, the wrapping function takes the parameter weight and passes it to <code>l.f</code> before returning the function. 

<pre><code>loss<- function(weight){
  l.f<-function(x, x_decoded_mean){
    
    x <- k_flatten(x)
    x_decoded_mean <- k_flatten(x_decoded_mean)
    xent_loss <- 2.0 * dimensions[1]* dimensions[2]*loss_mean_squared_error(x, x_decoded_mean)
    kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
    xent_loss + weight*kl_loss}
  return(l.f)
}</code></pre>

Now we can compile the model and start the training process:

<pre><code>vae %>% compile(optimizer = "rmsprop", loss = loss(weight))</code></pre>

## Setting up the GPU to be used by Keras in R

Before describing the training process I would like to briefly describe the process of using the GPU of the computer instead the CPU, mainly because it took me quite few hours to have it up and running.

First. Why should you train on your GPU instead on your CPU? Mainly because of speed, even if you have a decent CPU (an Intel i7-8700 with 12 threads) a normal GPU (an NVIDIA GTX1060 embebded on a laptop) trains the very same model 15 times faster. 

Second. How do you do it? I have to tell you that it is not so trivial to do have the GPU working for training ML models in R, the documentation is pretty sparse. 

The main guidelines are [here](https://tensorflow.rstudio.com/tools/local_gpu.html)
Briefly, you have to download an install the *CUDA Toolkit* and the *cuDNN* libraries. It is **VERY IMPORTANT** that you download the proper versions of each library. In the guidelines they say that you need to download the *CUDA Toolkit v9* and the *cuDNN v7.0* but the truth is that those versions are a bit outdated so you can use more recent versions of the libraries, the problem is that not all versions are compatible among them and with TensorFlow. 

Here is my setting, if you download this versions it is going to work:

CUDA Toolkit: *Cuda compilation tools, release 10.0, V10.0.130*
cuDNN: *7.4.2*
Tensorflow: *1.13*

To install them you need to follow the instructions provided in the link above and you shouldn't get any problem. In case by the time you read these lines there are new versions of the libraaries remember that it is essential to check the compatiblity before doing any installation. 

![GPU](/images/GPU.png)


## Training the model

Now we are all set for the model training. 

<pre><code>
date<-as.character(date())
logs<-gsub(" ","_",date)
logs<-gsub(":",".",logs)
logs<-paste("logs/",logs,sep = "")

#Callbcks
tb<-callback_tensorboard(logs)
anneling <- KL.Ann$new()

history<-vae %>% fit(x= df,
                y= df,
                batch_size=batch_size,
                epoch=epochs,
                callbacks = list(tb,annealing),
                view_metrics=FALSE,
                shuffle=TRUE)</code></pre>

We can check the training process using tensorboard

<pre><code>tensorboard(logs)</code></pre>

and we can save the weights to avoid doing the training each time you run the script:

<pre><code>save_model_weights_hdf5(vae, "/home/nacho/VAE_Faces/Baeh.h5")</code></pre>

You can download my trained weights [here]() and load them into your model by doing this:

<pre><code></code></pre>


## Exploring the latent space

## Generation of new pictures

## Annomaly detection



## Going further

## Conclusions



http://anotherdatum.com/vae.html
