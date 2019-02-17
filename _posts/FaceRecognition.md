---
layout: post
title:  "Face Recognition in R using Keras"
date: '2019-01-27 19:19:00'
---

### Introduction.
For millions of years, human evolution has selected and improved the ability to recognize faces. 
Yes! We humans are very good at recognizing faces. During the courses of our lives, we remember around 5000 faces that we can then recall despite poor illumination conditions, major changes such as strong facial expressions, the presence of beards, glasses, hats, etc... 
The ability to recognize a face is one of those hard-encoded capacities in our brains.  Nobody taught you how to recognize a face, it is something that you just can do without knowing how.
{: style="text-align: justify"}
<!--more-->

Despite all this, training a computer to reconize a face is an extremely complex task because faces are indeed very simmilar, all faces follow the same patterns, they have two eyes, a nose and a mouth in the same area. What makes faces recognizable are the details, but how can we train a machine to find these details? 
Easy, using convolutional neural networks (CNN).
{: style="text-align: justify"}

### CNNs.
CNNs are special types of neural nets in which the data is processed by one or several convolutional layers before being fed into the *classical* neural network part. 
A convolutional layer processes each value of the data differently depending on the neighboring data. If we are talking of images, the processed of each pixel of the image depends on the surrounding pixels and the rules applied, those rules are what we call filters. 
{: style="text-align: justify"}

The good thing about convolutional layers is that they have the property of being very good at finding patterns.
Let's see a very basic but intuitive example of how CNNs work using an image of 36 pixels and a bit-depth of 2. One of the few shapes that it is possible to draw in such a rudimentary system is a diagonal line. There are two possible diagonal lines, ascending and descending and it is possible to devise a filter to find only the descending diagonals in the pictures ignoring the ascending ones:
{: style="text-align: justify"}
![CNN1](/images/CNN1.png)

The output image is the result of multiplying the values of the input image by the filter in the different areas of the size of the filter. If the CNN finds the pattern it is looking the pattern is maintained. 
Otherwise, it is filtered out:
{: style="text-align: justify"}
![CNN2](/images/CNN2.png)

Of course, CNNs are much more complex than this. The filter can slide through the matrix position by position, the values obtained after applying the filter can be added together, etc... however, the concept is exactly the same.
{: style="text-align: justify"}
So the idea behind a face recognition algorithm is to send the images through several convolutional layers to find the patterns that are unique for each person and link them to the identity of the person through a neural net that we can train. {: style="text-align: justify"}

### The Olivetti face database. 
Since I just wanted to play around with these concepts testing different models, I needed a small database so I didn't expend several hours training the model each time that I wanted to change a parameter. For that purpose the Olivetti database is fantastic, it consists of 400 B/W images from 10 different subjects. The images are 92 x 114 with 256 tones of grey (8 bits). You can download the original dataset from the AT&T lab [here](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) or a version ready to use [here](/images/faces.zip) (The version ready to use is exactly the same as the original one but all images are under the same folder with the number of the subject already encoded in the name of the file).
{: style="text-align: justify"}

This is basically how the 10 images for one of the subjects (#5 in this case) look like: 
![Faces](/images/10faces.jpg)

If you use the *ready-to-use* file you will find that the images have this name structure *SX-Y.pgm* where X is the number of the subject and Y the number of the picture. 

### Loading of images into the R environment.
Here is the code that loads all the images into the R environment

<pre><code>library(imager)

Path<-"/home/nacho/ML/faces/"

files<-list.files(Path)
setwd(Path)

df.temp <- load.image(files[1])
df.temp <- as.matrix(df.temp)

df<-array(data = 0, dim = c(length(files),nrow(df.temp),ncol(df.temp),1))

for (i in 2:length(files)) {
  df.temp <- load.image(files[i])
  df.temp <- as.matrix(df.temp)
  df[i,,,1] <- df.temp
   }</code></pre>

First, we need the *imager* library and the path where the images are located. Next, in order to make the script working idependently the number size of the images, the first image in the folder is loaded, transformed into a matrix and acommodated in an array with 4 dimensions: Number of images, hight, width and 1 (since the images are b/w one channel is enough the describe the colour depth. Finally, all the images are sequentially loaded into the array with a *for()* loop.

By inspecting the array it is possible to see that all the grey values have already been normalized so the maximum possible white value is 1. To visualize the images in R I use the *image()* function like this:

<pre><code>image(df[1,,,],
        useRaster = TRUE,
        axes=FALSE,
        col = gray.colors(256, start = 0, end = 1, gamma = 2.2, alpha = NULL))</pre></code>

This code reconstruct the first image stored in the array (df[1,,,]) to produce this plot:
![Face1](/images/Face1.png)

It seems that the data is loaded upside down in the array, but I don't really care since this inversion is common to all the images in the array.

Next, I wanted to prepare the dependent variable to match it with each image. To do that I extracted the number of the subject from the name of the file:

<pre><code>Y <- gsub(" .*","",files)
Y <- gsub("s","",Y)</code></pre>

and the last step before constructing the model is to divide the dataset in training and testing. I decided to use 9 images from each subject for the training proccess and the remaing image for testing purposes. The number in the sequence of pictures is the same for all subjects and it is randomly defined.  

<pre><code>#Training and test datasets
TestSeq<-(c(1:40)*10)+round(runif(1, min = 1, max = 10))-10

x_test <- df[TestSeq,,,1]
y_test <- Y[TestSeq]  
x_train <- df[-TestSeq,,,1]
y_train <- Y[-TestSeq]</code></pre>

### Constructing the model.

To construct the model I used Keras, which is a very flexible and powerful library for machine learning. Although most of the tutorials and examples using Keras are based on python, it is possible to use the library in R. The first time you use Keras you have to install the library:
<pre><code>
devtools::install_github("rstudio/keras")
library(keras)
install_keras()
</code></pre>
These commands will install Keras and TensorFlow, which is the core of Keras. Once Keras is installed it is possible to load it as the rest of the R libraries <code>library(keras)</code>

With Keras it is possible to create recursive models in which some layers are reused or models with several inputs/output but the most simple and common type of models are the sequential models. In a sequential model the data flows through the different layers to end up in the otuput. 

My face recogniztion model is a sequential model in which the data extracted from the images is transformed through the different layers to be matched in the last layer with the dependent variable while the weights are tuned to minimize the *loss function*.

<pre><code>#Model 
model <-keras_model_sequential()</code></pre>

The use of the *pipe operator %>%* is extremely useful to add the different layers that conform the model:
<pre><code>model %>%
  #CNN part
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",input_shape=c(92,112,1) ) %>%  
  layer_activation("relu") %>%  
  layer_conv_2d(filter=32 ,kernel_size=c(3,3))  %>%  
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>% </code></pre>
 
In the CNN part, the data enters into a 2D convolutional layer with 32 filters of 3x3 size. In this layer the shape of the data is also defined input_shape=c(width, height, channels). A common activation function in CNNs is [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). Then the data from the first CNN is processed by a second CNN to find patterns of a higher order. Finally, a *max pooling* layer is added for regularization. In this layer it occurs a downsampling of the data that also prevents overfitting. The pool size is 2x2, that means that the matrix is aggregated in a 2 by 2 manner and the maximum value of the 4 pixels is selected:
![MaxPool](/images/MaxPool.png)

Let'see now the neural network section.

 <pre><code>#Neural net part 
  layer_flatten() %>% 
  layer_dense(1024) %>%
  layer_activation("relu") %>% 
  layer_dense(128) %>% 
  layer_activation("relu") %>% 
  layer_dropout(0.3) %>% 
  layer_dense(40) %>% 
  layer_activation("softmax")</code></pre>

First the data from the CNN is flattened meaning that the array is reshaped into a vector with only one dimension. Then the data is sent through two *fully-connected* layers of 1024 and 128 neurons with ReLU as activation function. Next, a regularization layer is added to drop out 30% of the neurons. Finally a the output layer with the same number of units as elements to classify (40 in this case) is added, the activation function of this layer is softmax, that means that for each prediction the probability of belonging to each one of the 40 classes is calculated. 

It is possible to visualize the model using <code>summary(model)</code>

<pre><code>____________________________________________________________________________________________________________________________________
Layer (type)                                               Output Shape                                         Param #             
====================================================================================================================================
conv2d_3 (Conv2D)                                          (None, 92, 112, 32)                                  320                 
____________________________________________________________________________________________________________________________________
activation_6 (Activation)                                  (None, 92, 112, 32)                                  0                   
____________________________________________________________________________________________________________________________________
conv2d_4 (Conv2D)                                          (None, 90, 110, 32)                                  9248                
____________________________________________________________________________________________________________________________________
activation_7 (Activation)                                  (None, 90, 110, 32)                                  0                   
____________________________________________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)                             (None, 45, 55, 32)                                   0                   
____________________________________________________________________________________________________________________________________
flatten_2 (Flatten)                                        (None, 79200)                                        0                   
____________________________________________________________________________________________________________________________________
dense_4 (Dense)                                            (None, 1024)                                         81101824            
____________________________________________________________________________________________________________________________________
activation_8 (Activation)                                  (None, 1024)                                         0                   
____________________________________________________________________________________________________________________________________
dense_5 (Dense)                                            (None, 128)                                          131200              
____________________________________________________________________________________________________________________________________
activation_9 (Activation)                                  (None, 128)                                          0                   
____________________________________________________________________________________________________________________________________
dropout_2 (Dropout)                                        (None, 128)                                          0                   
____________________________________________________________________________________________________________________________________
dense_6 (Dense)                                            (None, 40)                                           5160                
____________________________________________________________________________________________________________________________________
activation_10 (Activation)                                 (None, 40)                                           0                   
====================================================================================================================================
Total params: 81,247,752
Trainable params: 81,247,752
Non-trainable params: 0
____________________________________________________________________________________________________________________________________</code></pre>


The next step is to compile the model with the optimization parameters and the loss function.

<pre><code>opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )
compile(model,optimizer = opt, loss = 'categorical_crossentropy' )</code></pre>

For the optimization I am using the Adam optimizer which is one of the state of the art algorithms for weight tuning commonly used in image classification. The loss function is the cross entropy. [Here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) you can read more about the Adam optimizer and [here](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) an interesting reading where the author explains the cross entropy. 

The final step is to train the model:

<pre><code>history<-model %>% fit(x_train, y_train,
              batch_size=10,
              epoch=10,
              validation_data = list(x_test, y_test),
              callbacks = callback_tensorboard("logs/run_a"),
              view_metrics=TRUE,
              shuffle=TRUE)</code></pre>

The important parameters are the batch size which is the number that samples that are processed before the model is updated and the number of epochs which is the number of times that the model goes through the entire dataset. I started setting both to 10. 

After 10 minutes of training in my slow computer (i5-4200U/4Gb) the model achieves a 92.5% prediction accuracy which considering that the model only had 9 samples to train and it only run through the whole dataset 10 times it's pretty good. The model only failed predicting 3 faces; moreover, by looking at the shape of the training curve it is possible to predict that more epochs would lead to better predictions.

![Loss](/images/RPlotLoss.png)

An amazing feature of Keras/TensorFlow is the possibility of using *TensorBoard* which is a kind of front-end for TensorFlow with a lot of information presented in a beautiful way

![TensorBoard](/images/TensorBoard.png)

## Conclusions.

In this post we have seen a very basic example of image recognition and classification in R using Keras. Using this a playground it is possible to implement more advanced models to solve complex classification tasks. 
I hope you enjoy it as much as I did. 

## Bibliography and Sources of Inspiration
[Keras for R](https://keras.rstudio.com/)
[How to implement Deep Learning in R using Keras and Tensorflow](https://towardsdatascience.com/how-to-implement-deep-learning-in-r-using-keras-and-tensorflow-82d135ae4889)


