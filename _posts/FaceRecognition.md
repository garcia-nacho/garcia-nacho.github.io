---
layout: post
title:  "Face Recognition in R using Keras"
date: '2019-01-27 19:19:00'
---

### Introduction.
For millions of years human evolution has selected and improved the ability to recognize faces. 
Yes! We humans are very good recognizing faces. During the courses of our lives we remember around 5000 faces that we can then recall despiste poor ilumination conditions, major changes such as strong facial expressions, the presence of a beards, glasses, hats, etc... 
The ability to recgonize a face is a one of those hard-encoded capacities in our brains.  Nobody taught you how to recognize a face, it is something that you just can do without knowing how.
{: style="text-align: justify"}

Despite all this training a computer to reconize a face is an extremely complex task because faces are very simmilar, all faces follow the same patterns, they have two eyes, a nose and a mouth in the same area. What makes faces recognizable are the details, but how can we train a machine to find these details? 
Easy, using convolutional neural networks (CNN).
{: style="text-align: justify"}

### CNN

CNNs are special types of neural nets in which the data is processed by one or several convolutional layers before beeing fed into the *classical* neural network part. 
A convolutional layers processes each value of the data in differently depending on the neighboring data. If we are talking of images, the processed of each pixel of the image depends of the surrounding pixels and the rules applied, those rules are what we call filters. 
{: style="text-align: justify"}

The good thing about convolutional layers is that they have the propierty of being very good at finding patterns.
Let's see a very basic but intuitive example of how CNNs work using an image of with 36 pixels and a bit-depth of 2. One of the few shapes that it is possible to draw in such rudimentary image is a diagonal. There are two possible diagonal lines, ascending and descending and it is possible to devise a filter to find only the descending diagonals in the pictures ignoring the ascending ones:
![CNN1](/images/CNN1.png)

The output image is the result of multiplying the values of the input image by the filter in the different areas of the size of the filter. If the CNN finds the pattern it is looking the pattern is maintained. 
Otherwise it is filtered out:

![CNN2](/images/CNN2.png)

Of course CNN are much mor ecomplex than this. The filter can slide through the matrix position by position, the values obtained after applying the filter can be added together, etc... however the concept is exactly the same.

So the idea behind the face recognition algorithm is to send the images through several convolutional layers to find the patterns that are unique for each person and link them to the identity of the person through a neural net that we can train.  
### The Olivetti face database 

Since I just wanted to play around with this concepts testing different models, I needed a small database so I did't expend several hours training the model each time that I wanted to change a parameter. For that purpose the Olivetti database is fantastic, it consist of 400 B/W images from 10 different subjects. The images are 92 x 114 with 256 tones of grey (8 bits). You can dowload the original dataset from the AT&T lab [here](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) or a version ready to use [here](/images/faces.zip) (The version ready to use is exactly the same as the original one but all images are under the same folder with the number of the subject already encoded in the name of the file).

This is basically how the 10 images of a subject (#5 in this case) look like: 
![Faces](/images/10faces.jpg)

If you use the *ready-to-use* file you will find that the images have this name structure *SX-Y.pgm* where X is the number of the subject and Y the number of the picture.

### Loading of images into the R environment.
Here is the code that loads all the images into the R environment

<pre><code>library(imager)

Path<-"/home/nacho/ML/faces/"

files<-list.files(Path)
setwd(Path)

df.temp <- load.image(files[1])
df.temp<-as.matrix(df.temp)

df<-array(data = 0, dim = c(length(files),nrow(df.temp),ncol(df.temp),1))

for (i in 2:length(files)) {
  df.temp <- load.image(files[i])
  df.temp<-as.matrix(df.temp)
  df[i,,,1]<-df.temp
   }</code></pre>

First, we need the *imager* library and the path where the images are located. Next, in order to make the script working idependently the number size of the images, the first image in the folder is loaded, transformed into a matrix and acommodated in an array with 4 dimensions: Number of images, hight, width and 1 (since the images are b/w one channel is enough the describe the colour depth. Finally, all the images are sequentially loaded into the array with a *for()* loop.

By inspecting the array it is possible to see that all the grey values have already been normalized so the maximum possible white value is 1. To visualize the images in R I use the *image()* function like this:

<pre><code>image(df[1,,,],
        useRaster = TRUE,
        axes=FALSE,
        col = gray.colors(256, start = 0, end = 1, gamma = 2.2, alpha = NULL))</pre></code>

This code reconstruct the first image stored in the array (df[1,,,]) to produce this plot:
![Face1](/images/Face1.png)

It seems that the data is loaded in the array upside down, but we don't really care since this inversion is common to all the images in the array.
