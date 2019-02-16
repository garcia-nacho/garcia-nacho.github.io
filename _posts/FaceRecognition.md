---
layout: post
title:  "Face Recognition in R using Keras"
date: '2019-01-27 19:19:00'
---

### Introduction.
For millions of years human evolution has selected and improved the ability to recognize faces. Yes, we humans are very good recognizing faces. During the courses of our lives we remember around 5000 faces that we can recall despiste poor ilumination conditions, major changes such as strong facial expressions, the presence of a beards, glasses, hats, etc... 
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

