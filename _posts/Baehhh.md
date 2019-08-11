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

To increase the probability of drawing a sheep P(X) we neeed to increase two factors: The area of the latent space that belongs to a sheep-region P(z) and the probability of drawing a sheep if we select any point in that sheep-region P(X|z):

$\[\[P(X)=\sum_i P(X|z_i)(z_i)\]]$

And this is exactly what a VAE does: It increases the P(z) by costraining it so it approximates a normal distribution and it increases P(X|z) by reducing the mean squared error between inputs and predictions during the training. You can read more of the statistical details [here](https://papers.nips.cc/paper/6528-variational-autoencoder-for-deep-learning-of-images-labels-and-captions.pdf). This is achieved in the loss functions function which has two terms, one to maximize P(z) and another to maximazie P(X|z)





http://anotherdatum.com/vae.html